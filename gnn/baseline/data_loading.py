import pandas as pd
import numpy as np
import torch
import logging
from data_util import GraphData, HeteroData, z_norm, create_hetero_obj


def _time_split(timestamps_np):
    """Timestamp 누적 행 수 기준 60/20/20 split.

    동일 timestamp의 거래는 쪼개지 않는다.
    반환: tr_inds, val_inds, te_inds (torch.LongTensor)
    """
    n = len(timestamps_np)
    train_cut = n * 0.60
    val_cut   = n * 0.80

    ts_series = pd.Series(timestamps_np)
    ts_counts = (
        ts_series.value_counts()
        .sort_index()
        .reset_index()
    )
    ts_counts.columns = ['Timestamp', 'count']
    ts_counts['cum_count'] = ts_counts['count'].cumsum()
    ts_counts['split'] = np.where(
        ts_counts['cum_count'] <= train_cut, 'train',
        np.where(ts_counts['cum_count'] <= val_cut, 'val', 'test')
    )

    ts_to_split = dict(zip(ts_counts['Timestamp'], ts_counts['split']))
    row_splits = ts_series.map(ts_to_split).values

    tr_inds  = torch.where(torch.tensor(row_splits == 'train'))[0]
    val_inds = torch.where(torch.tensor(row_splits == 'val'))[0]
    te_inds  = torch.where(torch.tensor(row_splits == 'test'))[0]

    return tr_inds, val_inds, te_inds


def get_data(args, data_config):
    '''Loads the AML transaction data from preprocessed parquet (ml_exp00.parquet).

    1. The data is loaded from the csv and the necessary features are chosen.
    2. The data is split into training, validation and test data (timestamp 누적 행 수 기준 60/20/20).
    3. PyG Data objects are created with the respective data splits.
    '''

    from pathlib import Path
    parquet_file = Path(data_config['paths']['aml_data']) / args.data / 'ml_exp00_sample10k.parquet'
    df_edges = pd.read_parquet(parquet_file)

    logging.info(f'Available Edge Features: {df_edges.columns.tolist()}')

    # timestamp(datetime) → 경과 초 (ports/time-delta 내부 계산용, 모델 입력 아님)
    ts = pd.to_datetime(df_edges['timestamp'])
    ts_elapsed = (ts - ts.min()).dt.total_seconds()

    from_id, to_id, max_n_id = _build_node_map(df_edges)
    # 미등장 노드(-1) → unknown token 인덱스(max_n_id)로 치환
    from_id = np.where(from_id == -1, max_n_id, from_id)
    to_id   = np.where(to_id   == -1, max_n_id, to_id)
    # node feature matrix: train 노드 + unknown token 슬롯(마지막 행)
    df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id + 1), 'Feature': np.ones(max_n_id + 1)})
    timestamps = torch.tensor(ts_elapsed.to_numpy()).float()
    y = torch.LongTensor(df_edges['label'].to_numpy())

    logging.info(f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
    logging.info(f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}")
    logging.info(f"Number of transactions = {df_edges.shape[0]}")

    amount_recv_col = (
        'amount_received__current__log1p'
        if 'amount_received__current__log1p' in df_edges.columns
        else 'amount__current__log1p'
    )
    if amount_recv_col == 'amount__current__log1p':
        logging.warning('amount_received__current__log1p not found -> falling back to amount__current__log1p')
    edge_features = [
        'amount__current__log1p',
        'cat__payment_currency__code',
        amount_recv_col,
        'cat__receiving_currency__code',
        'cat__payment_format__code',
        'time__row__hour',
        'time__row__dayofweek',
        'time__row__is_weekend',
    ]
    node_features = ['Feature']

    logging.info(f'Edge features being used: {edge_features}')
    logging.info(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

    # categorical edge feature -1 → cardinality(unknown token) 치환
    cat_cardinality = _load_cardinality(Path(data_config['paths']['aml_data']) / args.data)
    CAT_COL_MAP = {
        'cat__payment_currency__code':  'payment_currency',
        'cat__receiving_currency__code': 'receiving_currency',
        'cat__payment_format__code':     'payment_format',
    }
    for feat_col, raw_col in CAT_COL_MAP.items():
        if feat_col in df_edges.columns and raw_col in cat_cardinality:
            df_edges[feat_col] = df_edges[feat_col].replace(-1, int(cat_cardinality[raw_col]))

    x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
    edge_index = torch.LongTensor(np.stack([from_id, to_id]))
    edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()

    # Train/Val/Test split (timestamp 누적 행 수 기준 60/20/20)
    tr_inds, val_inds, te_inds = _time_split(df_edges['Timestamp'].to_numpy())

    logging.info(f"Total train samples: {tr_inds.shape[0] / y.shape[0] * 100:.2f}% || IR: {y[tr_inds].float().mean() * 100:.2f}%")
    logging.info(f"Total val samples:   {val_inds.shape[0] / y.shape[0] * 100:.2f}% || IR: {y[val_inds].float().mean() * 100:.2f}%")
    logging.info(f"Total test samples:  {te_inds.shape[0] / y.shape[0] * 100:.2f}% || IR: {y[te_inds].float().mean() * 100:.2f}%")

    tr_x, val_x, te_x = x, x, x
    e_tr = tr_inds.numpy()
    e_val = np.concatenate([tr_inds, val_inds])

    tr_edge_index,  tr_edge_attr,  tr_y,  tr_edge_times  = edge_index[:,e_tr],  edge_attr[e_tr],  y[e_tr],  timestamps[e_tr]
    val_edge_index, val_edge_attr, val_y, val_edge_times = edge_index[:,e_val], edge_attr[e_val], y[e_val], timestamps[e_val]
    te_edge_index,  te_edge_attr,  te_y,  te_edge_times  = edge_index,          edge_attr,        y,        timestamps

    tr_data  = GraphData(x=tr_x,  y=tr_y,  edge_index=tr_edge_index,  edge_attr=tr_edge_attr,  timestamps=tr_edge_times)
    val_data = GraphData(x=val_x, y=val_y, edge_index=val_edge_index, edge_attr=val_edge_attr, timestamps=val_edge_times)
    te_data  = GraphData(x=te_x,  y=te_y,  edge_index=te_edge_index,  edge_attr=te_edge_attr,  timestamps=te_edge_times)

    if args.ports:
        logging.info(f"Start: adding ports")
        tr_data.add_ports()
        val_data.add_ports()
        te_data.add_ports()
        logging.info(f"Done: adding ports")
    if args.tds:
        logging.info(f"Start: adding time-deltas")
        tr_data.add_time_deltas()
        val_data.add_time_deltas()
        te_data.add_time_deltas()
        logging.info(f"Done: adding time-deltas")

    tr_data.x = val_data.x = te_data.x = z_norm(tr_data.x)
    if not args.model == 'rgcn':
        tr_data.edge_attr, val_data.edge_attr, te_data.edge_attr = z_norm(tr_data.edge_attr), z_norm(val_data.edge_attr), z_norm(te_data.edge_attr)
    else:
        tr_data.edge_attr[:, :-1], val_data.edge_attr[:, :-1], te_data.edge_attr[:, :-1] = z_norm(tr_data.edge_attr[:, :-1]), z_norm(val_data.edge_attr[:, :-1]), z_norm(te_data.edge_attr[:, :-1])

    if args.reverse_mp:
        tr_data  = create_hetero_obj(tr_data.x,  tr_data.y,  tr_data.edge_index,  tr_data.edge_attr,  tr_data.timestamps,  args)
        val_data = create_hetero_obj(val_data.x, val_data.y, val_data.edge_index, val_data.edge_attr, val_data.timestamps, args)
        te_data  = create_hetero_obj(te_data.x,  te_data.y,  te_data.edge_index,  te_data.edge_attr,  te_data.timestamps,  args)

    logging.info(f'train data object: {tr_data}')
    logging.info(f'validation data object: {val_data}')
    logging.info(f'test data object: {te_data}')

    # 각 data 객체는 자신의 엣지만 보유하므로 로컬 0-based 인덱스로 반환
    return tr_data, val_data, te_data, \
           torch.arange(len(e_tr)), \
           torch.arange(len(e_val)), \
           torch.arange(len(e_te))
