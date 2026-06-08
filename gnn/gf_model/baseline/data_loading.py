import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from data_util import GraphData, HeteroData, z_norm, create_hetero_obj

GF_FEATURES = [
    'recency__sender__out__seconds_since_last',
    'recency__receiver__in__seconds_since_last',
    'flag__sender__out__is_first_tx',
    'flag__receiver__in__is_first_tx',
    'timehist__sender__out__tx_count__count__w1h',
    'timehist__sender__out__amount__sum__w1d',
    'timehist__sender__out__amount__max__w1d',
    'timehist__sender__out__amount__std__w7d',
    'timehist__receiver__in__tx_count__count__w1h',
    'timehist__receiver__in__amount__sum__w7d',
    'timehist__sender__all__tx_count__cum__whist',
    'timehist__receiver__all__tx_count__cum__whist',
    'fanout__sender__out__counterparty__nunique__w1d',
    'fanout__sender__out__counterparty__nunique__w7d',
    'fanin__receiver__in__counterparty__nunique__w1d',
    'fanout__sender__out__counterparty_amount__top1_share__w1d',
    'fanin__receiver__in__counterparty_amount__top1_share__w1d',
    'bankfan__sender__out__to_bank__nunique__w7d',
    'pair__sender_receiver__forward__tx_count__count__w1h',
    'pair__sender_receiver__forward__tx_count__count__w1d',
    'accountstats__receiver__out__amount__sum__w1d',
    'accountstats__sender__in__amount__sum__w1d',
]


def _split_indices(split_col: pd.Series):
    """split 컬럼(train/val/test) 기준으로 index 반환."""
    arr = split_col.to_numpy()
    tr_inds  = torch.where(torch.tensor(arr == 'train'))[0]
    val_inds = torch.where(torch.tensor(arr == 'val'))[0]
    te_inds  = torch.where(torch.tensor(arr == 'test'))[0]
    return tr_inds, val_inds, te_inds


def _encode_categoricals(df_edges, tr_inds):
    """train rows 기준으로 LabelEncoder fit 후 전체 적용. 미등장 카테고리 → n_unique_train."""
    cat_cols = [
        'cat__payment_currency__code',
        'cat__receiving_currency__code',
        'cat__payment_format__code',
    ]
    for col in cat_cols:
        if col not in df_edges.columns:
            continue
        le = LabelEncoder()
        le.fit(df_edges.iloc[tr_inds][col].astype(str))
        n_unique = len(le.classes_)

        arr = df_edges[col].astype(str).to_numpy()
        known = np.isin(arr, le.classes_)
        encoded = np.where(
            known,
            le.transform(np.where(known, arr, le.classes_[0])),
            n_unique,
        )
        df_edges[col] = encoded
    return df_edges


def get_data(args, data_config):
    '''gnn/baseline-v2 코드 기반 + gf.parquet 22개 피처 추가.

    엣지 피처:
        [amount__current__log1p, cat__payment_currency__code,
         cat__receiving_currency__code, cat__payment_format__code,
         time__row__hour, time__row__dayofweek, time__row__is_weekend]  ← base 7개
        + [GF_FEATURES 22개]                                             ← gf.parquet
        + ports (args.ports=True 시)
        + time_deltas (args.tds=True 시)
    '''
    gnn_dir = Path(data_config['paths']['gnn_inputs'])
    gf_path = Path(data_config['paths']['gf_parquet'])

    df_edges = pd.read_csv(gnn_dir / 'formatted_transactions_gf.csv')
    df_ts    = pd.read_csv(gnn_dir / 'formatted_transactions.csv', usecols=['timestamp'])
    mapping  = pd.read_csv(gnn_dir / 'account_mapping.csv')

    logging.info(f'Available Edge Features: {df_edges.columns.tolist()}')

    # gf.parquet join: formatted_transactions_gf.csv에 tx_id 없으므로
    # ml.parquet에서 tx_id 로드 후 positional join
    parquet_path = Path(data_config['paths']['aml_data']) / 'ml' / 'ml.parquet'
    ml_tx = pd.read_parquet(parquet_path, columns=['tx_id'])
    assert len(ml_tx) == len(df_edges), (
        f"Row count mismatch: ml={len(ml_tx)}, formatted_transactions_gf={len(df_edges)}"
    )
    gf = pd.read_parquet(gf_path, columns=['tx_id'] + GF_FEATURES)
    gf['tx_id'] = gf['tx_id'].astype(int)
    ml_tx['tx_id'] = ml_tx['tx_id'].astype(int)
    gf = ml_tx.merge(gf, on='tx_id', how='left')
    df_edges[GF_FEATURES] = gf[GF_FEATURES].fillna(0.0).to_numpy()
    logging.info(f'GF features joined via ml tx_id: {len(GF_FEATURES)} features')

    # timestamp → 경과 초 (ports/time-delta 내부 계산용, 모델 입력 아님)
    ts = pd.to_datetime(df_ts['timestamp'])
    ts_elapsed = (ts - ts.min()).dt.total_seconds()
    timestamps = torch.tensor(ts_elapsed.to_numpy()).float()

    # from_id/to_id → node_idx (account_mapping.csv 기준)
    id_to_idx = dict(zip(mapping['account_id'].astype(str), mapping['node_idx']))
    max_n_id  = int(mapping['node_idx'].max()) + 1

    def _to_str_id(series):
        try:
            return series.astype(float).astype(int).astype(str)
        except (ValueError, TypeError):
            return series.astype(str)

    from_id = _to_str_id(df_edges['from_id']).map(id_to_idx).fillna(max_n_id).astype(int).to_numpy()
    to_id   = _to_str_id(df_edges['to_id']).map(id_to_idx).fillna(max_n_id).astype(int).to_numpy()

    n_unk_from = int((from_id == max_n_id).sum())
    n_unk_to   = int((to_id   == max_n_id).sum())
    logging.info(f"Unknown from_id: {n_unk_from}/{len(from_id)} ({n_unk_from/len(from_id)*100:.1f}%)")
    logging.info(f"Unknown to_id:   {n_unk_to}/{len(to_id)} ({n_unk_to/len(to_id)*100:.1f}%)")

    n_nodes  = max_n_id + 1
    df_nodes = pd.DataFrame({'Feature': np.ones(n_nodes)})

    y = torch.LongTensor(df_edges['label'].to_numpy())

    logging.info(f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
    logging.info(f"Number of nodes = {n_nodes}")
    logging.info(f"Number of transactions = {len(df_edges)}")

    tr_inds, val_inds, te_inds = _split_indices(df_edges['split'])

    logging.info(f"Total train samples: {tr_inds.shape[0] / y.shape[0] * 100:.2f}% || IR: {y[tr_inds].float().mean() * 100:.2f}%")
    logging.info(f"Total val samples:   {val_inds.shape[0] / y.shape[0] * 100:.2f}% || IR: {y[val_inds].float().mean() * 100:.2f}%")
    logging.info(f"Total test samples:  {te_inds.shape[0] / y.shape[0] * 100:.2f}% || IR: {y[te_inds].float().mean() * 100:.2f}%")

    df_edges = _encode_categoricals(df_edges, tr_inds.numpy())

    base_edge_features = [
        'amount__current__log1p',
        'cat__payment_currency__code',
        'cat__receiving_currency__code',
        'cat__payment_format__code',
        'time__row__hour',
        'time__row__dayofweek',
        'time__row__is_weekend',
    ]
    edge_features = base_edge_features + GF_FEATURES
    node_features = ['Feature']

    logging.info(f'Edge features: {len(edge_features)} total ({len(base_edge_features)} base + {len(GF_FEATURES)} GF)')
    logging.info(f'Node features: {node_features} (placeholder all 1s)')

    x          = torch.tensor(df_nodes[node_features].to_numpy()).float()
    edge_index = torch.LongTensor(np.stack([from_id, to_id]))
    edge_attr  = torch.tensor(df_edges[edge_features].to_numpy()).float()

    e_tr  = tr_inds.numpy()
    e_val = val_inds.numpy()
    e_te  = np.arange(len(df_edges))

    tr_edge_index,  tr_edge_attr,  tr_y,  tr_edge_times  = edge_index[:, e_tr],  edge_attr[e_tr],  y[e_tr],  timestamps[e_tr]
    val_edge_index, val_edge_attr, val_y, val_edge_times = edge_index[:, e_val], edge_attr[e_val], y[e_val], timestamps[e_val]
    te_edge_index,  te_edge_attr,  te_y,  te_edge_times  = edge_index[:, e_te],  edge_attr[e_te],  y[e_te],  timestamps[e_te]

    tr_data  = GraphData(x=x, y=tr_y,  edge_index=tr_edge_index,  edge_attr=tr_edge_attr,  timestamps=tr_edge_times)
    val_data = GraphData(x=x, y=val_y, edge_index=val_edge_index, edge_attr=val_edge_attr, timestamps=val_edge_times)
    te_data  = GraphData(x=x, y=te_y,  edge_index=te_edge_index,  edge_attr=te_edge_attr,  timestamps=te_edge_times)

    if args.ports:
        logging.info("Start: adding ports")
        rp = getattr(args, 'reverse_ports', True)
        tr_data.add_ports(reverse_ports=rp)
        val_data.add_ports(reverse_ports=rp)
        te_data.add_ports(reverse_ports=rp)
        logging.info("Done: adding ports")
    if args.tds:
        logging.info("Start: adding time-deltas")
        tr_data.add_time_deltas()
        val_data.add_time_deltas()
        te_data.add_time_deltas()
        logging.info("Done: adding time-deltas")

    tr_data.x = val_data.x = te_data.x = z_norm(tr_data.x)
    if not args.model == 'rgcn':
        tr_mean = tr_data.edge_attr.mean(0).unsqueeze(0)
        tr_std  = tr_data.edge_attr.std(0).unsqueeze(0)
        tr_std  = torch.where(tr_std == 0, torch.ones_like(tr_std), tr_std)
        tr_data.edge_attr  = (tr_data.edge_attr  - tr_mean) / tr_std
        val_data.edge_attr = (val_data.edge_attr - tr_mean) / tr_std
        te_data.edge_attr  = (te_data.edge_attr  - tr_mean) / tr_std
    else:
        tr_mean = tr_data.edge_attr[:, :-1].mean(0).unsqueeze(0)
        tr_std  = tr_data.edge_attr[:, :-1].std(0).unsqueeze(0)
        tr_std  = torch.where(tr_std == 0, torch.ones_like(tr_std), tr_std)
        tr_data.edge_attr[:, :-1]  = (tr_data.edge_attr[:, :-1]  - tr_mean) / tr_std
        val_data.edge_attr[:, :-1] = (val_data.edge_attr[:, :-1] - tr_mean) / tr_std
        te_data.edge_attr[:, :-1]  = (te_data.edge_attr[:, :-1]  - tr_mean) / tr_std

    if args.reverse_mp:
        tr_data  = create_hetero_obj(tr_data.x,  tr_data.y,  tr_data.edge_index,  tr_data.edge_attr,  tr_data.timestamps,  args)
        val_data = create_hetero_obj(val_data.x, val_data.y, val_data.edge_index, val_data.edge_attr, val_data.timestamps, args)
        te_data  = create_hetero_obj(te_data.x,  te_data.y,  te_data.edge_index,  te_data.edge_attr,  te_data.timestamps,  args)

    logging.info(f'train data object: {tr_data}')
    logging.info(f'validation data object: {val_data}')
    logging.info(f'test data object: {te_data}')

    return tr_data, val_data, te_data, \
           torch.arange(len(e_tr)), \
           torch.arange(len(e_val)), \
           te_inds
