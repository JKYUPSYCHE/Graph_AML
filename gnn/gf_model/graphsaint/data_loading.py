import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from data_util import GraphData, z_norm, create_hetero_obj

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

NODE_FEATURE_COLS = [
    'train_only_pagerank',
    'train_only_in_degree',
    'train_only_out_degree',
    'train_only_in_amount_sum',
    'train_only_out_amount_sum',
]

BASE_EDGE_FEATURES = ['log_amount', 'currency', 'payment_format']


def _split_indices(split_col: pd.Series):
    arr = split_col.to_numpy()
    tr_inds  = torch.where(torch.tensor(arr == 'train'))[0]
    val_inds = torch.where(torch.tensor(arr == 'val'))[0]
    te_inds  = torch.where(torch.tensor(arr == 'test'))[0]
    return tr_inds, val_inds, te_inds


def _encode_categoricals(df_edges, tr_inds_np):
    for col in ['currency', 'payment_format']:
        if col not in df_edges.columns:
            continue
        le = LabelEncoder()
        le.fit(df_edges.iloc[tr_inds_np][col].astype(str))
        n_unknown = len(le.classes_)
        arr = df_edges[col].astype(str).to_numpy()
        known = np.isin(arr, le.classes_)
        df_edges[col] = np.where(
            known,
            le.transform(np.where(known, arr, le.classes_[0])),
            n_unknown,
        )
    return df_edges


def get_data(args, data_config):
    """
    formatted_transactions_gf.csv + gf.parquet 22개 피처를 합쳐 로딩.

    엣지 피처 구성:
        [log_amount, currency(encoded), payment_format(encoded)]  ← base 3개
        + [GF_FEATURES 22개]                                       ← gf.parquet
        + ports (args.ports=True 시)
        + time_deltas (args.tds=True 시)

    분할 구조 (GraphSAINT용):
        tr_data  = train 엣지만
        val_data = val 엣지만
        te_data  = test 엣지만
        반환 인덱스: torch.arange(len) — 배치 내 te_inds 필터링용
    """
    gnn_dir = Path(data_config['paths']['gnn_inputs'])
    gf_path = Path(data_config['paths']['gf_parquet'])

    df_edges = pd.read_csv(gnn_dir / 'formatted_transactions_gf.csv')
    logging.info(f'Available Edge Features: {df_edges.columns.tolist()}')

    # gf.parquet join on tx_id
    gf = pd.read_parquet(gf_path, columns=['tx_id'] + GF_FEATURES)
    df_edges['tx_id'] = df_edges['tx_id'].astype(int)
    gf['tx_id'] = gf['tx_id'].astype(int)
    df_edges = df_edges.merge(gf, on='tx_id', how='left')
    df_edges[GF_FEATURES] = df_edges[GF_FEATURES].fillna(0.0)
    logging.info(f'GF features joined: {len(GF_FEATURES)} features')

    # timestamps: elapsed seconds since min (ports/time-delta 계산용)
    ts_raw = pd.to_datetime(df_edges['timestamp'])
    ts_elapsed = (ts_raw - ts_raw.min()).dt.total_seconds()
    timestamps = torch.tensor(ts_elapsed.to_numpy()).float()

    y = torch.LongTensor(df_edges['label'].to_numpy())
    logging.info(f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y)/len(y)*100:.2f}%")
    logging.info(f"Number of transactions = {len(df_edges)}")

    # split (split 컬럼: train / val / test)
    tr_inds, val_inds, te_inds = _split_indices(df_edges['split'])
    logging.info(f"Train: {len(tr_inds)/len(y)*100:.2f}% | IR: {y[tr_inds].float().mean()*100:.2f}%")
    logging.info(f"Val  : {len(val_inds)/len(y)*100:.2f}% | IR: {y[val_inds].float().mean()*100:.2f}%")
    logging.info(f"Test : {len(te_inds)/len(y)*100:.2f}% | IR: {y[te_inds].float().mean()*100:.2f}%")

    # categorical encoding: train 기준 fit, 미등장 카테고리 → unknown token
    df_edges = _encode_categoricals(df_edges, tr_inds.numpy())

    edge_features = BASE_EDGE_FEATURES + GF_FEATURES
    logging.info(f'Edge features: {len(edge_features)} total ({len(BASE_EDGE_FEATURES)} base + {len(GF_FEATURES)} GF)')

    # node features: account_node_features.csv (train-only PageRank/degree/amount)
    df_nf = pd.read_csv(gnn_dir / 'account_node_features.csv')
    n_nodes = len(df_nf)
    df_nf[NODE_FEATURE_COLS] = df_nf[NODE_FEATURE_COLS].fillna(0.0)
    # amount 컬럼은 스케일 차이가 커서 log1p 적용
    df_nf['train_only_in_amount_sum']  = np.log1p(df_nf['train_only_in_amount_sum'].to_numpy())
    df_nf['train_only_out_amount_sum'] = np.log1p(df_nf['train_only_out_amount_sum'].to_numpy())
    logging.info(f'Node features: {NODE_FEATURE_COLS} (amount cols: log1p applied)')
    logging.info(f"Number of nodes = {n_nodes}")

    x          = torch.tensor(df_nf[NODE_FEATURE_COLS].to_numpy()).float()
    edge_index = torch.LongTensor(df_edges[['from_idx', 'to_idx']].to_numpy().T)
    edge_attr  = torch.tensor(df_edges[edge_features].to_numpy()).float()

    e_tr  = tr_inds.numpy()
    e_val = val_inds.numpy()
    e_te  = te_inds.numpy()

    tr_data  = GraphData(x=x, y=y[e_tr],  edge_index=edge_index[:, e_tr],  edge_attr=edge_attr[e_tr],  timestamps=timestamps[e_tr])
    val_data = GraphData(x=x, y=y[e_val], edge_index=edge_index[:, e_val], edge_attr=edge_attr[e_val], timestamps=timestamps[e_val])
    te_data  = GraphData(x=x, y=y[e_te],  edge_index=edge_index[:, e_te],  edge_attr=edge_attr[e_te],  timestamps=timestamps[e_te])

    if args.ports:
        logging.info("Start: adding ports")
        tr_data.add_ports()
        val_data.add_ports()
        te_data.add_ports()
        logging.info("Done: adding ports")
    if args.tds:
        logging.info("Start: adding time-deltas")
        tr_data.add_time_deltas()
        val_data.add_time_deltas()
        te_data.add_time_deltas()
        logging.info("Done: adding time-deltas")

    tr_data.x = val_data.x = te_data.x = z_norm(tr_data.x)
    if not args.model == 'rgcn':
        tr_data.edge_attr  = z_norm(tr_data.edge_attr)
        val_data.edge_attr = z_norm(val_data.edge_attr)
        te_data.edge_attr  = z_norm(te_data.edge_attr)
    else:
        tr_data.edge_attr[:, :-1]  = z_norm(tr_data.edge_attr[:, :-1])
        val_data.edge_attr[:, :-1] = z_norm(val_data.edge_attr[:, :-1])
        te_data.edge_attr[:, :-1]  = z_norm(te_data.edge_attr[:, :-1])

    if args.reverse_mp:
        tr_data  = create_hetero_obj(tr_data.x,  tr_data.y,  tr_data.edge_index,  tr_data.edge_attr,  tr_data.timestamps,  args)
        val_data = create_hetero_obj(val_data.x, val_data.y, val_data.edge_index, val_data.edge_attr, val_data.timestamps, args)
        te_data  = create_hetero_obj(te_data.x,  te_data.y,  te_data.edge_index,  te_data.edge_attr,  te_data.timestamps,  args)

    logging.info(f'train data: {tr_data}')
    logging.info(f'val data  : {val_data}')
    logging.info(f'test data : {te_data}')

    return tr_data, val_data, te_data, \
           torch.arange(len(e_tr)), \
           torch.arange(len(e_val)), \
           torch.arange(len(e_te))
