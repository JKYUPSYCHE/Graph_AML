import pandas as pd
import numpy as np
import torch
import logging
import itertools
from data_util import GraphData, z_norm, create_hetero_obj

# gf.parquet에서 사용할 22개 피처 (카탈로그 기준)
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


def get_data(args, data_config):
    """
    기존 formatted_transactions.csv 엣지 피처 + gf.parquet 22개 피처를 합쳐서 로딩.

    엣지 피처 구성:
        [Timestamp, Amount Received, Received Currency, Payment Format]  ← 기존 4개
        + [GF_FEATURES 22개]                                              ← gf.parquet
        + ports (args.ports=True 시)
        + time_deltas (args.tds=True 시)
    """
    transaction_file = f"{data_config['paths']['gnn_inputs']}/formatted_transactions.csv"
    gf_path = data_config['paths']['gf_parquet']

    df_edges = pd.read_csv(transaction_file)
    logging.info(f'Available Edge Features (formatted_transactions): {df_edges.columns.tolist()}')

    # gf.parquet 로드 후 tx_id(str) → EdgeID(int) 변환하여 join
    gf = pd.read_parquet(gf_path, columns=['tx_id'] + GF_FEATURES)
    gf['tx_id'] = gf['tx_id'].astype(int)
    df_edges = df_edges.merge(gf, left_on='EdgeID', right_on='tx_id', how='left')
    df_edges[GF_FEATURES] = df_edges[GF_FEATURES].fillna(0.0)

    logging.info(f'GF features joined: {GF_FEATURES}')

    df_edges['Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()

    max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
    df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})
    timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())
    y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())

    logging.info(f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
    logging.info(f"Number of nodes = {df_nodes.shape[0]}")
    logging.info(f"Number of transactions = {df_edges.shape[0]}")

    base_edge_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
    edge_features = base_edge_features + GF_FEATURES
    node_features = ['Feature']

    logging.info(f'Edge features being used: {len(edge_features)}개 ({base_edge_features} + GF 22개)')

    x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
    edge_index = torch.LongTensor(df_edges.loc[:, ['from_id', 'to_id']].to_numpy().T)
    edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()

    n_days = int(timestamps.max() / (3600 * 24) + 1)
    n_samples = y.shape[0]

    daily_irs, weighted_daily_irs, daily_inds, daily_trans = [], [], [], []
    for day in range(n_days):
        l = day * 24 * 3600
        r = (day + 1) * 24 * 3600
        day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
        daily_irs.append(y[day_inds].float().mean())
        weighted_daily_irs.append(y[day_inds].float().mean() * day_inds.shape[0] / n_samples)
        daily_inds.append(day_inds)
        daily_trans.append(day_inds.shape[0])

    split_per = [0.6, 0.2, 0.2]
    daily_totals = np.array(daily_trans)
    d_ts = daily_totals
    I = list(range(len(d_ts)))
    split_scores = dict()
    for i, j in itertools.combinations(I, 2):
        if j >= i:
            split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
            split_totals_sum = np.sum(split_totals)
            split_props = [v / split_totals_sum for v in split_totals]
            split_error = [abs(v - t) / t for v, t in zip(split_props, split_per)]
            score = max(split_error)
            split_scores[(i, j)] = score

    i, j = min(split_scores, key=split_scores.get)
    split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]
    logging.info(f'Calculate split: {split}')

    split_inds = {k: [] for k in range(3)}
    for i in range(3):
        for day in split[i]:
            split_inds[i].append(daily_inds[day])

    tr_inds  = torch.cat(split_inds[0])
    val_inds = torch.cat(split_inds[1])
    te_inds  = torch.cat(split_inds[2])

    logging.info(f"Train: {tr_inds.shape[0] / y.shape[0] * 100:.2f}% | IR: {y[tr_inds].float().mean() * 100:.2f}%")
    logging.info(f"Val  : {val_inds.shape[0] / y.shape[0] * 100:.2f}% | IR: {y[val_inds].float().mean() * 100:.2f}%")
    logging.info(f"Test : {te_inds.shape[0] / y.shape[0] * 100:.2f}% | IR: {y[te_inds].float().mean() * 100:.2f}%")

    e_tr  = tr_inds.numpy()
    e_val = np.concatenate([tr_inds, val_inds])

    tr_edge_index,  tr_edge_attr,  tr_y,  tr_edge_times  = edge_index[:, e_tr],  edge_attr[e_tr],  y[e_tr],  timestamps[e_tr]
    val_edge_index, val_edge_attr, val_y, val_edge_times = edge_index[:, e_val], edge_attr[e_val], y[e_val], timestamps[e_val]
    te_edge_index,  te_edge_attr,  te_y,  te_edge_times  = edge_index,           edge_attr,        y,        timestamps

    tr_data  = GraphData(x=x, y=tr_y,  edge_index=tr_edge_index,  edge_attr=tr_edge_attr,  timestamps=tr_edge_times)
    val_data = GraphData(x=x, y=val_y, edge_index=val_edge_index, edge_attr=val_edge_attr, timestamps=val_edge_times)
    te_data  = GraphData(x=x, y=te_y,  edge_index=te_edge_index,  edge_attr=te_edge_attr,  timestamps=te_edge_times)

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

    if args.reverse_mp:
        tr_data  = create_hetero_obj(tr_data.x,  tr_data.y,  tr_data.edge_index,  tr_data.edge_attr,  tr_data.timestamps,  args)
        val_data = create_hetero_obj(val_data.x, val_data.y, val_data.edge_index, val_data.edge_attr, val_data.timestamps, args)
        te_data  = create_hetero_obj(te_data.x,  te_data.y,  te_data.edge_index,  te_data.edge_attr,  te_data.timestamps,  args)

    logging.info(f'train data: {tr_data}')
    logging.info(f'val data  : {val_data}')
    logging.info(f'test data : {te_data}')

    return tr_data, val_data, te_data, tr_inds, val_inds, te_inds
