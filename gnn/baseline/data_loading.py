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
    '''Loads the AML transaction data.

    1. The data is loaded from the csv and the necessary features are chosen.
    2. The data is split into training, validation and test data (timestamp 누적 행 수 기준 60/20/20).
    3. PyG Data objects are created with the respective data splits.
    '''

    transaction_file = f"{data_config['paths']['aml_data']}/{args.data}/formatted_transactions.csv"
    df_edges = pd.read_csv(transaction_file)

    logging.info(f'Available Edge Features: {df_edges.columns.tolist()}')

    df_edges['Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()

    max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
    df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})
    timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())
    y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())

    logging.info(f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
    logging.info(f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}")
    logging.info(f"Number of transactions = {df_edges.shape[0]}")

    edge_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
    node_features = ['Feature']

    logging.info(f'Edge features being used: {edge_features}')
    logging.info(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

    x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
    edge_index = torch.LongTensor(df_edges.loc[:, ['from_id', 'to_id']].to_numpy().T)
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

    return tr_data, val_data, te_data, tr_inds, val_inds, te_inds
