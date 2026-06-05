import os
import time
import json
import torch
import tqdm
import psutil
import logging
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from sklearn.metrics import (f1_score, recall_score, precision_score,
                              average_precision_score, log_loss, confusion_matrix)


# ── Misc utils ────────────────────────────────────────────────────────────────
def extract_param(parameter_name: str, args) -> float:
    with open('./model_settings.json', 'r') as f:
        data = json.load(f)
    return data.get(args.model, {}).get('params', {}).get(parameter_name, None)


def add_arange_ids(data_list):
    """엣지 피처 맨 앞에 전역 엣지 인덱스(arange)를 prepend."""
    for data in data_list:
        data.edge_attr = torch.cat(
            [torch.arange(data.edge_attr.shape[0]).view(-1, 1), data.edge_attr], dim=1)


def save_model(model, optimizer, epoch, args, data_config):
    suffix = "" if not args.finetune else "_finetuned"
    base = f'{data_config["paths"]["model_to_save"]}/checkpoint_{args.unique_name}{suffix}'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'{base}.tar')
    with open(f'{base}_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)


def load_model(model, device, args, config, data_config):
    checkpoint = torch.load(
        f'{data_config["paths"]["model_to_load"]}/checkpoint_{args.unique_name}.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


# ── homo → hetero 변환 (배치 내 on-the-fly) ──────────────────────────────────
def homo_to_hetero(batch, args):
    """GraphSAINT homo 배치를 reverse_mp용 HeteroData로 변환.
    edge_attr 첫 번째 컬럼(arange ID)은 아직 제거하지 않은 상태로 호출해야 함."""
    data = HeteroData()
    data['node'].x = batch.x
    data['node', 'to', 'node'].edge_index     = batch.edge_index
    data['node', 'rev_to', 'node'].edge_index = batch.edge_index.flipud()
    data['node', 'to', 'node'].edge_attr      = batch.edge_attr
    data['node', 'rev_to', 'node'].edge_attr  = batch.edge_attr.clone()
    if getattr(args, 'ports', False) and getattr(args, 'reverse_ports', True):
        rev = data['node', 'rev_to', 'node'].edge_attr
        rev[:, [-1, -2]] = rev[:, [-2, -1]].clone()
    data['node', 'to', 'node'].y = batch.y
    return data


# ── GraphSAINT DataLoaders ────────────────────────────────────────────────────
def _to_pyg_data(data):
    """커스텀 GraphData → 표준 PyG Data 변환.
    GraphSAINTRandomWalkSampler가 내부에서 data.__class__()로 빈 인스턴스를
    생성하는데, GraphData는 x가 필수라 crash. 표준 Data는 빈 생성자 허용.

    edge_index의 node ID를 contiguous [0, N)으로 remap.
    (edge_index에 x.shape[0]보다 큰 node ID가 있으면 GraphSAINT가 3M 노드짜리
    SparseTensor를 생성하려다 OOM/crash가 남 — remap으로 num_nodes를 작게 유지)"""
    edge_index = data.edge_index
    unique_nodes, new_flat = edge_index.flatten().unique(return_inverse=True)
    new_edge_index = new_flat.view(2, -1)
    num_nodes_new = unique_nodes.shape[0]

    orig_n = data.x.shape[0] if data.x is not None else 0
    if data.x is not None:
        x_new = torch.zeros(num_nodes_new, data.x.shape[1], dtype=data.x.dtype)
        has_feat = unique_nodes < orig_n
        x_new[has_feat] = data.x[unique_nodes[has_feat]]
    else:
        x_new = None

    return Data(
        x=x_new,
        edge_index=new_edge_index,
        edge_attr=data.edge_attr,
        y=data.y,
        num_nodes=num_nodes_new,
    )


def get_loaders(tr_data, val_data, te_data, args):
    """
    GraphSAINTRandomWalkSampler 기반 로더 생성.
    reverse_mp=True이더라도 샘플러는 항상 homo Data를 받음.
    hetero 변환은 학습/평가 루프 내부에서 on-the-fly로 수행.

    args 필드:
        walk_length      : 랜덤워크 길이 (기본 2)
        num_steps        : 에폭당 배치 수 (기본 30)
        saint_batch_size : 워크당 시작 노드 수 (기본 200)
    """
    walk_length = getattr(args, 'walk_length', 2)
    num_steps   = getattr(args, 'num_steps', 30)
    batch_size  = getattr(args, 'saint_batch_size', 200)

    tr_loader  = GraphSAINTRandomWalkSampler(
        _to_pyg_data(tr_data),  batch_size=batch_size, walk_length=walk_length,
        num_steps=num_steps,  sample_coverage=100, num_workers=0)
    val_loader = GraphSAINTRandomWalkSampler(
        _to_pyg_data(val_data), batch_size=batch_size, walk_length=walk_length,
        num_steps=num_steps,  sample_coverage=100, num_workers=0)
    te_loader  = GraphSAINTRandomWalkSampler(
        _to_pyg_data(te_data),  batch_size=batch_size, walk_length=walk_length,
        num_steps=num_steps,  sample_coverage=100, num_workers=0)

    return tr_loader, val_loader, te_loader


# ── Evaluate (homo) ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_graphsaint(loader, model, device, args, te_inds=None):
    """te_inds=None이면 배치 내 모든 엣지 평가 (val용), 지정 시 test 엣지만 평가."""
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter()

    preds, pred_probas, ground_truths = [], [], []

    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        if te_inds is not None:
            mask = torch.isin(batch.edge_attr[:, 0].cpu(), te_inds.cpu())
        else:
            mask = torch.ones(batch.edge_attr.shape[0], dtype=torch.bool)

        batch.edge_attr = batch.edge_attr[:, 1:]

        if mask.sum() == 0:
            continue

        batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        out = out[mask.to(device)]
        gts = batch.y[mask.to(device)]

        preds.append(out.argmax(dim=-1).cpu())
        pred_probas.append(out.softmax(dim=-1)[:, 1].detach().cpu())
        ground_truths.append(gts.cpu())

    t_end = time.perf_counter()
    memory_mb = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                 if torch.cuda.is_available()
                 else psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    model.train()

    return _compute_metrics(preds, pred_probas, ground_truths, memory_mb, t_end - t_start)


# ── Evaluate (hetero) ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_graphsaint_hetero(loader, model, device, args, te_inds=None):
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter()

    preds, pred_probas, ground_truths = [], [], []

    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        if te_inds is not None:
            mask = torch.isin(batch.edge_attr[:, 0].cpu(), te_inds.cpu())
        else:
            mask = torch.ones(batch.edge_attr.shape[0], dtype=torch.bool)

        if mask.sum() == 0:
            continue

        hbatch = homo_to_hetero(batch, args)
        hbatch['node', 'to', 'node'].edge_attr     = hbatch['node', 'to', 'node'].edge_attr[:, 1:]
        hbatch['node', 'rev_to', 'node'].edge_attr = hbatch['node', 'rev_to', 'node'].edge_attr[:, 1:]

        hbatch.to(device)
        out = model(hbatch.x_dict, hbatch.edge_index_dict, hbatch.edge_attr_dict)
        out = out[('node', 'to', 'node')][mask.to(device)]
        gts = hbatch['node', 'to', 'node'].y[mask.to(device)]

        preds.append(out.argmax(dim=-1).cpu())
        pred_probas.append(out.softmax(dim=-1)[:, 1].detach().cpu())
        ground_truths.append(gts.cpu())

    t_end = time.perf_counter()
    memory_mb = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                 if torch.cuda.is_available()
                 else psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    model.train()

    return _compute_metrics(preds, pred_probas, ground_truths, memory_mb, t_end - t_start)


# ── 공통 메트릭 계산 ──────────────────────────────────────────────────────────
def _compute_metrics(preds, pred_probas, ground_truths, memory_mb, time_s):
    pred         = torch.cat(preds).numpy()
    pred_proba   = torch.cat(pred_probas).numpy()
    ground_truth = torch.cat(ground_truths).numpy()

    tn, fp, fn, tp = confusion_matrix(ground_truth, pred, labels=[0, 1]).ravel()
    return {
        'f1':        f1_score(ground_truth, pred, zero_division=0),
        'recall':    recall_score(ground_truth, pred, zero_division=0),
        'precision': precision_score(ground_truth, pred, zero_division=0),
        'auprc':     average_precision_score(ground_truth, pred_proba),
        'log_loss':  log_loss(ground_truth, pred_proba),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'memory_mb': memory_mb, 'time_s': time_s,
    }
