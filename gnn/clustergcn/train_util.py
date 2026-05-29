import os
import time
import torch
import tqdm
import psutil
import logging
import json
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import ClusterData, ClusterLoader
from sklearn.metrics import (f1_score, recall_score, precision_score,
                              average_precision_score, log_loss, confusion_matrix)


# ── Ego IDs ───────────────────────────────────────────────────────────────────
def add_ego_ids(batch):
    """ClusterGCN은 seed edge 개념이 없으므로 모든 노드에 ego=1 부여."""
    batch.x = torch.cat([batch.x, torch.ones(batch.x.shape[0], 1)], dim=1)
    return batch


# ── HeteroData 변환 ───────────────────────────────────────────────────────────
def homo_to_hetero(batch, args):
    """ClusterLoader homo 배치를 reverse_mp용 HeteroData로 변환.
    edge_attr 첫 번째 컬럼(arange ID)은 아직 제거하지 않은 상태로 호출해야 함."""
    data = HeteroData()
    data['node'].x = batch.x
    data['node', 'to', 'node'].edge_index     = batch.edge_index
    data['node', 'rev_to', 'node'].edge_index = batch.edge_index.flipud()
    data['node', 'to', 'node'].edge_attr      = batch.edge_attr
    data['node', 'rev_to', 'node'].edge_attr  = batch.edge_attr.clone()
    if getattr(args, 'ports', False):
        rev = data['node', 'rev_to', 'node'].edge_attr
        rev[:, [-1, -2]] = rev[:, [-2, -1]].clone()
    data['node', 'to', 'node'].y = batch.y
    return data


# ── Misc utils ────────────────────────────────────────────────────────────────
def extract_param(parameter_name: str, args) -> float:
    file_path = './model_settings.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data.get(args.model, {}).get('params', {}).get(parameter_name, None)


def add_arange_ids(data_list):
    """엣지 피처 맨 앞에 전역 엣지 인덱스(arange)를 prepend."""
    for data in data_list:
        if isinstance(data, HeteroData):
            n = data['node', 'to', 'node'].edge_attr.shape[0]
            data['node', 'to', 'node'].edge_attr = torch.cat(
                [torch.arange(n).view(-1, 1), data['node', 'to', 'node'].edge_attr], dim=1)
            offset = n
            m = data['node', 'rev_to', 'node'].edge_attr.shape[0]
            data['node', 'rev_to', 'node'].edge_attr = torch.cat(
                [torch.arange(offset, offset + m).view(-1, 1),
                 data['node', 'rev_to', 'node'].edge_attr], dim=1)
        else:
            data.edge_attr = torch.cat(
                [torch.arange(data.edge_attr.shape[0]).view(-1, 1), data.edge_attr], dim=1)


def save_model(model, optimizer, epoch, args, data_config):
    suffix = "" if not args.finetune else "_finetuned"
    base = f'{data_config["paths"]["model_to_save"]}/checkpoint_{args.unique_name}{suffix}'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
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


# ── ClusterGCN DataLoaders ────────────────────────────────────────────────────
def get_loaders(tr_data, val_data, te_data, args, cache_dir=None):
    """
    tr_data   : train 엣지만 포함 (GraphData)
    val_data  : val 엣지만 포함 (GraphData)
    te_data   : 전체 엣지 포함 (GraphData) — te_inds로 test 엣지 식별
    cache_dir : METIS 결과 캐시 경로 (지정 시 재실행 때 로드, None이면 캐시 없음)
    """
    import os
    num_parts = getattr(args, 'num_parts', 300)
    cpb       = getattr(args, 'clusters_per_batch', 10)
    recursive = getattr(args, 'recursive', True)

    if cache_dir:
        tr_save  = os.path.join(cache_dir, 'tr')
        val_save = os.path.join(cache_dir, 'val')
        te_save  = os.path.join(cache_dir, 'te')
        for d in [tr_save, val_save, te_save]:
            os.makedirs(d, exist_ok=True)
    else:
        tr_save = val_save = te_save = None

    logging.info(f'[ClusterGCN] Partitioning into {num_parts} parts, {cpb} clusters/batch, recursive={recursive}...')
    tr_cluster  = ClusterData(tr_data,  num_parts=num_parts, recursive=recursive, log=False, save_dir=tr_save)
    val_cluster = ClusterData(val_data, num_parts=num_parts, recursive=recursive, log=False, save_dir=val_save)
    te_cluster  = ClusterData(te_data,  num_parts=num_parts, recursive=recursive, log=False, save_dir=te_save)
    logging.info('[ClusterGCN] Partitioning done.')

    tr_loader  = ClusterLoader(tr_cluster,  batch_size=cpb, shuffle=True,  drop_last=True,  num_workers=0)
    val_loader = ClusterLoader(val_cluster, batch_size=cpb, shuffle=False, num_workers=0)
    te_loader  = ClusterLoader(te_cluster,  batch_size=cpb, shuffle=False, num_workers=0)

    return tr_loader, val_loader, te_loader


# ── Evaluate (homo) ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_homo_cluster(loader, model, device, args, te_inds=None):
    """te_inds=None이면 배치 내 모든 엣지 평가 (val용), 지정 시 test 엣지만 평가."""
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter()

    preds, pred_probas, ground_truths = [], [], []

    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        if getattr(args, 'ego', False):
            batch.x = torch.cat([batch.x, torch.ones(batch.x.shape[0], 1)], dim=1)

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
        'memory_mb': memory_mb, 'time_s': t_end - t_start,
    }


# ── Evaluate (hetero) ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_hetero_cluster(loader, model, device, args, te_inds=None):
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter()

    preds, pred_probas, ground_truths = [], [], []

    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        if getattr(args, 'ego', False):
            batch.x = torch.cat([batch.x, torch.ones(batch.x.shape[0], 1)], dim=1)

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
        'memory_mb': memory_mb, 'time_s': t_end - t_start,
    }
