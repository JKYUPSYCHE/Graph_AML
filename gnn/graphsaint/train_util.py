import os
import time
import json
import torch
import tqdm
import psutil
import logging
from torch_geometric.data import HeteroData
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


# ── GraphSAINT DataLoaders ────────────────────────────────────────────────────
def get_loaders(tr_data, val_data, te_data, args):
    """
    GraphSAINTRandomWalkSampler 기반 로더 생성.

    args 필드:
        walk_length      : 랜덤워크 길이 (기본 2)
        num_steps        : 에폭당 배치 수 (기본 30)
        saint_batch_size : 워크당 시작 노드 수 (기본 200)
    """
    walk_length = getattr(args, 'walk_length', 2)
    num_steps   = getattr(args, 'num_steps', 30)
    batch_size  = getattr(args, 'saint_batch_size', 200)

    # TODO: GraphSAINT는 homo Data만 직접 지원 — reverse_mp=True이면 homo로 학습 후 hetero 변환
    tr_loader  = GraphSAINTRandomWalkSampler(
        tr_data,  batch_size=batch_size, walk_length=walk_length,
        num_steps=num_steps,  sample_coverage=0, num_workers=0)
    val_loader = GraphSAINTRandomWalkSampler(
        val_data, batch_size=batch_size, walk_length=walk_length,
        num_steps=num_steps,  sample_coverage=0, num_workers=0)
    te_loader  = GraphSAINTRandomWalkSampler(
        te_data,  batch_size=batch_size, walk_length=walk_length,
        num_steps=num_steps,  sample_coverage=0, num_workers=0)

    return tr_loader, val_loader, te_loader


# ── Evaluate ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_graphsaint(loader, model, device, args, te_inds=None):
    """
    te_inds=None : 배치 내 모든 엣지 평가 (val용)
    te_inds 지정  : arange ID로 test 엣지만 필터링
    """
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

        batch.edge_attr = batch.edge_attr[:, 1:]  # arange ID 제거

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
