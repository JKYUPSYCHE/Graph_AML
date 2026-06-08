"""
gnn/graphsaint/xai.py

TP / FP / FN / TN 그룹별 gradient saliency 기반 엣지 피처 중요도 분석
  - GraphSAINT 전용: te_data는 test 엣지만 포함 → te_inds 마스킹 불필요
  - batch.edge_attr[:, 0] = add_arange_ids가 붙인 te_data 내 인덱스
  - reverse_mp=True이면 homo_to_hetero on-the-fly 적용
"""

import logging
import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import k_hop_subgraph

from train_util import homo_to_hetero

_BASE_FEATURE_NAMES = [
    'amount__current__log1p',
    'cat__payment_currency__code',
    'cat__receiving_currency__code',
    'cat__payment_format__code',
    'time__row__hour',
    'time__row__dayofweek',
    'time__row__is_weekend',
]

_GF_FEATURE_NAMES = [
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

N_HOPS    = 2
N_SAMPLES = 500


def _feature_names(args):
    names = list(_BASE_FEATURE_NAMES) + list(_GF_FEATURE_NAMES)
    if getattr(args, 'ports', False):
        names += ['port_in', 'port_out']
    if getattr(args, 'tds', False):
        names += ['td_in', 'td_out']
    return names


def _collect_predictions(loader, model, device, args):
    """te_loader 전체를 순회하며 (preds, gts, edge_inds) 수집.
    GraphSAINT는 te_data가 test 전용이므로 te_inds 마스킹 없이 전 엣지 사용."""
    model.eval()
    all_preds, all_gts, all_einds = [], [], []

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc='[XAI] inference'):
            edge_ids = batch.edge_attr[:, 0].cpu()   # te_data 내 arange 인덱스

            if getattr(args, 'reverse_mp', False):
                hbatch = homo_to_hetero(batch, args)
                hbatch['node', 'to', 'node'].edge_attr     = hbatch['node', 'to', 'node'].edge_attr[:, 1:]
                hbatch['node', 'rev_to', 'node'].edge_attr = hbatch['node', 'rev_to', 'node'].edge_attr[:, 1:]
                hbatch.to(device)
                out = model(hbatch.x_dict, hbatch.edge_index_dict, hbatch.edge_attr_dict)
                out = out[('node', 'to', 'node')]
                gts = hbatch['node', 'to', 'node'].y.cpu()
            else:
                batch.edge_attr = batch.edge_attr[:, 1:]
                batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                gts = batch.y.cpu()

            all_preds.append(out.argmax(dim=-1).cpu())
            all_gts.append(gts)
            all_einds.append(edge_ids)

    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_gts).numpy(),
        torch.cat(all_einds).numpy(),
    )


def _sample_groups(preds, gts, edge_inds, n_samples, seed=42):
    """TP/FP/FN/TN으로 분류 후 각 그룹에서 최대 n_samples개 샘플링."""
    rng  = np.random.default_rng(seed)
    defs = {
        'TP': (preds == 1) & (gts == 1),
        'FP': (preds == 1) & (gts == 0),
        'FN': (preds == 0) & (gts == 1),
        'TN': (preds == 0) & (gts == 0),
    }
    groups = {}
    for name, cond in defs.items():
        pool   = edge_inds[cond]
        k      = min(len(pool), n_samples)
        chosen = rng.choice(pool, size=k, replace=False) if k > 0 else np.array([], dtype=int)
        groups[name] = chosen
        logging.info(f'[XAI] {name}: 전체 {int(cond.sum())}개 → {k}개 샘플링')
    return groups


def _get_subgraph(te_x, te_edge_index, te_edge_attr, te_y, te_num_nodes, edge_idx, args):
    """edge_idx의 N_HOPS-hop 서브그래프와 서브그래프 내 로컬 엣지 인덱스를 반환."""
    src = int(te_edge_index[0, edge_idx])
    dst = int(te_edge_index[1, edge_idx])

    subset, sub_ei, _, edge_mask = k_hop_subgraph(
        node_idx=[src, dst],
        num_hops=N_HOPS,
        edge_index=te_edge_index,
        relabel_nodes=True,
        num_nodes=te_num_nodes,
    )

    positions = edge_mask.nonzero(as_tuple=True)[0]
    matches   = (positions == edge_idx).nonzero(as_tuple=True)[0]
    if len(matches) == 0:
        return None, None
    local_idx = int(matches[0])

    sub_x = te_x[subset]

    return {
        'x':          sub_x,
        'edge_index': sub_ei,
        'edge_attr':  te_edge_attr[edge_mask],
        'y':          te_y[edge_mask],
        'num_nodes':  int(subset.shape[0]),
    }, local_idx


def _gradient_saliency(model, sub, local_idx, args, device):
    """gradient saliency로 엣지 피처 중요도 계산. class 1 확률에 대한 |grad| 반환."""
    model.eval()
    edge_attr = sub['edge_attr'].detach().clone().to(device).requires_grad_(True)
    x         = sub['x'].to(device)
    ei        = sub['edge_index'].to(device)

    if getattr(args, 'reverse_mp', False):
        out = model(
            {'node': x},
            {('node', 'to',     'node'): ei,
             ('node', 'rev_to', 'node'): ei.flip(0)},
            {('node', 'to',     'node'): edge_attr,
             ('node', 'rev_to', 'node'): sub['edge_attr'].detach().clone().to(device)},
        )
        out = out[('node', 'to', 'node')]
    else:
        out = model(x, ei, edge_attr)

    score = out[local_idx].softmax(dim=-1)[1]
    score.backward()

    return edge_attr.grad[local_idx].abs().detach().cpu().numpy()


def run_xai(te_loader, model, te_data, device, args, out_dir,
            run_name=None, n_samples=N_SAMPLES):
    """
    XAI 전체 파이프라인 실행 (GraphSAINT 전용).

    Args:
        te_loader : GraphSAINTRandomWalkSampler (te_data 기반, add_arange_ids 적용 후)
        model     : 학습된 GNN 모델
        te_data   : get_data에서 반환한 te_data (homo, add_arange_ids 적용 후)
        device    : torch.device
        args      : SimpleNamespace
        out_dir   : 결과 저장 경로
        run_name  : 저장 파일명 prefix (None이면 'xai' 사용)
        n_samples : 그룹당 최대 샘플 수 (기본 500)
    """
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix     = run_name if run_name else 'xai'
    feat_names = _feature_names(args)

    te_x          = te_data.x
    te_edge_index = te_data.edge_index
    te_edge_attr  = te_data.edge_attr[:, 1:]   # col 0 = arange ID 제거
    te_y          = te_data.y
    te_num_nodes  = int(te_data.num_nodes)

    logging.info('=== XAI 분석 시작 ===')

    logging.info('[XAI] Step 1: 전체 test 예측 수집')
    preds, gts, edge_inds = _collect_predictions(te_loader, model, device, args)
    logging.info(
        f'[XAI] TP={int(((preds==1)&(gts==1)).sum())} '
        f'FP={int(((preds==1)&(gts==0)).sum())} '
        f'FN={int(((preds==0)&(gts==1)).sum())} '
        f'TN={int(((preds==0)&(gts==0)).sum())}'
    )

    logging.info('[XAI] Step 2: TP/FP/FN/TN 그룹 샘플링')
    groups = _sample_groups(preds, gts, edge_inds, n_samples=n_samples)

    logging.info('[XAI] Step 3: 그룹별 gradient saliency 계산')
    records          = []
    group_importance = {g: [] for g in groups}

    for group_name, sampled_einds in groups.items():
        if len(sampled_einds) == 0:
            logging.warning(f'[XAI] {group_name}: 샘플 없음, 건너뜀')
            continue

        logging.info(f'[XAI] {group_name} ({len(sampled_einds)}개) 처리 중...')

        for edge_idx in tqdm.tqdm(sampled_einds, desc=f'XAI {group_name}'):
            sub, local_idx = _get_subgraph(
                te_x, te_edge_index, te_edge_attr, te_y, te_num_nodes, int(edge_idx), args
            )
            if sub is None:
                logging.warning(f'[XAI] 서브그래프 추출 실패: edge_idx={edge_idx}')
                continue

            try:
                importance = _gradient_saliency(model, sub, local_idx, args, device)
            except Exception as e:
                logging.warning(f'[XAI] saliency 실패 (edge_idx={edge_idx}): {e}')
                continue

            importance = importance[:len(feat_names)]
            group_importance[group_name].append(importance)

            rec = {'group': group_name, 'edge_idx': int(edge_idx)}
            for fname, fval in zip(feat_names, importance):
                rec[fname] = float(fval)
            records.append(rec)

    summary_rows = []
    for g, imps in group_importance.items():
        if not imps:
            continue
        arr  = np.stack(imps)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        row  = {'group': g, 'n_samples': len(imps)}
        for fname, m, s in zip(feat_names, mean, std):
            row[f'{fname}__mean'] = float(m)
            row[f'{fname}__std']  = float(s)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index('group')
    records_df = pd.DataFrame(records)

    summary_path = out_dir / f'{prefix}_feature_importance.csv'
    records_path = out_dir / f'{prefix}_feature_importance_individual.csv'
    summary_df.to_csv(summary_path)
    records_df.to_csv(records_path, index=False)

    mean_cols = [c for c in summary_df.columns if c.endswith('__mean')]
    logging.info(f'[XAI] 저장 완료:\n  {summary_path}\n  {records_path}')
    logging.info(f'[XAI] 그룹별 평균 피처 중요도:\n{summary_df[mean_cols].to_string()}')

    return summary_df, records_df
