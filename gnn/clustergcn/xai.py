"""
gnn/clustergcn/xai.py

TP / FP / FN / TN 그룹별 gradient saliency 기반 엣지 피처 중요도 분석 (ClusterGCN 버전)
  - baseline과 달리 ClusterLoader 배치 기준 → input_id 대신 te_inds mask 사용
  - k-hop 서브그래프 고정 후 gradient saliency 계산 (te_data homo GraphData 기준)
  - 그룹별 평균·표준편차를 CSV로 저장
"""

import logging
import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
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
N_HOPS    = 2
N_SAMPLES = 500


def _feature_names(args):
    names = list(_BASE_FEATURE_NAMES)
    if getattr(args, 'ports', False):
        names += ['port_in', 'port_out']
    if getattr(args, 'tds', False):
        names += ['td_in', 'td_out']
    return names


def _collect_predictions(loader, te_inds, model, device, args):
    """ClusterLoader 배치를 순회하며 test 엣지의 (preds, gts, edge_inds) 수집.

    ClusterGCN에서는 input_id가 없으므로 te_inds mask로 test 엣지를 식별.
    arange ID(edge_attr col 0) == te_data 내 엣지 인덱스이므로 einds로 바로 사용.
    """
    model.eval()
    all_preds, all_gts, all_einds = [], [], []

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc='[XAI] inference'):
            # te_data 전체 엣지 중 test 엣지 식별 (arange ID == te_data 인덱스)
            batch_arange = batch.edge_attr[:, 0].cpu()
            mask         = torch.isin(batch_arange, te_inds.cpu())

            if mask.sum() == 0:
                continue

            if getattr(args, 'ego', False):
                batch.x = torch.cat([batch.x, torch.ones(batch.x.shape[0], 1)], dim=1)

            if getattr(args, 'reverse_mp', False):
                hbatch = homo_to_hetero(batch, args)
                hbatch['node', 'to', 'node'].edge_attr     = hbatch['node', 'to', 'node'].edge_attr[:, 1:]
                hbatch['node', 'rev_to', 'node'].edge_attr = hbatch['node', 'rev_to', 'node'].edge_attr[:, 1:]
                hbatch.to(device)
                out = model(hbatch.x_dict, hbatch.edge_index_dict, hbatch.edge_attr_dict)
                out = out[('node', 'to', 'node')][mask]
                gts = hbatch['node', 'to', 'node'].y[mask].cpu()
            else:
                batch.edge_attr = batch.edge_attr[:, 1:]
                batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                out = out[mask]
                gts = batch.y[mask].cpu()

            # arange ID가 곧 te_data 내 엣지 인덱스
            einds = batch_arange[mask].long()

            all_preds.append(out.argmax(dim=-1).cpu())
            all_gts.append(gts)
            all_einds.append(einds)

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
    """
    edge_idx의 N_HOPS-hop 서브그래프와 서브그래프 내 로컬 엣지 인덱스를 반환.
    args.ego=True이면 seed 노드(src, dst)에 ego 피처(1)를 추가합니다.
    """
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

    # ego=True이면 seed 노드 위치에 1을 추가
    if getattr(args, 'ego', False):
        ego_ids     = torch.zeros(sub_x.shape[0], 1)
        subset_list = subset.tolist()
        for seed_node in [src, dst]:
            if seed_node in subset_list:
                ego_ids[subset_list.index(seed_node)] = 1.0
        sub_x = torch.cat([sub_x, ego_ids], dim=1)

    return {
        'x':          sub_x,
        'edge_index': sub_ei,
        'edge_attr':  te_edge_attr[edge_mask],
        'y':          te_y[edge_mask],
        'num_nodes':  int(subset.shape[0]),
    }, local_idx


def _gradient_saliency(model, sub, local_idx, args, device):
    """
    고정된 서브그래프에서 gradient saliency로 엣지 피처 중요도를 계산합니다.
    class 1 (자금세탁) 예측 확률에 대한 edge_attr 각 피처의 gradient 절댓값을 반환합니다.
    Returns: np.ndarray, shape [n_edge_features]
    """
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

    score = out[local_idx].softmax(dim=-1)[1]   # 자금세탁(class 1) 확률
    score.backward()

    return edge_attr.grad[local_idx].abs().detach().cpu().numpy()


def run_xai(te_loader, te_inds, model, te_data, device, args, out_dir,
            run_name=None, n_samples=N_SAMPLES):
    """
    XAI 전체 파이프라인을 실행합니다.

    Args:
        te_loader : test DataLoader (add_arange_ids 적용 후)
        te_inds   : test 엣지 인덱스 (get_data 반환값)
        model     : 학습된 GNN 모델
        te_data   : get_data에서 반환한 te_data (전체 엣지 포함)
        device    : torch.device
        args      : argparse Namespace
        out_dir   : 결과 저장 경로
        run_name  : 저장 파일명 prefix (None이면 'xai' 사용)
        n_samples : 그룹당 최대 샘플 수 (기본 50)

    Returns:
        summary_df : 그룹별 평균·표준편차 DataFrame (index=group)
        records_df : 개별 샘플 결과 DataFrame
    """
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix     = run_name if run_name else 'xai'
    feat_names = _feature_names(args)

    # te_data에서 homo 포맷으로 꺼냄 (add_arange_ids 이후 첫 컬럼은 ID → 제거)
    if isinstance(te_data, HeteroData):
        te_x          = te_data['node'].x
        te_edge_index = te_data['node', 'to', 'node'].edge_index
        te_edge_attr  = te_data['node', 'to', 'node'].edge_attr[:, 1:]
        te_y          = te_data['node', 'to', 'node'].y
        te_num_nodes  = int(te_x.shape[0])
    else:
        te_x          = te_data.x
        te_edge_index = te_data.edge_index
        te_edge_attr  = te_data.edge_attr[:, 1:]
        te_y          = te_data.y
        te_num_nodes  = int(te_data.num_nodes)

    logging.info('=== XAI 분석 시작 ===')

    # Step 1: 예측 수집
    logging.info('[XAI] Step 1: 전체 test 예측 수집')
    preds, gts, edge_inds = _collect_predictions(te_loader, te_inds, model, device, args)
    logging.info(
        f'[XAI] TP={int(((preds==1)&(gts==1)).sum())} '
        f'FP={int(((preds==1)&(gts==0)).sum())} '
        f'FN={int(((preds==0)&(gts==1)).sum())} '
        f'TN={int(((preds==0)&(gts==0)).sum())}'
    )

    # Step 2: 그룹 샘플링
    logging.info('[XAI] Step 2: TP/FP/FN/TN 그룹 샘플링')
    groups = _sample_groups(preds, gts, edge_inds, n_samples=n_samples)

    # Step 3: 그룹별 gradient saliency
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

    # Step 4: 그룹별 평균·표준편차 집계
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

    # Step 5: 저장
    summary_path = out_dir / f'{prefix}_feature_importance.csv'
    records_path = out_dir / f'{prefix}_feature_importance_individual.csv'
    summary_df.to_csv(summary_path)
    records_df.to_csv(records_path, index=False)

    mean_cols = [c for c in summary_df.columns if c.endswith('__mean')]
    logging.info(f'[XAI] 저장 완료:\n  {summary_path}\n  {records_path}')
    logging.info(f'[XAI] 그룹별 평균 피처 중요도:\n{summary_df[mean_cols].to_string()}')

    return summary_df, records_df
