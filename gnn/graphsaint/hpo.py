import logging
from copy import copy
from copy import deepcopy

import optuna
from training import train_gnn

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, tr_data, val_data, te_data, tr_inds, val_inds, te_inds, base_args, data_config):
    args = copy(base_args)

    # GraphSAINT 샘플링 파라미터
    args.walk_length      = trial.suggest_int('walk_length', 2, 10, step=2)
    args.saint_batch_size = trial.suggest_categorical('saint_batch_size', [500, 1000, 2000])
    args.num_steps        = trial.suggest_categorical('num_steps', [50, 100])

    # 클래스 가중치 (log scale — 낮은 값도 촘촘히 탐색)
    args.hpo_w_ce2 = trial.suggest_float('w_ce2', 3.0, 100.0, log=True)

    # trial마다 빠르게 수렴 여부 확인
    args.n_epochs   = 30
    args.patience   = 10
    args.save_model = False
    args.tqdm       = False
    args.unique_name = f'hpo_trial_{trial.number}'

    # add_arange_ids가 edge_attr을 in-place로 수정하므로 매 trial마다 deepcopy
    best_te, _ = train_gnn(
        deepcopy(tr_data), deepcopy(val_data), deepcopy(te_data),
        tr_inds, val_inds, te_inds, args, data_config,
    )

    if best_te is None:
        return 0.0
    return best_te.get('val_f1', 0.0)


def run_hpo(tr_data, val_data, te_data, tr_inds, val_inds, te_inds,
            base_args, data_config, n_trials=20):
    """
    Optuna TPE로 GraphSAINT(RW) 하이퍼파라미터 탐색.
    목적 함수: best epoch val F1 최대화.

    탐색 파라미터:
        walk_length      : 2, 4, 6, 8, 10
        saint_batch_size : 500, 1000, 2000
        num_steps        : 50, 100
        w_ce2            : 3.0 ~ 100.0 (log scale)
    """
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name='graphsaint_rw_hpo',
    )
    study.optimize(
        lambda trial: objective(
            trial, tr_data, val_data, te_data,
            tr_inds, val_inds, te_inds, base_args, data_config,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    return study


def print_hpo_results(study):
    best = study.best_trial
    print(f"\n=== HPO 결과 (총 {len(study.trials)}개 trial) ===")
    print(f"Best val F1 : {best.value:.4f}")
    print(f"Best params :")
    for k, v in best.params.items():
        print(f"  {k:20s} = {v}")

    print("\nTop-5 trials:")
    trials = sorted(study.trials, key=lambda t: t.value if t.value else -1, reverse=True)
    for t in trials[:5]:
        params_str = ', '.join(f'{k}={v}' for k, v in t.params.items())
        print(f"  trial {t.number:3d} | val F1 {t.value:.4f} | {params_str}")
