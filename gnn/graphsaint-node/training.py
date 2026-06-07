import copy
import time
import torch
import torch.nn.functional as F
import tqdm
import logging
from types import SimpleNamespace
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score, average_precision_score, log_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree
from train_util import (extract_param, add_arange_ids, get_loaders,
                        homo_to_hetero, evaluate_graphsaint, evaluate_graphsaint_hetero,
                        save_model, load_model)


def _log_best(best_epoch, best_val, best_te, total_time_s, avg_memory_mb):
    logging.info('Training complete.')
    logging.info(f'Best epoch: {best_epoch}')
    logging.info(f'Total training time: {total_time_s:.1f}s | Avg epoch memory: {avg_memory_mb:.1f}MB')
    logging.info(f"  Val  — F1: {best_val['f1']:.4f} | Recall: {best_val['recall']:.4f} | Precision: {best_val['precision']:.4f} | AUPRC: {best_val['auprc']:.4f} | LogLoss: {best_val['log_loss']:.4f}")
    logging.info(f"  Test — F1: {best_te['f1']:.4f} | Recall: {best_te['recall']:.4f} | Precision: {best_te['precision']:.4f} | AUPRC: {best_te['auprc']:.4f} | LogLoss: {best_te['log_loss']:.4f}")


def _write_metrics(writer, tr_result, val_result, te_result, epoch):
    for metric in ('f1', 'recall', 'precision', 'auprc', 'log_loss'):
        writer.add_scalars(metric.upper(), {
            'train': tr_result[metric],
            'val':   val_result[metric],
            'test':  te_result[metric],
        }, epoch)


def get_model(tr_data, config, args):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'baseline'))
    from models import GINe, PNA, GATe, RGCN

    n_feats = tr_data.x.shape[1]
    e_dim   = tr_data.edge_attr.shape[1] - 1  # col 0 = arange ID

    if args.model == 'gin':
        return GINe(
            num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
            n_hidden=round(config.n_hidden), residual=False, edge_updates=args.emlps,
            edge_dim=e_dim, dropout=config.dropout, final_dropout=config.final_dropout)
    elif args.model == 'gat':
        return GATe(
            num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
            n_hidden=round(config.n_hidden), n_heads=round(config.n_heads),
            edge_updates=args.emlps, edge_dim=e_dim,
            dropout=config.dropout, final_dropout=config.final_dropout)
    elif args.model == 'pna':
        d   = degree(tr_data.edge_index[1], dtype=torch.long)
        deg = torch.bincount(d, minlength=1)
        return PNA(
            num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
            n_hidden=round(config.n_hidden), edge_updates=args.emlps, edge_dim=e_dim,
            dropout=config.dropout, deg=deg, final_dropout=config.final_dropout)
    elif args.model == 'rgcn':
        return RGCN(
            num_features=n_feats, edge_dim=e_dim, num_relations=8,
            num_gnn_layers=round(config.n_gnn_layers), n_classes=2,
            n_hidden=round(config.n_hidden), edge_update=args.emlps,
            dropout=config.dropout, final_dropout=config.final_dropout, n_bases=None)


# ── Train (homo) ──────────────────────────────────────────────────────────────
def train_homo(tr_loader, val_loader, te_loader,
               model, optimizer, scheduler, loss_fn, args, config, device, data_config, writer):
    best_val_f1, best_val_result, best_te_result = 0, None, None
    best_epoch, best_model_state, patience_counter = 0, None, 0
    memory_mb_list = []
    t_train_start = time.perf_counter()

    for epoch in range(config.epochs):
        logging.info(f"[epoch {epoch + 1}/{config.epochs}]")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        preds, pred_probas, ground_truths = [], [], []

        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            node_norm = batch.node_norm if hasattr(batch, 'node_norm') else None
            batch.edge_attr = batch.edge_attr[:, 1:]
            batch.to(device)

            try:
                out = model(batch.x, batch.edge_index, batch.edge_attr)
            except ValueError as e:
                if 'Expected more than 1 value per channel' in str(e):
                    logging.warning(f"Small batch skipped: {batch.x.shape[0]} nodes")
                    continue
                raise

            if node_norm is not None:
                node_norm = node_norm.to(device)
                edge_norm = (node_norm[batch.edge_index[0]] + node_norm[batch.edge_index[1]]) / 2
                per_edge = F.cross_entropy(out, batch.y, weight=loss_fn.weight, reduction='none')
                loss = (per_edge * edge_norm).sum() / edge_norm.sum()
            else:
                loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()

            preds.append(out.argmax(dim=-1).detach().cpu())
            pred_probas.append(out.softmax(dim=-1)[:, 1].detach().cpu())
            ground_truths.append(batch.y.detach().cpu())

        if torch.cuda.is_available():
            memory_mb_list.append(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
        else:
            memory_mb_list.append(0.0)

        if not preds:
            logging.warning(f"Epoch {epoch + 1}: 모든 배치 skip됨.")
            continue

        _log_train(preds, pred_probas, ground_truths)
        tr_result = _make_tr_result(preds, pred_probas, ground_truths)

        val_result = evaluate_graphsaint(val_loader, model, device, args, te_inds=None)
        te_result  = evaluate_graphsaint(te_loader,  model, device, args, te_inds=None)
        _log_val_te(val_result, te_result)
        _write_metrics(writer, tr_result, val_result, te_result, epoch)

        scheduler.step(val_result['f1'])
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch)
        logging.info(f"LR: {current_lr:.2e}")

        best_val_f1, best_val_result, best_te_result, best_epoch, best_model_state, patience_counter, stop = \
            _update_best(val_result, te_result, best_val_f1, best_val_result, best_te_result,
                         best_epoch, best_model_state, patience_counter, epoch, model, optimizer, args, data_config)
        if stop:
            break

    return _finalize(model, best_val_result, best_te_result, best_epoch, best_model_state,
                     memory_mb_list, t_train_start, writer)


# ── Train (hetero) ────────────────────────────────────────────────────────────
def train_hetero(tr_loader, val_loader, te_loader,
                 model, optimizer, scheduler, loss_fn, args, config, device, data_config, writer):
    best_val_f1, best_val_result, best_te_result = 0, None, None
    best_epoch, best_model_state, patience_counter = 0, None, 0
    memory_mb_list = []
    t_train_start = time.perf_counter()

    for epoch in range(config.epochs):
        logging.info(f"[epoch {epoch + 1}/{config.epochs}]")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        preds, pred_probas, ground_truths = [], [], []

        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            node_norm = batch.node_norm if hasattr(batch, 'node_norm') else None

            hbatch = homo_to_hetero(batch, args)
            hbatch['node', 'to', 'node'].edge_attr     = hbatch['node', 'to', 'node'].edge_attr[:, 1:]
            hbatch['node', 'rev_to', 'node'].edge_attr = hbatch['node', 'rev_to', 'node'].edge_attr[:, 1:]
            hbatch.to(device)

            try:
                out = model(hbatch.x_dict, hbatch.edge_index_dict, hbatch.edge_attr_dict)
            except ValueError as e:
                if 'Expected more than 1 value per channel' in str(e):
                    logging.warning(f"Small batch skipped: {hbatch['node'].x.shape[0]} nodes")
                    continue
                raise

            out = out[('node', 'to', 'node')]
            y  = hbatch['node', 'to', 'node'].y
            ei = hbatch['node', 'to', 'node'].edge_index

            if node_norm is not None:
                node_norm = node_norm.to(device)
                edge_norm = (node_norm[ei[0]] + node_norm[ei[1]]) / 2
                per_edge = F.cross_entropy(out, y, weight=loss_fn.weight, reduction='none')
                loss = (per_edge * edge_norm).sum() / edge_norm.sum()
            else:
                loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            preds.append(out.argmax(dim=-1).detach().cpu())
            pred_probas.append(out.softmax(dim=-1)[:, 1].detach().cpu())
            ground_truths.append(y.detach().cpu())

        if torch.cuda.is_available():
            memory_mb_list.append(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
        else:
            memory_mb_list.append(0.0)

        if not preds:
            logging.warning(f"Epoch {epoch + 1}: 모든 배치 skip됨.")
            continue

        tr_result = _make_tr_result(preds, pred_probas, ground_truths)

        val_result = evaluate_graphsaint_hetero(val_loader, model, device, args, te_inds=None)
        te_result  = evaluate_graphsaint_hetero(te_loader,  model, device, args, te_inds=None)
        _log_val_te(val_result, te_result)
        _write_metrics(writer, tr_result, val_result, te_result, epoch)

        scheduler.step(val_result['f1'])
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch)
        logging.info(f"LR: {current_lr:.2e}")

        best_val_f1, best_val_result, best_te_result, best_epoch, best_model_state, patience_counter, stop = \
            _update_best(val_result, te_result, best_val_f1, best_val_result, best_te_result,
                         best_epoch, best_model_state, patience_counter, epoch, model, optimizer, args, data_config)
        if stop:
            break

    return _finalize(model, best_val_result, best_te_result, best_epoch, best_model_state,
                     memory_mb_list, t_train_start, writer)


# ── 공통 헬퍼 ─────────────────────────────────────────────────────────────────
def _make_tr_result(preds, pred_probas, ground_truths):
    import numpy as np
    pred         = torch.cat(preds).numpy()
    pred_proba   = torch.cat(pred_probas).numpy()
    ground_truth = torch.cat(ground_truths).numpy()
    result = {
        'f1':        f1_score(ground_truth, pred, zero_division=0),
        'recall':    recall_score(ground_truth, pred, zero_division=0),
        'precision': precision_score(ground_truth, pred, zero_division=0),
        'auprc':     average_precision_score(ground_truth, pred_proba),
        'log_loss':  log_loss(ground_truth, pred_proba),
    }
    logging.info(f"Train F1: {result['f1']:.4f} | Recall: {result['recall']:.4f} | Precision: {result['precision']:.4f} | AUPRC: {result['auprc']:.4f} | LogLoss: {result['log_loss']:.4f}")
    return result


def _log_train(preds, pred_probas, ground_truths):
    pass  # _make_tr_result에서 로깅 처리


def _log_val_te(val_result, te_result):
    logging.info(f"Val  — F1: {val_result['f1']:.4f} | Recall: {val_result['recall']:.4f} | Precision: {val_result['precision']:.4f} | AUPRC: {val_result['auprc']:.4f} | Mem: {val_result['memory_mb']:.1f}MB | Time: {val_result['time_s']:.1f}s")
    logging.info(f"Test — F1: {te_result['f1']:.4f} | Recall: {te_result['recall']:.4f} | Precision: {te_result['precision']:.4f} | AUPRC: {te_result['auprc']:.4f} | Mem: {te_result['memory_mb']:.1f}MB | Time: {te_result['time_s']:.1f}s")


def _update_best(val_result, te_result, best_val_f1, best_val_result, best_te_result,
                 best_epoch, best_model_state, patience_counter, epoch, model, optimizer, args, data_config):
    stop = False
    if val_result['f1'] > best_val_f1:
        best_val_f1      = val_result['f1']
        best_val_result  = val_result
        best_te_result   = te_result
        best_epoch       = epoch + 1
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
        if args.save_model:
            save_model(model, optimizer, epoch, args, data_config)
    else:
        patience_counter += 1
        if args.patience is not None and patience_counter >= args.patience:
            logging.info(f'Early stopping at epoch {epoch + 1} (patience={args.patience})')
            stop = True
    return best_val_f1, best_val_result, best_te_result, best_epoch, best_model_state, patience_counter, stop


def _finalize(model, best_val_result, best_te_result, best_epoch, best_model_state,
              memory_mb_list, t_train_start, writer):
    total_time_s  = time.perf_counter() - t_train_start
    avg_memory_mb = sum(memory_mb_list) / len(memory_mb_list) if memory_mb_list else 0.0
    writer.add_scalar('Total/training_time_s', total_time_s, 0)
    writer.add_scalar('Total/avg_memory_mb',   avg_memory_mb, 0)

    if best_val_result is None:
        logging.warning("학습 중 val F1 개선 없음.")
    else:
        _log_best(best_epoch, best_val_result, best_te_result, total_time_s, avg_memory_mb)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_te_result


# ── Entry point ───────────────────────────────────────────────────────────────
def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = SimpleNamespace(
        epochs=args.n_epochs,
        model=args.model,
        data=args.data,
        lr=extract_param("lr", args),
        n_hidden=extract_param("n_hidden", args),
        n_gnn_layers=extract_param("n_gnn_layers", args),
        loss="ce",
        w_ce1=extract_param("w_ce1", args),
        w_ce2=extract_param("w_ce2", args),
        dropout=extract_param("dropout", args),
        final_dropout=extract_param("final_dropout", args),
        n_heads=extract_param("n_heads", args) if args.model == 'gat' else None,
    )

    add_arange_ids([tr_data, val_data, te_data])
    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, args)

    model = get_model(tr_data, config, args)

    if args.reverse_mp:
        # metadata 추출용으로 첫 배치 hetero 변환
        sample = next(iter(tr_loader))
        hsample = homo_to_hetero(sample, args)
        hsample['node', 'to', 'node'].edge_attr     = hsample['node', 'to', 'node'].edge_attr[:, 1:]
        hsample['node', 'rev_to', 'node'].edge_attr = hsample['node', 'rev_to', 'node'].edge_attr[:, 1:]
        model = to_hetero(model, hsample.metadata(), aggr='mean')
        logging.info(summary(model, hsample.x_dict, hsample.edge_index_dict, hsample.edge_attr_dict))

    if args.finetune:
        model, optimizer = load_model(model, device, args, config, data_config)
    else:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    scheduler = ReduceLROnPlateau(
        optimizer, mode='max',
        factor=getattr(args, 'lr_factor', 0.5),
        patience=getattr(args, 'lr_patience', 5),
        min_lr=1e-6,
    )

    n_pos = int(tr_data.y.sum().item())
    n_neg = int((tr_data.y == 0).sum().item())
    auto_w_ce2 = (n_neg / max(n_pos, 1)) ** 0.5
    logging.info(f"Train IR: {n_pos/(n_pos+n_neg)*100:.4f}% — auto w_ce2={auto_w_ce2:.1f} (config={config.w_ce2:.2f})")

    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.FloatTensor([config.w_ce1, auto_w_ce2]).to(device))

    run_name   = args.unique_name
    tb_log_dir = data_config["paths"].get("tb_log_dir", "runs")
    writer = SummaryWriter(log_dir=str(Path(tb_log_dir) / run_name))
    logging.info(f"TensorBoard log dir: {str(Path(tb_log_dir) / run_name)}")

    if args.reverse_mp:
        model, best_te_result = train_hetero(
            tr_loader, val_loader, te_loader,
            model, optimizer, scheduler, loss_fn, args, config, device, data_config, writer)
    else:
        model, best_te_result = train_homo(
            tr_loader, val_loader, te_loader,
            model, optimizer, scheduler, loss_fn, args, config, device, data_config, writer)

    writer.close()
    return best_te_result, model
