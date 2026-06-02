import copy
import time
import torch
import tqdm
import datetime
from pathlib import Path
from types import SimpleNamespace
from sklearn.metrics import f1_score, recall_score, precision_score, average_precision_score, log_loss
from torch.utils.tensorboard import SummaryWriter
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero, save_model, load_model
from models import GINe, PNA, GATe, RGCN
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree
import logging

def _log_best(best_epoch, best_val, best_te, total_time_s, peak_memory_mb):
    logging.info('Training complete.')
    logging.info(f'Best epoch: {best_epoch}')
    logging.info(f'Total training time: {total_time_s:.1f}s | Avg epoch memory: {peak_memory_mb:.1f}MB')
    logging.info(f"  Val  — F1: {best_val['f1']:.4f} | Recall: {best_val['recall']:.4f} | Precision: {best_val['precision']:.4f} | AUPRC: {best_val['auprc']:.4f} | LogLoss: {best_val['log_loss']:.4f} | Mem: {best_val['memory_mb']:.1f}MB | Time: {best_val['time_s']:.1f}s")
    logging.info(f"  Test — F1: {best_te['f1']:.4f} | Recall: {best_te['recall']:.4f} | Precision: {best_te['precision']:.4f} | AUPRC: {best_te['auprc']:.4f} | LogLoss: {best_te['log_loss']:.4f} | Mem: {best_te['memory_mb']:.1f}MB | Time: {best_te['time_s']:.1f}s")

def _write_metrics(writer, tr_result, val_result, te_result, epoch):
    for metric in ('f1', 'recall', 'precision', 'auprc', 'log_loss'):
        writer.add_scalars(metric.upper(), {
            'train': tr_result[metric],
            'val':   val_result[metric],
            'test':  te_result[metric],
        }, epoch)

def train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config, writer):
    best_val_f1 = 0
    best_val_result = best_te_result = None
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    memory_mb_list = []
    t_train_start = time.perf_counter()
    for epoch in range(config.epochs):
        logging.info(f"[epoch {epoch + 1}/{config.epochs}]")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        total_loss = total_examples = 0
        preds = []
        pred_probas = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch.input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            batch.edge_attr = batch.edge_attr[:, 1:]

            batch.to(device)
            try:
                out = model(batch.x, batch.edge_index, batch.edge_attr)
            except ValueError as e:
                if 'Expected more than 1 value per channel' in str(e):
                    logging.warning(f"Small batch skipped (BatchNorm): {batch.x.shape[0]} nodes | {e}")
                    continue
                raise
            pred = out[mask]
            ground_truth = batch.y[mask]
            threshold = getattr(args, 'threshold', 0.5)
            preds.append((pred.softmax(dim=-1)[:, 1] >= threshold).long())
            pred_probas.append(pred.softmax(dim=-1)[:, 1].detach().cpu())
            ground_truths.append(ground_truth)
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        if torch.cuda.is_available():
            memory_mb_list.append(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
        else:
            memory_mb_list.append(0.0)

        if not preds:
            logging.warning(f"Epoch {epoch}: 모든 배치 skip됨 (BatchNorm 오류). 다음 에폭으로 진행합니다.")
            continue

        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        pred_proba = torch.cat(pred_probas, dim=0).numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        tr_result = {
            'f1':        f1_score(ground_truth, pred, zero_division=0),
            'recall':    recall_score(ground_truth, pred, zero_division=0),
            'precision': precision_score(ground_truth, pred, zero_division=0),
            'auprc':     average_precision_score(ground_truth, pred_proba),
            'log_loss':  log_loss(ground_truth, pred_proba),
        }
        logging.info(f"Train F1: {tr_result['f1']:.4f} | Recall: {tr_result['recall']:.4f} | Precision: {tr_result['precision']:.4f} | AUPRC: {tr_result['auprc']:.4f} | LogLoss: {tr_result['log_loss']:.4f}")

        val_result = evaluate_homo(val_loader, val_inds, model, val_data, device, args)
        te_result  = evaluate_homo(te_loader,  te_inds,  model, te_data,  device, args)

        logging.info(f"Val  — F1: {val_result['f1']:.4f} | Recall: {val_result['recall']:.4f} | Precision: {val_result['precision']:.4f} | AUPRC: {val_result['auprc']:.4f} | LogLoss: {val_result['log_loss']:.4f} | Mem: {val_result['memory_mb']:.1f}MB | Time: {val_result['time_s']:.1f}s")
        logging.info(f"Test — F1: {te_result['f1']:.4f} | Recall: {te_result['recall']:.4f} | Precision: {te_result['precision']:.4f} | AUPRC: {te_result['auprc']:.4f} | LogLoss: {te_result['log_loss']:.4f} | Mem: {te_result['memory_mb']:.1f}MB | Time: {te_result['time_s']:.1f}s")

        _write_metrics(writer, tr_result, val_result, te_result, epoch)

        if val_result['f1'] > best_val_f1:
            best_val_f1 = val_result['f1']
            best_val_result = val_result
            best_te_result = te_result
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)
        else:
            patience_counter += 1
            if args.patience is not None and patience_counter >= args.patience:
                logging.info(f'Early stopping at epoch {epoch + 1} (patience={args.patience})')
                break

    total_time_s = time.perf_counter() - t_train_start
    avg_memory_mb = sum(memory_mb_list) / len(memory_mb_list) if memory_mb_list else 0.0
    writer.add_scalar('Total/training_time_s', total_time_s, 0)
    writer.add_scalar('Total/avg_memory_mb', avg_memory_mb, 0)
    if best_val_result is None:
        logging.warning("학습 중 val F1 개선 없음. best 결과 없음.")
    else:
        _log_best(best_epoch, best_val_result, best_te_result, total_time_s, avg_memory_mb)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_te_result

def train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config, writer):
    best_val_f1 = 0
    best_val_result = best_te_result = None
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    memory_mb_list = []
    t_train_start = time.perf_counter()
    for epoch in range(config.epochs):
        logging.info(f"[epoch {epoch + 1}/{config.epochs}]")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        total_loss = total_examples = 0
        preds = []
        pred_probas = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
            batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]

            batch.to(device)
            try:
                out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            except ValueError as e:
                if 'Expected more than 1 value per channel' in str(e):
                    n_nodes = batch['node'].x.shape[0]
                    logging.warning(f"Small batch skipped (BatchNorm): {n_nodes} nodes | {e}")
                    continue
                raise
            out = out[('node', 'to', 'node')]
            pred = out[mask]
            ground_truth = batch['node', 'to', 'node'].y[mask]
            threshold = getattr(args, 'threshold', 0.5)
            preds.append((pred.softmax(dim=-1)[:, 1] >= threshold).long())
            pred_probas.append(pred.softmax(dim=-1)[:, 1].detach().cpu())
            ground_truths.append(batch['node', 'to', 'node'].y[mask])
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        if torch.cuda.is_available():
            memory_mb_list.append(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
        else:
            memory_mb_list.append(0.0)

        if not preds:
            logging.warning(f"Epoch {epoch}: 모든 배치 skip됨 (BatchNorm 오류). 다음 에폭으로 진행합니다.")
            continue

        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        pred_proba = torch.cat(pred_probas, dim=0).numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        tr_result = {
            'f1':        f1_score(ground_truth, pred, zero_division=0),
            'recall':    recall_score(ground_truth, pred, zero_division=0),
            'precision': precision_score(ground_truth, pred, zero_division=0),
            'auprc':     average_precision_score(ground_truth, pred_proba),
            'log_loss':  log_loss(ground_truth, pred_proba),
        }
        logging.info(f"Train F1: {tr_result['f1']:.4f} | Recall: {tr_result['recall']:.4f} | Precision: {tr_result['precision']:.4f} | AUPRC: {tr_result['auprc']:.4f} | LogLoss: {tr_result['log_loss']:.4f}")

        val_result = evaluate_hetero(val_loader, val_inds, model, val_data, device, args)
        te_result  = evaluate_hetero(te_loader,  te_inds,  model, te_data,  device, args)

        logging.info(f"Val  — F1: {val_result['f1']:.4f} | Recall: {val_result['recall']:.4f} | Precision: {val_result['precision']:.4f} | AUPRC: {val_result['auprc']:.4f} | LogLoss: {val_result['log_loss']:.4f} | Mem: {val_result['memory_mb']:.1f}MB | Time: {val_result['time_s']:.1f}s")
        logging.info(f"Test — F1: {te_result['f1']:.4f} | Recall: {te_result['recall']:.4f} | Precision: {te_result['precision']:.4f} | AUPRC: {te_result['auprc']:.4f} | LogLoss: {te_result['log_loss']:.4f} | Mem: {te_result['memory_mb']:.1f}MB | Time: {te_result['time_s']:.1f}s")

        _write_metrics(writer, tr_result, val_result, te_result, epoch)

        if val_result['f1'] > best_val_f1:
            best_val_f1 = val_result['f1']
            best_val_result = val_result
            best_te_result = te_result
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)
        else:
            patience_counter += 1
            if args.patience is not None and patience_counter >= args.patience:
                logging.info(f'Early stopping at epoch {epoch + 1} (patience={args.patience})')
                break

    total_time_s = time.perf_counter() - t_train_start
    avg_memory_mb = sum(memory_mb_list) / len(memory_mb_list) if memory_mb_list else 0.0
    writer.add_scalar('Total/training_time_s', total_time_s, 0)
    writer.add_scalar('Total/avg_memory_mb', avg_memory_mb, 0)
    if best_val_result is None:
        logging.warning("학습 중 val F1 개선 없음. best 결과 없음.")
    else:
        _log_best(best_epoch, best_val_result, best_te_result, total_time_s, avg_memory_mb)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_te_result

def get_model(sample_batch, config, args):
    n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
    e_dim = (sample_batch.edge_attr.shape[1] - 1) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1)

    if args.model == "gin":
        model = GINe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), residual=False, edge_updates=args.emlps, edge_dim=e_dim,
                dropout=config.dropout, final_dropout=config.final_dropout
                )
    elif args.model == "gat":
        model = GATe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), n_heads=round(config.n_heads),
                edge_updates=args.emlps, edge_dim=e_dim,
                dropout=config.dropout, final_dropout=config.final_dropout
                )
    elif args.model == "pna":
        if not isinstance(sample_batch, HeteroData):
            d = degree(sample_batch.edge_index[1], dtype=torch.long)
        else:
            index = torch.cat((sample_batch['node', 'to', 'node'].edge_index[1], sample_batch['node', 'rev_to', 'node'].edge_index[1]), 0)
            d = degree(index, dtype=torch.long)
        deg = torch.bincount(d, minlength=1)
        model = PNA(
            num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
            n_hidden=round(config.n_hidden), edge_updates=args.emlps, edge_dim=e_dim,
            dropout=config.dropout, deg=deg, final_dropout=config.final_dropout
            )
    elif args.model == "rgcn":
        model = RGCN(
            num_features=n_feats, edge_dim=e_dim, num_relations=8, num_gnn_layers=round(config.n_gnn_layers),
            n_classes=2, n_hidden=round(config.n_hidden),
            edge_update=args.emlps, dropout=config.dropout, final_dropout=config.final_dropout, n_bases=None
        )

    return model

def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = SimpleNamespace(
        epochs=args.n_epochs,
        batch_size=args.batch_size,
        model=args.model,
        data=args.data,
        num_neighbors=args.num_neighs,
        lr=extract_param("lr", args),
        n_hidden=extract_param("n_hidden", args),
        n_gnn_layers=extract_param("n_gnn_layers", args),
        loss="ce",
        w_ce1=extract_param("w_ce1", args),
        w_ce2=extract_param("w_ce2", args),
        dropout=extract_param("dropout", args),
        final_dropout=extract_param("final_dropout", args),
        n_heads=extract_param("n_heads", args) if args.model == 'gat' else None
    )

    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None

    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)

    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')

    if args.finetune:
        model, optimizer = load_model(model, device, args, config, data_config)
    else:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    sample_batch.to(device)
    sample_x = sample_batch.x if not isinstance(sample_batch, HeteroData) else sample_batch.x_dict
    sample_edge_index = sample_batch.edge_index if not isinstance(sample_batch, HeteroData) else sample_batch.edge_index_dict
    if isinstance(sample_batch, HeteroData):
        sample_batch['node', 'to', 'node'].edge_attr = sample_batch['node', 'to', 'node'].edge_attr[:, 1:]
        sample_batch['node', 'rev_to', 'node'].edge_attr = sample_batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
    else:
        sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]
    sample_edge_attr = sample_batch.edge_attr if not isinstance(sample_batch, HeteroData) else sample_batch.edge_attr_dict
    logging.info(summary(model, sample_x, sample_edge_index, sample_edge_attr))

    if getattr(args, 'weighted_sampler', False):
        loss_fn = torch.nn.CrossEntropyLoss()
        logging.info("WeightedRandomSampler 활성화: CE loss 가중치 [1, 1] (균등)")
    else:
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device))

    run_name   = args.unique_name
    tb_log_dir = data_config["paths"].get("tb_log_dir", "runs")
    tb_run_dir = str(Path(tb_log_dir) / run_name)
    writer = SummaryWriter(log_dir=tb_run_dir)
    logging.info(f"TensorBoard log dir: {tb_run_dir}")

    if args.reverse_mp:
        model, best_te_result = train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config, writer)
    else:
        model, best_te_result = train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config, writer)

    writer.close()
    return best_te_result, model
