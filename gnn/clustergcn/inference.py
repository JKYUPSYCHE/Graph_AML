import torch
import logging
from types import SimpleNamespace
from train_util import (homo_to_hetero, extract_param, add_arange_ids, get_loaders,
                        evaluate_homo_cluster, evaluate_hetero_cluster, load_model)
from training import get_model
from torch_geometric.nn import to_hetero


def infer_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
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
        sample = next(iter(tr_loader))
        if getattr(args, 'ego', False):
            sample.x = torch.cat([sample.x, torch.ones(sample.x.shape[0], 1)], dim=1)
        hsample = homo_to_hetero(sample, args)
        hsample['node', 'to', 'node'].edge_attr     = hsample['node', 'to', 'node'].edge_attr[:, 1:]
        hsample['node', 'rev_to', 'node'].edge_attr = hsample['node', 'rev_to', 'node'].edge_attr[:, 1:]
        model = to_hetero(model, hsample.metadata(), aggr='mean')

    logging.info("=> loading model checkpoint")
    checkpoint = torch.load(
        f'{data_config["paths"]["model_to_load"]}/checkpoint_{args.unique_name}.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    logging.info(f"=> loaded checkpoint (epoch {checkpoint['epoch']})")

    if args.reverse_mp:
        te_result = evaluate_hetero_cluster(te_loader, model, device, args, te_inds=te_inds)
    else:
        te_result = evaluate_homo_cluster(te_loader, model, device, args, te_inds=te_inds)

    logging.info(
        f"Test — F1: {te_result['f1']:.4f} | Recall: {te_result['recall']:.4f} | "
        f"Precision: {te_result['precision']:.4f} | AUPRC: {te_result['auprc']:.4f} | "
        f"Mem: {te_result['memory_mb']:.1f}MB | Time: {te_result['time_s']:.1f}s"
    )
    return te_result
