import torch
import tqdm
from sklearn.metrics import f1_score
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero, save_model, load_model
from models import GINe, PNA, GATe, RGCN, GINe_FHE, Model_Wrapper
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree
from torch.utils.data import TensorDataset
from concrete.ml.torch.compile import compile_brevitas_qat_model
import numpy as np
import wandb
import logging
import time
import tempfile
from pathlib import Path

def compile_and_test(tr_loader, te_loader, inds, torch_model, tr_data, te_data, args, use_sim=True):
    X_test = te_data.x
    y_test = te_data.y

    for batch in tqdm.tqdm(te_loader, disable=not args.tqdm):
        batch = batch
        break

    #remove the unique edge id from the edge features, as it's no longer needed
    batch.edge_attr = batch.edge_attr[:, 1:]

    #wrapped_model = Model_Wrapper(torch_model)
    wrapped_model = Model_Wrapper(torch_model, batch.x, batch.edge_index, batch.edge_attr)

    onnx_model_path = "debug_gnn_model.onnx"

    # Compile the model
    print("Compiling the model")
    print("batch.x", batch.x)

    start_compile = time.time()
    quantized_numpy_module = compile_brevitas_qat_model(
        wrapped_model,  # Our model
        batch.x,  # A representative input-set to be used for both quantization and compilation\
        output_onnx_file=onnx_model_path,
        verbose=True
    )

    end_compile = time.time()
    print(f"Compilation finished in {end_compile - start_compile:.2f} seconds")

    # Check that the network is compatible with FHE constraints
    bitwidth = quantized_numpy_module.fhe_circuit.graph.maximum_integer_bit_width()
    print(
        f"Max bit-width: {bitwidth} bits" + " -> it works in FHE!!"
        if bitwidth <= 16
        else " too high for FHE computation"
    )

    # Execute prediction using simulation
    # (not encrypted but fast, and results are equivalent)

    if not use_sim:
        print("Generating key")
        start_key = time.time()
        quantized_numpy_module.fhe_circuit.keygen()
        end_key = time.time()
        print(f"Key generation finished in {end_key - start_key:.2f} seconds")

    fhe_mode = "simulate" if use_sim else "execute"

    predictions = np.zeros_like(y_test)

    preds = []
    ground_truths = []
    for batch in tqdm.tqdm(te_loader, disable=not args.tqdm):
        #select the seed edges from which the batch was created
        inds = inds.detach().cpu()
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = te_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        #add the seed edges that have not been sampled to the batch
        missing = ~torch.isin(batch_edge_ids, batch.edge_attr[:, 0].detach().cpu())

        if missing.sum() != 0 and (args.data == 'Small_J' or args.data == 'Small_Q'):
            missing_ids = batch_edge_ids[missing].int()
            n_ids = batch.n_id
            add_edge_index = te_data.edge_index[:, missing_ids].detach().clone()
            node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
            add_edge_index = torch.tensor([[node_mapping[val.item()] for val in row] for row in add_edge_index])
            add_edge_attr = te_data.edge_attr[missing_ids, :].detach().clone()
            add_y = te_data.y[missing_ids].detach().clone()
        
            batch.edge_index = torch.cat((batch.edge_index, add_edge_index), 1)
            batch.edge_attr = torch.cat((batch.edge_attr, add_edge_attr), 0)
            batch.y = torch.cat((batch.y, add_y), 0)

            mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))

        #remove the unique edge id from the edge features, as it's no longer needed
        batch.edge_attr = batch.edge_attr[:, 1:]

        print("Starting inference")
        start_infer = time.time()
        pred = quantized_numpy_module.forward(batch.x).argmax(1)
        end_infer = time.time()

        print(f"Compilation finished in {end_compile - start_compile:.2f} seconds")
        if not use_sim:
            print(f"Key generation finished in {end_key - start_key:.2f} seconds")
            print(
                f"Inferences finished in {end_infer - start_infer:.2f} seconds "
                f"({(end_infer - start_infer)/len(batch.x):.2f} seconds/sample)"
            )

        preds.append(pred)
        ground_truths.append(batch.y[mask])

    # Compute accuracy
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    accuracy = np.mean(pred == ground_truth) * 100
    print(f"Test Quantized Accuracy: {accuracy:.2f}% on {len(X_test)} examples.")
    f1 = f1_score(ground_truth, pred)

    return bitwidth, accuracy, f1, predictions, quantized_numpy_module

def train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, tr_data, val_data, te_data):
    best_te_f1 = 0
    total_training_time = 0
    total_val_time = 0
    total_test_time = 0
    for epoch in range(config.epochs):
        logging.info(f"\nEpoch {epoch}")
        #training
        train_start_time = time.time()
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            #select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch.input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            #remove the unique edge id from the edge features, as it's no longer needed
            batch.edge_attr = batch.edge_attr[:, 1:]

            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            pred = out[mask]
            ground_truth = batch.y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(ground_truth)
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        wandb.log({"f1/train": f1}, step=epoch)
        logging.info(f'Train F1: {f1:.4f}')

        train_time = time.time() - train_start_time
        total_training_time += train_time
        wandb.log({"time/train": train_time}, step=epoch)
        logging.info(f'Training time: {train_time:.2f}s')

        #evaluate
        val_start_time = time.time()
        val_f1 = evaluate_homo(val_loader, val_inds, model, val_data, device, args)
        val_time = time.time() - val_start_time
        total_val_time += val_time

        test_start_time = time.time()
        if args.fhe:
            _, _, te_f1, clear_prediction, vl_quantized_numpy_module = compile_and_test(
                tr_loader, te_loader, te_inds, model, tr_data.x, te_data, args, use_sim=True
            )
        else:
           te_f1 = evaluate_homo(te_loader, te_inds, model, te_data, device, args)
        test_time = time.time() - test_start_time
        total_test_time += test_time

        wandb.log({"time/val": val_time}, step=epoch)
        wandb.log({"f1/validation": val_f1}, step=epoch)
        wandb.log({"time/test": test_time}, step=epoch)
        wandb.log({"f1/test": te_f1}, step=epoch)
        logging.info(f'Validation time: {val_time:.2f}s')
        logging.info(f'Validation F1: {val_f1:.4f}')
        logging.info(f'Test time: {test_time:.2f}s')
        logging.info(f'Test F1: {te_f1:.4f}')

        if epoch == 0:
            wandb.log({"best_test_f1": te_f1}, step=epoch)
        elif te_f1 > best_te_f1:
            best_te_f1 = te_f1
            wandb.log({"best_test_f1": te_f1}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args)
        
        if epoch == config.epochs-1: #if last epoch
            mean_training_time = total_training_time/epoch
            mean_val_time = total_val_time/epoch
            mean_test_time = total_test_time/epoch
            logging.info(f'Best Test F1: {te_f1:.5f}')
            logging.info(f'Mean Training Time per epoch: {mean_training_time:.5f}s')
            wandb.run.summary["time/mean_training_time"] = mean_training_time
            logging.info(f'Mean Validation Time per epoch: {mean_val_time:.5f}s')
            wandb.run.summary["time/mean_val_time"] = mean_val_time
            logging.info(f'Mean Test Time per epoch: {mean_test_time:.5f}s')
            wandb.run.summary["time/mean_test_time"] = mean_test_time
    
    return model

def train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data):
    #training
    best_val_f1 = 0
    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            #select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)
            
            #remove the unique edge id from the edge features, as it's no longer needed
            batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
            batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]

            batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            out = out[('node', 'to', 'node')]
            pred = out[mask]
            ground_truth = batch['node', 'to', 'node'].y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(batch['node', 'to', 'node'].y[mask])
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
            
        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        wandb.log({"f1/train": f1}, step=epoch)
        logging.info(f'Train F1: {f1:.4f}')

        #evaluate
        val_f1 = evaluate_hetero(val_loader, val_inds, model, val_data, device, args)
        te_f1 = evaluate_hetero(te_loader, te_inds, model, te_data, device, args)

        wandb.log({"f1/validation": val_f1}, step=epoch)
        wandb.log({"f1/test": te_f1}, step=epoch)
        logging.info(f'Validation F1: {val_f1:.4f}')
        logging.info(f'Test F1: {te_f1:.4f}')

        if epoch == 0:
            wandb.log({"best_test_f1": te_f1}, step=epoch)
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wandb.log({"best_test_f1": te_f1}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args)
        
    return model

def get_model(sample_batch, config, args):
    n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
    e_dim = (sample_batch.edge_attr.shape[1] - 1) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1)

    if args.model == "gin":
        model = GINe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), residual=False, edge_updates=args.emlps, edge_dim=e_dim, 
                dropout=config.dropout, final_dropout=config.final_dropout
                )
    elif args.model == "gin_fhe":
        model = GINe_FHE(
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
    elif config.model == "rgcn":
        model = RGCN(
            num_features=n_feats, edge_dim=e_dim, num_relations=8, num_gnn_layers=round(config.n_gnn_layers),
            n_classes=2, n_hidden=round(config.n_hidden),
            edge_update=args.emlps, dropout=config.dropout, final_dropout=config.final_dropout, n_bases=None #(maybe)
        )
    
    return model

def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args):
    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #define a model config dictionary and wandb logging at the same time
    wandb.init(
        mode="disabled" if args.testing else "online",
        project="your_proj_name", #replace this with your wandb project name if you want to use wandb logging

        config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None
        }
    )

    config = wandb.config

    #set the transform if ego ids should be used
    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None

    #add the unique ids to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    #get the model
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)
    if args.finetune:
        model, optimizer = load_model(model, device, args, config)
    else:
        model = get_model(sample_batch, config, args)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')
    
    sample_x = sample_batch.x if not isinstance(sample_batch, HeteroData) else sample_batch.x_dict
    sample_edge_index = sample_batch.edge_index if not isinstance(sample_batch, HeteroData) else sample_batch.edge_index_dict
    if isinstance(sample_batch, HeteroData):
        sample_batch['node', 'to', 'node'].edge_attr = sample_batch['node', 'to', 'node'].edge_attr[:, 1:]
        sample_batch['node', 'rev_to', 'node'].edge_attr = sample_batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
    else:
        sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]
    sample_edge_attr = sample_batch.edge_attr if not isinstance(sample_batch, HeteroData) else sample_batch.edge_attr_dict
    logging.info(summary(model, sample_x, sample_edge_index, sample_edge_attr))
    
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device))

    if args.reverse_mp:
        model = train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data)
    else:
        model = train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, tr_data, val_data, te_data)
    
    wandb.finish()