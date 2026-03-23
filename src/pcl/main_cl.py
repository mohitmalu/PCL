import argparse
import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from data_utils import load_esc50_data, load_gsc_data
from train import train_model_ewc, train_model_LwF
from models import create_model, update_fc
from eval import test, set_seed

import warnings
warnings.filterwarnings("ignore")

def main():
    # Argument parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--epochs_per_task", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_type", type=str, default="cnn2")  # resnet18 or cnn2
    ap.add_argument("--reg", type=str, default="ewc")  # ewc or lwf
    ap.add_argument("--fold", type=int, default=1)  # 1 to 5
    ap.add_argument("--dataset", type=str, default="GSC")  # GSC or ESC-50
    args = ap.parse_args()

    for fold in range(1):
        # Configuration setup
        cfg = Config(
            data_root=args.data,
            epochs_per_task=args.epochs_per_task, 
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            model_type=args.model_type,
            reg=args.reg,
            fold=fold,
            dataset=args.dataset
        )
        print(f"\nConfig: {cfg}")
        print(f"\n Running CL with \n model_type: {cfg.model_type} \n reg: {cfg.reg}")
        sys.stdout.flush()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        set_seed(cfg.seed)

        # Load data
        if cfg.dataset=="GSC" and cfg.data_root.endswith("GSC_data/embeddings"):
            print("Loading GSC dataset.")
            X, y, Z, task_train_indices, task_test_indices\
                = load_gsc_data(cfg.data_root, cfg.embedding_type, cfg.fold) 

        # Initializa model
        model = create_model(cfg.model_type, num_classes=1, device=device)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        task_test_loaders = {}
        overall_prev_task_perf = {}
        known_classes = 0

        # Main continual learning loop
        for t in range(cfg.n_tasks):
            task_name = f"task_{t+1}"
            print(f"\n=== Task {t+1}/{cfg.n_tasks} ===")
            if cfg.dataset=='ESC-50' and cfg.data_root.endswith("ESC_cls_incremental"):
                print(f"Loading ESC-50 dataset for {task_name}.")
                x_train_t, y_train_t, _, x_test_t, y_test_t, _ = \
                    load_esc50_data(cfg.data_root, cfg.embedding_type, cfg.fold+1, t)
            elif cfg.dataset=='GSC' and cfg.data_root.endswith("GSC_cls_incremental"):
                print(f"Loading GSC dataset for {task_name}.")
                x_train_t = X[task_train_indices[t]]
                y_train_t = y[task_train_indices[t]]
                # z_train_t = Z[task_train_indices[t]]
                x_test_t = X[task_test_indices[t]]
                y_test_t = y[task_test_indices[t]]
                # z_test_t = Z[task_test_indices[t]]
            else:
                raise ValueError("Unsupported dataset. Use ESC-50 or GSC.")
            
            train_ds_t = TensorDataset(torch.tensor(x_train_t, dtype=torch.float32),
                                    torch.tensor(y_train_t, dtype=torch.long))
            test_ds_t  = TensorDataset(torch.tensor(x_test_t, dtype=torch.float32),
                                    torch.tensor(y_test_t, dtype=torch.long))

            train_loader = DataLoader(train_ds_t, batch_size=cfg.batch_size, shuffle=True)
            test_loader  = DataLoader(test_ds_t,  batch_size=cfg.batch_size, shuffle=False)

            total_classes = known_classes + len(torch.unique(torch.tensor(y_train_t)))
            print(f"Known classes: {known_classes}, Total classes after this task: {total_classes}, Classes: {torch.unique(torch.tensor(y_train_t)).tolist()}")

            model = update_fc(model, total_classes)
            
            # Train
            if cfg.reg == 'lwf':
                model, _, _, _, _ = train_model_LwF(
                    train_loader,
                    model,
                    device=device, 
                    epochs=cfg.epochs_per_task,
                    task_name=task_name,
                    known_classes=known_classes,
                    total_classes=total_classes, 
                    lambda_lwf=cfg.lambda_lwf,
                    lr=cfg.lr,
                    wd=cfg.weight_decay
                )
            else:
                model, _, _, _, _, old_fisher_information, importance = train_model_ewc(
                    train_loader,
                    model,
                    device=device, 
                    epochs=cfg.epochs_per_task,
                    task_name=task_name, 
                    known_classes=known_classes,
                    total_classes=total_classes,
                    lambda_ewc=cfg.lambda_ewc,
                    lr=cfg.lr,
                    wd=cfg.weight_decay,
                    old_fisher_information=None if t==0 else old_fisher_information,
                    importance=None if t==0 else importance
                )
            
            known_classes = total_classes

            # Store loader for continual eval
            task_test_loaders[task_name] = test_loader

            # Eval on all seen tasks
            prev_task_perf = {}
            for name, loader in task_test_loaders.items():
                te_acc, te_loss, te_f1, te_rec = test(
                    model, 
                    loader, 
                    criterion, 
                    device=device, 
                    task_name=name
                )
                prev_task_perf[name] = (te_acc, te_f1)
            overall_prev_task_perf[task_name] = prev_task_perf

        if cfg.dataset == 'GSC':
            save_dir = "/home/mmalu/CL_HAR/code/modular_code/results_new"
        elif cfg.dataset == 'ESC-50':
            save_dir = "/home/mmalu/CL_HAR/code/modular_code/results_new"
        os.makedirs(save_dir, exist_ok=True)
        # Save the results to JSON
        with open(f'{save_dir}/{cfg.dataset}_{cfg.model_type}_{cfg.reg}_epochs_{cfg.epochs_per_task}_fold{cfg.fold}.json', 'w') as f:
            json.dump({
                "overall_previous_task_performance": overall_prev_task_perf
            }, f, indent=4)

        print("\nDone. Model was trained sequentially over 4 tasks with EWC/regularizers.")

if __name__ == "__main__":
    main()
