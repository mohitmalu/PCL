import argparse
import sys
import os
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from data_utils import load_esc50_data, load_gsc_data
from train import train_model_naive
from models import create_model
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
    ap.add_argument("--fold", type=int, default=1)  # 1 to 5
    ap.add_argument("--dataset", type=str, default="GSC")  # GSC or ESC-50
    args = ap.parse_args()
    
    for fold in range(5):
        # Configuration setup
        cfg = Config(
            data_root=args.data,
            epochs_per_task=args.epochs_per_task, 
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            model_type=args.model_type,
            fold=fold,
            dataset=args.dataset
        )
        print(f"\nConfig: {cfg}")
        print(f"\nRunning NO CL with \n model_type: {cfg.model_type}")
        sys.stdout.flush()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        print(f"\nUsing device: {device}")
        set_seed(cfg.seed)

        # Load data
        if cfg.dataset=="GSC" and cfg.data_root.endswith("GSC_data/embeddings"):
            print("Loading GSC dataset.")
            X, y, _, task_train_indices, task_test_indices\
                = load_gsc_data(cfg.data_root, cfg.embedding_type, cfg.fold) 
            
        x_train = []
        y_train = []
        x_test  = []
        y_test  = []
        for t in range(cfg.n_tasks):
            task_name = f"task_{t+1}"
            print(f"\n=== Task {t+1}/{cfg.n_tasks} ===")
            if cfg.dataset=='ESC-50' and cfg.data_root.endswith("ESC-50-master/new_CLAP_embeddings"):
                print(f"Loading ESC-50 dataset for {task_name}.")
                x_train_t, y_train_t, _, x_test_t, y_test_t, _ = \
                    load_esc50_data(cfg.data_root, cfg.embedding_type, cfg.fold+1, t)
            elif cfg.dataset=='GSC' and cfg.data_root.endswith("GSC_data/embeddings"):
                print(f"Loading GSC dataset for {task_name}.")
                x_train_t = X[task_train_indices[t]]
                y_train_t = y[task_train_indices[t]]
                # z_train_t = Z[task_train_indices[t]]
                x_test_t = X[task_test_indices[t]]
                y_test_t = y[task_test_indices[t]]
                # z_test_t = Z[task_test_indices[t]]
            else:
                raise ValueError("Unsupported dataset. Use ESC-50 or GSC.")
            x_train.append(x_train_t)
            y_train.append(y_train_t)
            x_test.append(x_test_t)
            y_test.append(y_test_t)

        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        x_test  = np.concatenate(x_test, axis=0)
        y_test  = np.concatenate(y_test, axis=0)

        # Create DataLoaders
        train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                torch.tensor(y_train, dtype=torch.long))
        test_ds  = TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                torch.tensor(y_test, dtype=torch.long))

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False)

        # Initialize model
        model = create_model(cfg.model_type, num_classes=10 if cfg.dataset == 'GSC' else 50, device=device)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()

        # Train
        model, _, _, _, _ = train_model_naive(
            train_loader,
            model,
            device=device, 
            epochs=cfg.epochs_per_task,
            task_name=task_name, 
            lambda_ewc=cfg.lambda_ewc,
            lr=cfg.lr,
            wd=cfg.weight_decay
        )

        te_acc, te_loss, te_f1, te_rec = test(
            model, 
            test_loader, 
            criterion, 
            device=device, 
        )

        if cfg.dataset == 'GSC':
            save_dir = "/home/mmalu/CL_HAR/code/modular_code/results_gsc"
        elif cfg.dataset == 'ESC-50':
            save_dir = "/home/mmalu/CL_HAR/code/modular_code/results"
        os.makedirs(save_dir, exist_ok=True)
        # Save the results to JSON
        with open(f'{save_dir}/{cfg.dataset}_{cfg.model_type}_nocl_epochs_{cfg.epochs_per_task}_fold{cfg.fold}.json', 'w') as f:
            json.dump({
                "final_test_performance": {
                    "test_loss": te_loss,
                    "test_acc": te_acc,
                    "test_f1": te_f1,
                    "test_recall": te_rec
                }
            }, f, indent=4)

        print("\nDone. Model was trained on all data at once (no CL).")

if __name__ == "__main__":
    main()
