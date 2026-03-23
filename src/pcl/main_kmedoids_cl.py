import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import argparse
import sys
import json
import torch
import numpy as np
from collections import OrderedDict, Counter
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

from config import Config
from data_split import prepare_continual_learning
from clustering import calc_clusters_kmedoids
from models import models_assign, update_fc
from train import train_model_ewc, train_model_LwF
from eval import get_task_cluster_data_kmedoids, test, task_performance, set_seed

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
    ap.add_argument("--dist_threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_type", type=str, default="cnn2")  # resnet18 or cnn2
    ap.add_argument("--reg", type=str, default="ewc")  # ewc or lwf
    ap.add_argument("--embedding_type", type=str, default="clap")  # clap or wav2vec2
    ap.add_argument("--fold", type=int, default=1)  # 1 to 5
    ap.add_argument("--dist_metric", type=str, default="euclidean")  # euclidean or cosine
    ap.add_argument("--dataset", type=str, default="ESC-50")  # ESC-50
    ap.add_argument("--n_tasks", type=int, default=5)
    ap.add_argument("--n_classes", type=int, default=50)
    ap.add_argument("--lambda_ewc", type=int, default=1000)
    ap.add_argument("--lambda_lwf", type=int, default=3)
    ap.add_argument("--k_min", type=int, default=1)
    ap.add_argument("--k_max", type=int, default=10)
    args = ap.parse_args()
    
    for fold in range(1):
        # Coniiguration setup
        cfg = Config(
            data_root=args.data,
            epochs_per_task=args.epochs_per_task, 
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dist_threshold=args.dist_threshold,
            seed=args.seed,
            model_type=args.model_type,
            reg=args.reg,
            embedding_type=args.embedding_type,
            fold=fold,
            dist_metric=args.dist_metric,
            dataset=args.dataset,
            n_tasks=args.n_tasks,
            n_classes=args.n_classes,
            lambda_ewc=args.lambda_ewc,
            lambda_lwf=args.lambda_lwf,
            k_min=args.k_min,
            k_max=args.k_max
        )
        print(f"\nConfig: {cfg}")
        print(f"Running k-medoids with \n model_type: {cfg.model_type} \n reg: {cfg.reg} \n embedding_type: {cfg.embedding_type} \n dist_threshold: {cfg.dist_threshold}")
        sys.stdout.flush()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.is_available():
        #     print(nvidia-smi)
        print(f"\n Using device: {device}")
        set_seed(cfg.seed)

        # Load data
        if cfg.dataset == 'ESC-50':
            print("Loading ESC-50 dataset.")
            X = np.load(f"{cfg.data_root}/mel_spectrograms.npy")   # (N, 1, 128, T')
            Y = np.load(f"{cfg.data_root}/labels.npy")              # (N,)
            if cfg.embedding_type == "clap":
                Z = np.load(f"{cfg.data_root}/clap_embeddings.npy")     # (N, 512)
            elif cfg.embedding_type == "wav2vec2":
                Z = np.load(f"{cfg.data_root}/wav2vec2_embeddings.npy")     # (N, 512)
            elif cfg.embedding_type == "ast":
                Z = np.load(f"{cfg.data_root}/ast_embeddings.npy")     # (N, 512)
            else:
                raise ValueError(f"Unknown embedding type: {cfg.embedding_type}")
            

        tasks_classes, task_loaders = prepare_continual_learning(
            X, Y, Z,
            n_classes=cfg.n_classes,
            base_classes=30,
            inc_classes=5,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
        )

        # Main continual learning initializations
        models_old, fishers_old, importances_old = OrderedDict(), OrderedDict(), OrderedDict()
        task_train_accuracy_list, task_train_loss_list = [], []
        task_test_accuracy_list, task_test_loss_list = [], []
        task_weighted_f1_list, task_weighted_recall_list = [], []
        z_extended = np.array([])
        kmedoids_model_dict = {}
        task_test_performance_dict = {}
        overall_test_performance_dict = {}
        test_ds_t_list = []
        z_test_t_list = []
        medoids_old = np.array([])
        criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
        task_counts = 0
        known_classes = 0

        # Main continual learning loop
        for t in range(cfg.n_tasks):
            task_counts += 1
            task_name = f"task_{t+1}"
            print(f"\n=== Task {t+1}/{cfg.n_tasks} ===")

            train_dataset = task_loaders[t]["train"].dataset
            x_train_t = train_dataset.X.numpy()
            y_train_t = train_dataset.Y.numpy()
            z_train_t = train_dataset.Z.numpy()

            test_dataset = task_loaders[t]["test"].dataset
            x_test_t = test_dataset.X.numpy()
            y_test_t = test_dataset.Y.numpy()
            z_test_t = test_dataset.Z.numpy()
            
            train_ds_t = TensorDataset(torch.tensor(x_train_t, dtype=torch.float32),
                                    torch.tensor(y_train_t, dtype=torch.long))
            test_ds_t  = TensorDataset(torch.tensor(x_test_t, dtype=torch.float32),
                                    torch.tensor(y_test_t, dtype=torch.long))
            
            test_ds_t_list.append(test_ds_t)
            z_test_t_list.append(z_test_t)
            total_classes = known_classes + len(np.unique(y_train_t))
            print(f"Known classes: {known_classes}, Total classes after this task: {total_classes}, Classes: {np.unique(y_train_t).tolist()}")

            # Extend the embeddings with the medoids of the previous tasks
            if len(medoids_old) > 0:
                # Calculate the frequency of each medoid
                medoid_freq = np.array(pd.Series(kmedoids_model.labels_).value_counts().sort_index())
                # Concatenate the medoids of the previous tasks with the frequency of each medoid
                z_medoids = np.concatenate([medoids_old[i:i+1].T@np.ones((1,medoid_freq[i])) for i in range(len(medoid_freq))], axis=1).T
                # Concatenate the embeddings of the current task with the medoids of the previous tasks
                z_extended = np.concatenate((z_train_t, z_medoids), axis=0)
            else:
                # If no previous tasks, use the embeddings of the current task
                z_extended = z_train_t

            # Calculate the clusters
            kmedoids_model, medoids_new = calc_clusters_kmedoids(
                z_extended,
                k_min=max(cfg.k_min, len(medoids_old)),
                k_max=max(cfg.k_max, len(medoids_old)),
                dist_threshold=cfg.dist_threshold,
                seed=cfg.seed,
                distance_metric=cfg.dist_metric
            )
            kmedoids_model_dict[task_name] = kmedoids_model

            # Get cluster ids
            if cfg.dist_metric == "euclidean":
                z_train_clus_ids = np.argmin(euclidean_distances(z_train_t, medoids_new), axis=1)
                z_test_clus_ids  = np.argmin(euclidean_distances(z_test_t, medoids_new), axis=1)
            elif cfg.dist_metric == "cosine":
                z_train_clus_ids = np.argmax(cosine_similarity(z_train_t, medoids_new), axis=1)
                z_test_clus_ids  = np.argmax(cosine_similarity(z_test_t, medoids_new), axis=1)
            else:
                raise ValueError(f"Unknown distance metric: {cfg.dist_metric}")
            unique_task_cluster_ids = np.unique([z_train_clus_ids])
            print(f"Unique cluster ids for {task_name}:", unique_task_cluster_ids)

            # Assign models to clusters and update the models, fishers, and importances
            models, fishers, importances, _ = models_assign(
                task_name, 
                medoids_new, 
                medoids_old, 
                models_old, 
                fishers_old,
                importances_old,
                device=device,
                model_type=cfg.model_type,
                num_classes=total_classes
            )
            models_old = models.copy()
            fishers_old = fishers.copy()
            importances_old = importances.copy()
            medoids_old = medoids_new.copy()

            # Train/evaluate per cluster
            cluster_train_accuracy_list, cluster_train_loss_list    = [], []
            cluster_test_accuracy_list, cluster_test_loss_list      = [], []
            cluster_weighted_f1_list, cluster_weighted_recall_list  = [], []

            # Train/evaluate per cluster
            for cluster_idx in range(len(medoids_new)):
                curr_cluster_model = models_old[cluster_idx].to(device)
                curr_cluster_model = update_fc(curr_cluster_model, total_classes)
                if cluster_idx not in unique_task_cluster_ids:
                    print(f"Skipping training for cluster {cluster_idx} as no data points assigned.")
                    continue
                cluster_train_indices = np.where(z_train_clus_ids == cluster_idx)[0]
                cluster_test_indices  = np.where(z_test_clus_ids == cluster_idx)[0]
                # Now index into the dataset, not the tuple
                clustered_train_ds = Subset(train_ds_t, cluster_train_indices)
                clustered_test_ds  = Subset(test_ds_t,  cluster_test_indices)
                print("Shapes of clustered train and test data in cluster", cluster_idx, ":",
                    len(clustered_train_ds), ",", len(clustered_test_ds))
                print(f"\nClasses in clustered train data for cluster {cluster_idx}:", np.unique([y.item() for _, y in clustered_train_ds]))
                print(f"Distribution of classes in clustered train data for cluster {cluster_idx}:\n",
                      Counter([y.item() for _, y in clustered_train_ds]))
                print(f"\nClasses in clustered test data for cluster {cluster_idx}:", np.unique([y.item() for _, y in clustered_test_ds]))
                print(f"Distribution of classes in clustered test data for cluster {cluster_idx}:\n",
                      Counter([y.item() for _, y in clustered_test_ds]))
                clus_train_loader = DataLoader(clustered_train_ds, batch_size=cfg.batch_size, shuffle=True)
                clus_test_loader  = DataLoader(clustered_test_ds,  batch_size=cfg.batch_size, shuffle=False)

                if cfg.reg == 'lwf':
                    curr_cluster_model, _, _, _, _,  = train_model_LwF(
                        clus_train_loader, 
                        curr_cluster_model, 
                        device=device,
                        epochs=cfg.epochs_per_task,
                        task_name=task_name,
                        known_classes=known_classes,
                        total_classes=total_classes,
                        lambda_lwf=cfg.lambda_lwf,
                        lr=cfg.lr,
                        wd=cfg.weight_decay
                    )
                elif cfg.reg == 'ewc':
                    curr_cluster_model, _, _, _, _, curr_old_fisher_information, curr_importance = train_model_ewc(
                        clus_train_loader, 
                        curr_cluster_model, 
                        device=device,
                        epochs=cfg.epochs_per_task,
                        task_name=task_name,
                        known_classes=known_classes,
                        total_classes=total_classes,
                        lambda_ewc=cfg.lambda_ewc,
                        lr=cfg.lr,
                        wd=cfg.weight_decay,
                        old_fisher_information=None if t==0 else fishers_old[cluster_idx],
                        importance=None if t==0 else importances_old[cluster_idx]
                    )
                    # Update fisher information and importance
                    fishers_old[cluster_idx] = curr_old_fisher_information
                    importances_old[cluster_idx] = curr_importance
                elif cfg.reg == 'icarl':
                    pass
                else:
                    raise ValueError(f"Unknown regularization: {cfg.reg}")
                # Update model
                models_old[cluster_idx] = curr_cluster_model

                # Evaluate
                print("\nClustered training data evaluation:")
                cluster_train_acc, cluster_train_loss, _, _ = \
                    test(
                        curr_cluster_model,
                        clus_train_loader, 
                        criterion, 
                        device=device, 
                        task_name=f"Train {task_name}_cluster_{cluster_idx}"
                    )
                
                print("\nClustered test data evaluation:")
                cluster_test_acc, cluster_test_loss, cluster_weighted_f1_test, cluster_weighted_recall_test = \
                    test(
                        curr_cluster_model, 
                        clus_test_loader, 
                        criterion, 
                        device=device, 
                        task_name=f"Test {task_name}_cluster_{cluster_idx}"
                    )
                print("------------------------------------------------------\n")

                # Collect metrics
                cluster_train_accuracy_list.append(cluster_train_acc)
                cluster_train_loss_list.append(cluster_train_loss)
                cluster_test_accuracy_list.append(cluster_test_acc)
                cluster_test_loss_list.append(cluster_test_loss)
                cluster_weighted_f1_list.append(cluster_weighted_f1_test)
                cluster_weighted_recall_list.append(cluster_weighted_recall_test)

            known_classes = total_classes
            # Store per-task metrics
            task_train_accuracy_list.append(cluster_train_accuracy_list)
            task_train_loss_list.append(cluster_train_loss_list)
            task_test_accuracy_list.append(cluster_test_accuracy_list)
            task_test_loss_list.append(cluster_test_loss_list)
            task_weighted_f1_list.append(cluster_weighted_f1_list)
            task_weighted_recall_list.append(cluster_weighted_recall_list)

            # ---- Evaluate current model mapping across ALL seen tasks ----
            print('================== Printing evaluations for previous tasks on current model =================')
            prev_task_perf = {}
            overall_prev_task_perf = []
            for prev_task_idx in range(cfg.n_tasks)[:task_counts]:
                prev_task_name = f"task_{prev_task_idx+1}"
                print(f"Evaluating {prev_task_name} on model at the end of {task_name}")
                prev_task_cluster_perf = {}
                tester_loaders = get_task_cluster_data_kmedoids(
                    z_test_t_list[prev_task_idx],
                    test_ds_t_list[prev_task_idx],
                    medoids_old,
                    dist_metric=cfg.dist_metric
                    )
                for cluster_idx, loader in enumerate(tester_loaders.values()):
                    print(f"Number of samples in cluster {cluster_idx} for {prev_task_name}: {len(loader.dataset)}")
                    acc_p, loss_p, f1_p, rec_p = test(
                        models_old[cluster_idx],
                        loader, 
                        criterion, 
                        device=device, 
                        task_name=f"{prev_task_name}_cluster_{cluster_idx}"
                    )
                    prev_task_cluster_perf[cluster_idx] = (acc_p, loss_p, f1_p, rec_p)   
                prev_task_perf[prev_task_name] = prev_task_cluster_perf
                print(f"=============== Overall task performance for task_{prev_task_name} ===============")
                acc_t, loss_t, f1_t, rec_t = task_performance(
                    models_old, 
                    tester_loaders, 
                    criterion, 
                    device=device, 
                    task_name=f"{prev_task_name}"
                    )
                overall_prev_task_perf.append([acc_t, f1_t])
                print(f"=================================================================================")
            task_test_performance_dict[task_name] = prev_task_perf
            overall_test_performance_dict[task_name] = (overall_prev_task_perf, kmedoids_model.n_clusters)
            # Dump results to json

        if cfg.dataset == 'ESC-50':
            save_dir = "../results/esc-50"
        else:
            raise ValueError(f"Unknown dataset: {cfg.dataset}")
        # os.makedirs(save_dir, exist_ok=True)
        with open(f'{save_dir}/{cfg.dataset}_{cfg.model_type}_kmedoids_{cfg.dist_metric}_{cfg.reg}_thr_{cfg.dist_threshold}_{cfg.embedding_type}_fold_{cfg.fold}.json', 'w') as f:
            json.dump({
                "overall_test_performance_dict": overall_test_performance_dict,
            }, f, indent=4)

        print(f"\nDone. Model was trained sequentially over {cfg.n_tasks} tasks with K-medoids and {cfg.reg} regularizers.")
        # wandb.finish()

if __name__ == "__main__":
    main()
