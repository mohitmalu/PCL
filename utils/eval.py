import torch
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import numpy as np
from torch.utils.data import DataLoader, Subset
import random


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def test(model, dataloader, criterion, device, task_name="Test"):
    model.eval()
    total, correct, losses = 0, 0, []
    all_pred, all_true = [], []
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(loss.detach().cpu().item())
        pred = logits.argmax(1)
        total += yb.size(0)
        correct += (pred == yb).sum().item()
        all_pred.extend(pred.cpu().numpy())
        all_true.extend(yb.cpu().numpy())
    acc = correct / max(1, total)
    loss = float(np.mean(losses)) if losses else 0.0
    f1 = f1_score(all_true, all_pred, average="weighted") if total > 0 else 0.0
    rec = recall_score(all_true, all_pred, average="weighted") if total > 0 else 0.0
    print(f"{task_name} | Loss: {loss:.5f}, Acc: {acc:.3f}, F1: {f1:.3f}, Recall: {rec:.3f}")
    # wandb.log({"Test": task_name, "Task Loss": loss, "Task Accuracy": acc, "Task F1": f1, "Task Recall": rec})
    return acc, loss, f1, rec


# Get data loaders for each cluster within a task
def get_task_cluster_data(z_test_t, test_ds_t, k_means_model):
    task_loaders = {}
    # Get the test data for the given task_name
    # Get the cluster assignments for the test data
    task_cluster_ids = k_means_model.predict(z_test_t)
    centroids = k_means_model.cluster_centers_
    for cluster_idx, _ in enumerate(centroids):
        # Get the test data for a given cluster
        cluster_idxs = np.where(task_cluster_ids == cluster_idx)[0]
        cluster_data = Subset(test_ds_t, cluster_idxs)
        cluster_loader = DataLoader(cluster_data, batch_size=32, shuffle=False)
        task_loaders[cluster_idx] = cluster_loader
    return task_loaders

# Get data loaders for each cluster within a task
def get_task_cluster_data_kmedoids(z_test_t, test_ds_t, medoids_old, dist_metric="euclidean"):
    task_loaders = {}
    # Get the test data for the given task_name
    # Get the cluster assignments for the test data
    if dist_metric == "euclidean":
        task_cluster_ids = np.argmin(euclidean_distances(z_test_t, medoids_old), axis=1)
    elif dist_metric == "cosine":
        task_cluster_ids = np.argmax(cosine_similarity(z_test_t, medoids_old), axis=1)
    else:
        raise ValueError(f"Unknown distance metric: {dist_metric}")
    for cluster_idx, _ in enumerate(medoids_old):
        # Get the test data for a given cluster
        cluster_idxs = np.where(task_cluster_ids == cluster_idx)[0]
        cluster_data = Subset(test_ds_t, cluster_idxs)
        cluster_loader = DataLoader(cluster_data, batch_size=32, shuffle=False)
        task_loaders[cluster_idx] = cluster_loader
    return task_loaders


# Evaluate performance across all clusters/models for a task
@torch.no_grad()
def task_performance(model, task_loaders, criterion, device='cpu', task_name="Task"):
    total, correct = 0, 0
    losses = []
    all_pred, all_true = [], []
    for cluster_idx, loader in task_loaders.items():
        model[cluster_idx].eval()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model[cluster_idx](xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())
            pred = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(yb.cpu().numpy())
    acc = correct / max(1, total)
    loss = float(np.mean(losses)) if losses else 0.0
    f1 = f1_score(all_true, all_pred, average='weighted') if total > 0 else 0.0
    rec = recall_score(all_true, all_pred, average='weighted') if total > 0 else 0.0
    print(f"{task_name} | Task Loss: {loss:.5f}, Task Accuracy: {acc:.3f}, Task F1: {f1:.3f}, Task Recall: {rec:.3f}")
    # wandb.log({"Test": task_name, "Task Loss": loss, "Task Accuracy": acc, "Task F1": f1, "Task Recall": rec})
    return acc, loss, f1, rec


## Evaluate performance across all tasks and clusters/models
@torch.no_grad()
def global_performance(model, task_loaders_dict, criterion, device='cpu'):
    total, correct = 0, 0
    losses = []
    all_pred, all_true = [], []
    # Check if this can be done (Mostly there must be an issue with the ordereing of the models (cluster ids) for each task)
    for task_name, task_loaders in task_loaders_dict.items():
        for cluster_idx, loader in task_loaders.items():
            model[cluster_idx].eval()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model[cluster_idx](xb)
                loss = criterion(logits, yb)
                losses.append(loss.item())
                pred = logits.argmax(dim=1)
                total += yb.size(0)
                correct += (pred == yb).sum().item()
                all_pred.extend(pred.cpu().numpy())
                all_true.extend(yb.cpu().numpy())
    acc = correct / max(1, total)
    loss = float(np.mean(losses)) if losses else 0.0
    f1 = f1_score(all_true, all_pred, average='weighted') if total > 0 else 0.0
    rec = recall_score(all_true, all_pred, average='weighted') if total > 0 else 0.0
    print(f"Task Accuracy: {acc:.3f}, Task Loss: {loss:.3f}, Task F1: {f1:.3f}, Task Recall: {rec:.3f}")
    return acc, loss, f1, rec
