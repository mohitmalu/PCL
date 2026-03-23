import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score
from tqdm.auto import tqdm
from ewc import _KD_loss, compute_fisher
from copy import deepcopy, copy


def train_model_ewc(train_loader, model, device, epochs=10, task_name=None,
                    known_classes=0, total_classes=0, lambda_ewc=1.0,
                    label_smoothing=0.0, lr=1e-3, wd=1e-4,
                    old_fisher_information=None, importance=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    # Update fisher information and importance after a task
    if task_name == 'task_1':
        importance = None
        old_fisher_information = None

    with tqdm(total=epochs, desc="Train", unit="epoch") as progress:
        for epoch in range(epochs):
            model.train()
            total_loss, total_ewc_loss, correct, total = 0.0, 0.0, 0, 0
            all_preds, all_targets = [], []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits[:, known_classes:], yb-known_classes)
                ewc_loss = 0
                if importance is not None and old_fisher_information is not None:
                    for name, p in model.named_parameters():
                        if name in old_fisher_information and p.requires_grad:
                            ewc_loss += 1/2 * (old_fisher_information[name] * (p[:len(importance[name])] - importance[name]).pow(2)).sum()
                loss = loss + lambda_ewc * ewc_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                total_ewc_loss += lambda_ewc * ewc_loss.item() if importance is not None else 0.0
                preds = logits.argmax(1)
                total += yb.size(0)
                correct += (preds == yb).sum().item()
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(yb.cpu().tolist())
            acc = correct / max(1, total)
            avg_loss = total_loss / max(1, len(train_loader))
            avg_ewc_loss = total_ewc_loss / max(1, len(train_loader))
            f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
            rec = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
            progress.update(1)
            progress.set_postfix({'epoch': epoch+1, 
                                  "task_name": task_name, 
                                  "train_loss": avg_loss, 
                                  "train_acc": acc, 
                                  "train_f1": f1, 
                                  "ewc_loss": avg_ewc_loss})

    
    # alpha = 1 # Weight for old fisher information
    if old_fisher_information is None:
        new_fisher_information = compute_fisher(model, train_loader, criterion, device)
    else:
        alpha = known_classes / total_classes if total_classes > 0 else 0.0 
        new_fisher_information = compute_fisher(model, train_loader, criterion, device)
        for n,p in new_fisher_information.items():
            new_fisher_information[n][:len(old_fisher_information[n])] += (
                alpha * old_fisher_information[n]) + ((1 - alpha) * new_fisher_information[n][:len(old_fisher_information[n])])
    # new_fisher_information = compute_fisher(model, train_loader, criterion, device)
    importance = {n:p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    old_fisher_information = new_fisher_information.copy()
    return model, acc, avg_loss, f1, rec, old_fisher_information, importance


def train_model_LwF(train_loader, model, device, epochs=10, task_name=None,
                    known_classes=0, total_classes=0, lambda_lwf=1.0,
                    label_smoothing=0.0, lr=1e-3, wd=1e-4):
    # DL_loss = DistillationLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if task_name == 'task_1':
        old_model = None
    else:
        old_model = deepcopy(model)

    # Update fisher information and importance after a task
    with tqdm(total=epochs, desc="Train", unit="epoch") as progress:
        for epoch in range(epochs):
            model.train()
            total_loss, total_lwf_loss, correct, total = 0.0, 0.0, 0, 0
            all_preds, all_targets = [], []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                new_logits = model(xb)
                loss = criterion(new_logits[:, known_classes:], yb - known_classes)
                loss_kl = 0
                if old_model is not None:
                    with torch.no_grad():
                        old_logits = old_model(xb)
                        # Since the update FC is done on old model as well, we need to slice the logits
                        # loss_kl = DL_loss(new_logits[:, :known_classes], old_logits[:, :known_classes])
                        loss_kl = _KD_loss(new_logits[:, :known_classes], old_logits[:, :known_classes], T=2)
                loss = loss + lambda_lwf * loss_kl
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                total_lwf_loss += lambda_lwf * loss_kl.item() if old_model is not None else 0.0
                preds = new_logits.argmax(1)
                total += yb.size(0)
                correct += (preds == yb).sum().item()
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(yb.cpu().tolist())
            acc = correct / max(1, total)
            avg_loss = total_loss / max(1, len(train_loader))
            avg_lwf_loss = total_lwf_loss / max(1, len(train_loader))
            f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
            rec = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
            progress.update(1)
            progress.set_postfix({'epoch': epoch+1, 
                                  "task_name": task_name, 
                                  "train_loss": avg_loss, 
                                  "train_acc": acc, 
                                  "train_f1": f1, 
                                  "lwf_loss": avg_lwf_loss})
    return model, acc, avg_loss, f1, rec,


def train_model_naive(train_loader, model, device, epochs=10, task_name=None, known_classes=0,
                      label_smoothing=0.0, lr=1e-3, wd=1e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    with tqdm(total=epochs, desc="Train", unit="epoch") as progress:
        for epoch in range(epochs):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            all_preds, all_targets = [], []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits[:, known_classes:], yb - known_classes)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                preds = logits.argmax(1)
                total += yb.size(0)
                correct += (preds == yb).sum().item()
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(yb.cpu().tolist())
            acc = correct / max(1, total)
            avg_loss = total_loss / max(1, len(train_loader))
            f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
            rec = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
            progress.update(1)
            progress.set_postfix({'epoch': epoch+1, 
                                  "task_name": task_name, 
                                  "train_loss": avg_loss, 
                                  "train_acc": acc, 
                                  "train_f1": f1})
    return model, acc, avg_loss, f1, rec

    