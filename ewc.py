import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_fisher(model, dataloader, criterion, device, lr=1e-3, wd=1e-4):
    model.train()
    fisher_information = {n: torch.zeros_like(p, device=device) for n,p in model.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        opt.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher_information[n] += p.grad.pow(2).clone()
    # Normalize by number of batches
    for n, p in fisher_information.items():
        fisher_information[n] = p / len(dataloader)
        # fisher_information[n] = torch.min(fisher_information[n], torch.tensor(0.0001))
    return fisher_information


# def compute_ewc(model, fisher_information, importance):
#     ewc_loss = 0
#     for name, p in model.named_parameters():
#         if name in fisher_information and p.requires_grad:
#             ewc_loss += (fisher_information[name] * (p - importance[name]).pow(2)).sum()

# class DistillationLoss(nn.Module):
#     """
#     Knowledge distillation loss for Learning without Forgetting (LwF).
#     """
#     def __init__(self, temperature=2.0):
#         super().__init__()
#         self.temperature = temperature
#         self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
#     def forward(self, student_logits, teacher_logits):
#         """
#         Args:
#             student_logits: Logits from the current model
#             teacher_logits: Logits from the old model
#         """
#         soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
#         soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)
        
#         # KL divergence loss
#         loss = self.kl_div(soft_prob, soft_targets) * (self.temperature ** 2)
        
#         return loss
    
def _KD_loss(pred, soft, T=2):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]