import torch
import torch.nn as nn
from torchvision.models import resnet18
import math
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from collections import OrderedDict
from copy import deepcopy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleLinear(nn.Module):
    '''
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


# PyTorch model
# ----------------------------
class CNN2(nn.Module):
    def __init__(self, input_dim=(128, 501), num_classes=50, in_channels=1, p_drop=0.2):
        super(CNN2, self).__init__()
        self.in_channels = in_channels
        self.input_len = input_dim[0]
        self.input_width = input_dim[1]   # fixed
        self.kernel_size = 3

        # 2 conv blocks with wider kernels
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=self.kernel_size, padding=3)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=self.kernel_size, padding=3)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.gap = nn.AdaptiveAvgPool2d((4,4))

        feat_dim = self._calc_feat_dim(self.input_len, self.input_width, self.in_channels)
        self.fc1 = nn.Linear(feat_dim, 128)
        self.dropout = nn.Dropout(p_drop)
        # self.fc2 = nn.Linear(128, num_classes)
        self.fc = nn.Linear(128, num_classes)

        # Kaiming init (optional but helps)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _calc_feat_dim(self, input_len, input_width, in_channels):
        with torch.no_grad():
            device = next(self.parameters()).device
            x = torch.zeros(1, in_channels, input_len, input_width, device=device)
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.gap(x)
            return x.numel() # flattened feature dimension

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, L, W)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.gap(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc(x)
    

class CNN4(nn.Module):
    def __init__(self, input_dim=(128, 501), num_classes=50, in_channels=1, p_drop=0.2):
        super(CNN4, self).__init__()
        self.in_channels = in_channels
        self.input_len = input_dim[0]
        self.input_width = input_dim[1]   # fixed
        self.kernel_size = 3

        # 2 conv blocks with wider kernels
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=self.kernel_size, padding=3)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=self.kernel_size, padding=3)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=self.kernel_size, padding=3)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=self.kernel_size, padding=3)
        self.bn4   = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)

        self.gap = nn.AdaptiveAvgPool2d((4,4))

        feat_dim = self._calc_feat_dim(self.input_len, self.input_width, self.in_channels)
        self.fc1 = nn.Linear(feat_dim, 128)
        self.dropout = nn.Dropout(p_drop)
        # self.fc2 = nn.Linear(128, num_classes)
        self.fc = nn.Linear(128, num_classes)

        # Kaiming init (optional but helps)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _calc_feat_dim(self, input_len, input_width, in_channels):
        with torch.no_grad():
            device = next(self.parameters()).device
            x = torch.zeros(1, in_channels, input_len, input_width, device=device)
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            x = self.pool4(F.relu(self.bn4(self.conv4(x))))
            x = self.gap(x)
            return x.numel() # flattened feature dimension

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, L, W)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.gap(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc(x)



def create_model(model_type, num_classes, device, in_channels=1):
    if model_type == "resnet18":
        model = resnet18(weights=None).to(device)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(model.conv1.weight, nonlinearity="relu")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        nn.init.normal_(model.fc.weight, 0.0, 0.01)
        nn.init.zeros_(model.fc.bias)
        return model
    elif model_type == "cnn2":
        return CNN2(num_classes=num_classes).to(device)
    elif model_type == "cnn4":
        return CNN4(num_classes=num_classes).to(device)
    raise ValueError(f"Unknown model_type '{model_type}'")


def update_fc(model, nb_classes):
    fc = generate_fc(model.fc.in_features, nb_classes).to(device)
    if model.fc is not None:
        nb_output = model.fc.out_features
        weight = deepcopy(model.fc.weight.data)
        bias = deepcopy(model.fc.bias.data)
        fc.weight.data[:nb_output] = weight
        fc.bias.data[:nb_output] = bias

    del model.fc
    model.fc = fc
    return model


def generate_fc(in_dim, out_dim):
    fc = SimpleLinear(in_dim, out_dim)
    return fc


# Assign models to clusters
def models_assign(task_name, centroids_new, centroids_old, models_old, fishers_old, importances_old, device='cpu', model_type='resnet18', num_classes=8):
    print(f"\nAssigning models for {task_name} with new centroids length {len(centroids_new)} and old centroids length {len(centroids_old)}...")
    models, fishers, importances = OrderedDict(), OrderedDict(), OrderedDict()
    model_count = 0
    if task_name == 'task_1':
        # First task, create models for each cluster
        for idx, centroid in enumerate(centroids_new):
            # print(f"Centroids for the {task_name} = {centroid}")
            models[idx] = create_model(model_type, num_classes=num_classes, device=device)
            fishers[idx] = None
            importances[idx] = None
            model_count += 1
    else:
        # Assign models based on centroid distances to previous centroids
        centroid_dists = cdist(centroids_old, centroids_new, metric='euclidean')
        models = OrderedDict()
        for idx, centroid_old in enumerate(centroids_old):
            # If the distance is greater than the threshold, create a new model
            c_new_idx = np.argmin(centroid_dists[idx, :])
            # if the new index is already assigned, select the next closest index
            next_idx = 1
            while c_new_idx in models:
                c_new_idx = np.argsort(centroid_dists[idx, :])[next_idx]
                next_idx += 1
            models[c_new_idx] = models_old[idx]
            fishers[c_new_idx] = fishers_old[idx]
            importances[c_new_idx] = importances_old[idx]
            model_count += 1
            print(f"model assignments, new idx {c_new_idx}, old idx = {idx}")
        for idx, centroid_new in enumerate(centroids_new):
            if idx not in models:
                models[idx] = create_model(model_type, num_classes=num_classes, device=device)
                fishers[idx] = None
                importances[idx] = None
                model_count += 1

    print(f"Total models after {task_name}: {model_count}, Centroids: {len(centroids_new)}")
    return models, fishers, importances, model_count
