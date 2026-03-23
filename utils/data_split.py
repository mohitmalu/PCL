import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class MelSpectrogramDataset(Dataset):
    """Simple dataset wrapping mel spectrograms, labels, and CLAP embeddings."""

    def __init__(self, X, Y, Z):
        self.X = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
        self.Y = torch.tensor(Y, dtype=torch.long) if not isinstance(Y, torch.Tensor) else Y
        self.Z = torch.tensor(Z, dtype=torch.float32) if not isinstance(Z, torch.Tensor) else Z

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Z[idx]


def build_class_order(n_classes=50, base_classes=30, inc_classes=5, seed=42):
    """Randomly shuffle class indices and partition into tasks.

    Returns:
        tasks_classes: list of lists, where tasks_classes[i] contains the
                       class indices assigned to task i.
    """
    rng = np.random.RandomState(seed)
    class_order = rng.permutation(n_classes).tolist()

    tasks_classes = [class_order[:base_classes]]
    remaining = class_order[base_classes:]
    for i in range(0, len(remaining), inc_classes):
        tasks_classes.append(remaining[i : i + inc_classes])

    return tasks_classes


def get_task_dataloaders(X, Y, Z, tasks_classes, batch_size=32,
                         test_size=0.25, seed=42):
    """Split data per task into train/test DataLoaders (stratified by class) and return the loaders.

    Args:
        X: np.ndarray or Tensor of shape (N, 1, 128, T')
        Y: np.ndarray or Tensor of shape (N,)
        Z: np.ndarray or Tensor of shape (N, 512)
        tasks_classes: list of lists from build_class_order()
        batch_size: batch size for DataLoaders
        test_size: fraction for test split
        seed: random seed for reproducibility

    Returns:
        task_loaders: list of dicts, each with 'train' and 'test' DataLoaders
                      and 'classes' (the class indices for that task).
    """
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.numpy()
    if isinstance(Z, torch.Tensor):
        Z = Z.numpy()

    # Remap original class labels to sequential indices (0, 1, 2, ...)
    # so that task 1 classes -> 0..base-1, task 2 -> base..base+inc-1, etc.
    class_order = [c for cls_list in tasks_classes for c in cls_list]
    label_map = {orig: new_idx for new_idx, orig in enumerate(class_order)}
    Y_remapped = np.array([label_map[int(y)] for y in Y])

    task_loaders = []

    for task_id, cls_list in enumerate(tasks_classes):
        # Get the mask for the current task
        mask = np.isin(Y, cls_list)
        # Get the data for the current task
        X_task, Z_task = X[mask], Z[mask]
        Y_task = Y_remapped[mask]

        X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
            X_task, Y_task, Z_task,
            test_size=test_size,
            stratify=Y_task,
            random_state=seed,
        )

        train_ds = MelSpectrogramDataset(X_train, Y_train, Z_train)
        test_ds = MelSpectrogramDataset(X_test, Y_test, Z_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        remapped_cls = sorted([label_map[c] for c in cls_list])
        task_loaders.append({
            "train": train_loader,
            "test": test_loader,
            "classes": remapped_cls,
        })

        print(f"Task {task_id} | orig classes {cls_list} | remapped {remapped_cls} | "
              f"train: {len(train_ds)}, test: {len(test_ds)}")

    return task_loaders


def prepare_continual_learning(X, Y, Z, n_classes=50, base_classes=30,
                                inc_classes=5, batch_size=32, seed=42):
    """End-to-end helper: build class order and return per-task DataLoaders.

    Args:
        X: (N, 1, 128, T') mel spectrograms
        Y: (N,) integer class labels
        Z: (N, 512) CLAP embeddings
        n_classes, base_classes, inc_classes: task structure
        batch_size, seed: DataLoader config

    Returns:
        tasks_classes: the class partition (list of lists)
        task_loaders:  list of dicts with 'train'/'test' DataLoaders and 'classes'
    """
    tasks_classes = build_class_order(n_classes, base_classes, inc_classes, seed)
    task_loaders = get_task_dataloaders(X, Y, Z, tasks_classes,
                                        batch_size=batch_size, seed=seed)
    return tasks_classes, task_loaders
