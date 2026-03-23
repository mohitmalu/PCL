from dataclasses import dataclass


@dataclass
class Config:
    data_root: str = "./data/esc-50/esc50_embeddings"
    dataset: str = "ESC-50"  # ESC-50 or GSC
    seed: int = 42
    epochs_per_task: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_tasks: int = 5
    n_classes: int = 50  # if dataset == "ESC-50"
    lambda_ewc: int = 1000
    # lambda_ewc: float = 1e7
    lambda_lwf: int = 3
    k_min: int = 1
    k_max: int = 10
    dist_metric: str = "euclidean"  # euclidean or cosine
    dist_threshold: float = 1.5
    fold: int = 1
    embedding_type: str = "clap"  # ast or wav2vec2
    reg: str = "ewc"  # ewc or lwf
    model_type: str = "cnn2"  # resnet18 or cnn2 or cnn4
    