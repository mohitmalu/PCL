import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

def load_gsc_data(data_root, embedding_type, fold):
    # Robust path checks
    x_path = os.path.join(data_root, "X_mel_spectrograms.npy")
    y_path = os.path.join(data_root, "Y_labels.npy")
    # File existence check
    assert os.path.exists(x_path) and os.path.exists(y_path), "Data files missing"
    X = np.load(x_path)
    X = np.expand_dims(X, axis=1)  # Add channel dimension
    y = np.load(y_path)
    z_types = dict(
        clap="Z_clap_embeddings.npy",
        wav2vec2="Z_wav2vec2_embeddings.npy",
        whisper="Z_whisper_embeddings.npy"
    )
    z_path = os.path.join(data_root, z_types.get(embedding_type, "Z_whisper_embeddings.npy"))
    assert os.path.exists(z_path), "Embeddings file missing"
    Z = np.load(z_path)

    # Split the data into K-folds for K-fold cross-validation
    KF = StratifiedKFold(n_splits=5, shuffle=False) # No shuffling to maintain consistency
    train_indices, test_indices = list(KF.split(X, y))[fold]

    # For each fold create a task splits
    task_KF = StratifiedKFold(n_splits=4, shuffle=False) # No shuffling to maintain consistency
    fold_train_indices = list(task_KF.split(X[train_indices], y[train_indices]))
    task_train_indices = [t[1] for t in fold_train_indices]
    fold_test_indices = list(task_KF.split(X[test_indices], y[test_indices]))
    task_test_indices = [t[1] for t in fold_test_indices]

    return X, y, Z, task_train_indices, task_test_indices


def load_esc50_data(data_root, embedding_type, fold, task):
    # Robust path checks
    x_path_train = os.path.join(data_root, f"f{fold}_mel_spectrograms_train_{task}.npy")
    y_path_train = os.path.join(data_root, f"f{fold}_labels_train_{task}.npy")
    x_path_test = os.path.join(data_root, f"f{fold}_mel_spectrograms_test_{task}.npy")
    y_path_test = os.path.join(data_root, f"f{fold}_labels_test_{task}.npy")
    print(x_path_train, y_path_train, x_path_test, y_path_test)
    # File existence check
    assert os.path.exists(x_path_train) and os.path.exists(y_path_train), "Data files missing"
    assert os.path.exists(x_path_test) and os.path.exists(y_path_test), "Data files missing"
    # Load data
    x_train_t = np.load(x_path_train)
    y_train_t = np.load(y_path_train)
    x_test_t = np.load(x_path_test)
    y_test_t = np.load(y_path_test)

    # Load embeddings based on the specified type
    z_train_types = dict(
        clap=f"f{fold}_clap_embeddings_train_{task}.npy",
        wav2vec2=f"f{fold}_wav2vec2_embeddings_train_{task}.npy",
        whisper=f"f{fold}_whisper_embeddings_train_{task}.npy"
    )
    z_test_types = dict(
        clap=f"f{fold}_clap_embeddings_test_{task}.npy",
        wav2vec2=f"f{fold}_wav2vec2_embeddings_test_{task}.npy",
        whisper=f"f{fold}_whisper_embeddings_test_{task}.npy"
    )
    # Check paths and load embeddings
    z_path_train = os.path.join(
        data_root,
        z_train_types.get(embedding_type, f"f{fold}_whisper_embeddings_train_{task}.npy")
    )
    assert os.path.exists(z_path_train), "Embeddings file missing"
    z_train_t = np.load(z_path_train)

    z_path_test = os.path.join(
        data_root,
        z_test_types.get(embedding_type, f"f{fold}_whisper_embeddings_test_{task}.npy")
    )
    assert os.path.exists(z_path_test), "Embeddings file missing"
    z_test_t = np.load(z_path_test)

    return x_train_t, y_train_t, z_train_t, x_test_t, y_test_t, z_test_t
