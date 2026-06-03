import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


def normalize_epoch(epoch: np.ndarray) -> np.ndarray:
    mean = np.mean(epoch)
    std = np.std(epoch)
    if std > 0:
        return (epoch - mean) / std
    return epoch - mean


def preprocess_data(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, val_size: float = 0.25
) -> Tuple:
    X_norm = np.array([normalize_epoch(epoch) for epoch in X])
    X_norm = X_norm.astype(np.float32)
    y = y.astype(np.int64)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X_norm, y, test_size=test_size, stratify=y, random_state=42
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_ratio, stratify=y_tv, random_state=42
    )
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Val samples:   {len(X_val)}")
    print(f"Test samples:  {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test
