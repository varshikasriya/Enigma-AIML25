# Tiny helpers: data I/O, split, standardize, metrics, timing, save

from typing import List, Tuple, Sequence, Dict, Any
import csv
import json
import time
import os


def load_csv_xy(path: str, label_col: str = "label") -> Tuple[List[List[float]], List[int], List[str]]:
    """Load CSV → (X, y, feature_names). Assumes numeric features and 0/1 'label'."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        feats = [c for c in reader.fieldnames if c !=
                 label_col]  # type: ignore
        X, y = [], []
        for row in reader:
            X.append([float(row[c]) for c in feats])
            y.append(int(row[label_col]))
    return X, y, feats


def train_test_split(X: Sequence[Sequence[float]], y: Sequence[int], test_size: float = 0.2):
    """Deterministic split by index (no shuffle)."""
    n = len(X)
    n_test = int(n * test_size)
    n_train = n - n_test
    Xtr = [list(X[i]) for i in range(n_train)]
    ytr = [y[i] for i in range(n_train)]
    Xte = [list(X[i]) for i in range(n_train, n)]
    yte = [y[i] for i in range(n_train, n)]
    return Xtr, Xte, ytr, yte


def standardize_fit(X: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    """Compute mean/std per feature (std≥1e-12)."""
    n = len(X)
    d = len(X[0]) if n else 0
    means = [0.0]*d
    for row in X:
        for j, v in enumerate(row):
            means[j] += v
    means = [m / n for m in means]
    var = [0.0]*d
    for row in X:
        for j, v in enumerate(row):
            dv = v - means[j]
            var[j] += dv*dv
    stds = [(v/max(n, 1))**0.5 for v in var]
    stds = [s if s > 1e-12 else 1.0 for s in stds]
    return means, stds


def standardize_apply(X: Sequence[Sequence[float]], means: Sequence[float], stds: Sequence[float]) -> List[List[float]]:
    """Apply z-score using provided mean/std."""
    Z = []
    for row in X:
        Z.append([(row[j]-means[j])/stds[j] for j in range(len(row))])
    return Z


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Simple accuracy."""
    n = len(y_true)
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / float(n or 1)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def time_block() -> float:
    """Return perf_counter() — call twice to diff."""
    return time.perf_counter()


def avg_predict_time_us(predict_fn, X: Sequence[Sequence[float]], loops: int = 100) -> float:
    """Average microseconds per sample over multiple loops."""
    # warmup
    _ = predict_fn(X)
    start = time.perf_counter()
    for _ in range(loops):
        _ = predict_fn(X)
    end = time.perf_counter()
    total_preds = loops * len(X)
    return ((end - start) / float(total_preds or 1)) * 1e6


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_csv(rows: List[List[Any]], header: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
