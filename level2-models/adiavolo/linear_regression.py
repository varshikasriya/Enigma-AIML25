# Linear Regression (GD) for binary labels; predict via 0.5 threshold.

from typing import List, Sequence, Tuple
import math
import time


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        s += x*y
    return s


def _add_bias(X: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(row) + [1.0] for row in X]


class LinearRegressionGD:
    def __init__(self, lr: float = 0.1, max_iters: int = 2000, tol: float = 1e-6):
        self.lr = lr
        self.max_iters = max_iters
        self.tol = tol
        self.w: List[float] = []
        self.n_iters_ = 0
        self.converged_ = False
        self.train_time_sec_ = 0.0
        self.final_loss_ = None

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "LinearRegressionGD":
        Xb = _add_bias(X)
        n, d = len(Xb), len(Xb[0])
        self.w = [0.0]*d
        prev = float("inf")
        t0 = time.perf_counter()
        for it in range(1, self.max_iters+1):
            # preds and loss
            preds = [_dot(self.w, xi) for xi in Xb]
            loss = 0.0
            for pi, yi in zip(preds, y):
                e = pi - yi
                loss += e*e
            loss /= (n or 1)
            # grad = (2/n) * Xb^T (pred - y)
            grad = [0.0]*d
            for xi, pi, yi in zip(Xb, preds, y):
                e = (pi - yi) * (2.0 / (n or 1))
                for j in range(d):
                    grad[j] += e * xi[j]
            # update
            for j in range(d):
                self.w[j] -= self.lr * grad[j]
            self.n_iters_ = it
            self.final_loss_ = loss
            # stop on small relative improvement
            if prev != float("inf"):
                denom = abs(prev) if abs(prev) > 1e-12 else 1.0
                rel = abs(prev - loss)/denom
                if rel < self.tol:
                    self.converged_ = True
                    break
            prev = loss
        self.train_time_sec_ = time.perf_counter() - t0
        return self

    def predict_scores(self, X: Sequence[Sequence[float]]) -> List[float]:
        Xb = _add_bias(X)
        return [_dot(self.w, xi) for xi in Xb]

    def predict(self, X: Sequence[Sequence[float]]) -> List[int]:
        # 0/1 via 0.5 threshold
        scores = self.predict_scores(X)
        return [1 if s >= 0.5 else 0 for s in scores]
