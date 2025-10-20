# Binary Perceptron (online updates). Predict 0/1, internal labels -1/+1.

from typing import List, Sequence
import random
import time


def _dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x*y
    return s


def _add_bias(X: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(row) + [1.0] for row in X]


class Perceptron:
    def __init__(self, max_iters: int = 2000, shuffle: bool = True, seed: int = 0):
        self.max_iters = max_iters
        self.shuffle = shuffle
        self.seed = seed
        self.w: List[float] = []
        self.n_iters_ = 0
        self.converged_ = False
        self.train_time_sec_ = 0.0
        self.mistakes_last_epoch_ = None

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "Perceptron":
        Xb = _add_bias(X)
        n, d = len(Xb), len(Xb[0])
        self.w = [0.0]*d
        ybin = [1 if yi == 1 else -1 for yi in y]
        idx = list(range(n))
        rng = random.Random(self.seed)
        t0 = time.perf_counter()
        for it in range(1, self.max_iters+1):
            if self.shuffle:
                rng.shuffle(idx)
            mistakes = 0
            for i in idx:
                xi = Xb[i]
                yi = ybin[i]
                pred = 1 if _dot(self.w, xi) >= 0 else -1
                if pred != yi:
                    # w <- w + yi*xi
                    for j in range(d):
                        self.w[j] += yi * xi[j]
                    mistakes += 1
            self.n_iters_ = it
            self.mistakes_last_epoch_ = mistakes
            if mistakes == 0:
                self.converged_ = True
                break
        self.train_time_sec_ = time.perf_counter() - t0
        return self

    def predict_scores(self, X: Sequence[Sequence[float]]) -> List[float]:
        Xb = _add_bias(X)
        return [_dot(self.w, xi) for xi in Xb]

    def predict(self, X: Sequence[Sequence[float]]) -> List[int]:
        return [1 if s >= 0.0 else 0 for s in self.predict_scores(X)]
