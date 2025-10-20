# level2-models/SamyuktaGade/perceptron.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class Perceptron:
    lr: float = 1.0
    max_epochs: int = 1000
    fit_intercept: bool = True
    shuffle: bool = True
    random_state: Optional[int] = 42
    verbose: bool = False
    early_stopping: bool = False
    patience: int = 10

    w_: Optional[np.ndarray] = None
    epochs_run_: int = 0
    converged_: bool = False
    mistake_history_: list[int] = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        return np.c_[np.ones((X.shape[0], 1)), X]

    def fit(self, X: np.ndarray, y01: np.ndarray) -> "Perceptron":
        rng = np.random.default_rng(self.random_state)
        Xb = self._add_intercept(X.astype(float))
        n, d = Xb.shape
        self.w_ = np.zeros(d)
        y = np.where(y01 > 0, 1, -1)
        self.mistake_history_ = []

        best_mistakes = np.inf
        no_improve = 0

        for epoch in range(1, self.max_epochs + 1):
            idx = np.arange(n)
            if self.shuffle:
                rng.shuffle(idx)

            mistakes = 0
            for i in idx:
                xi = Xb[i]
                yi = y[i]
                s = float(xi @ self.w_)
                if yi * s <= 0.0:
                    self.w_ += self.lr * yi * xi
                    mistakes += 1

            self.epochs_run_ = epoch
            self.mistake_history_.append(mistakes)

            if self.verbose:
                accuracy = np.mean(self.predict(X) == y01)
                print(f"[Epoch {epoch}] Mistakes: {mistakes}, Train Acc: {accuracy:.4f}")

            if mistakes == 0:
                self.converged_ = True
                if self.verbose:
                    print(f"Perceptron converged in {epoch} epochs.")
                break

            # Early stopping logic
            if self.early_stopping:
                if mistakes < best_mistakes:
                    best_mistakes = mistakes
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                    break

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Xb = self._add_intercept(X.astype(float))
        return Xb @ self.w_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) > 0.0).astype(int)
