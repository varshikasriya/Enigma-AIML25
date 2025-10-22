# Generate plots:
# - Dataset scatter (both)
# - Decision boundaries (LR, Perceptron)
# - Accuracy bars from metrics.csv
# - NEW: LR Loss vs. Iterations, Perceptron Mistakes per Epoch
#
# Requires: matplotlib, pandas, numpy

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    load_csv_xy, train_test_split, standardize_fit, standardize_apply, accuracy
)
from linear_regression import LinearRegressionGD
from perceptron import Perceptron


# ----------------------
# Basic dataset plotting
# ----------------------

def plot_dataset_scatter(path: str, title: str, outpath: str):
    df = pd.read_csv(path)
    n0 = (df["label"] == 0).sum()
    n1 = (df["label"] == 1).sum()
    plt.figure(figsize=(6, 4))
    plt.scatter(df["x1"], df["x2"], c=df["label"], s=15)
    plt.title(f"{title}  (class0={n0}, class1={n1})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# -------------------------
# Train splits & boundaries
# -------------------------

def fit_for_boundary(path: str, standardize: bool):
    X, y, _ = load_csv_xy(path)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
    if standardize:
        mu, sd = standardize_fit(Xtr)
        Xtr = standardize_apply(Xtr, mu, sd)
        Xte = standardize_apply(Xte, mu, sd)
    return np.array(Xtr), np.array(ytr)


def _meshgrid_from_X(X: np.ndarray, pads: float = 0.5, N: int = 250):
    x_min, x_max = X[:, 0].min() - pads, X[:, 0].max() + pads
    y_min, y_max = X[:, 1].min() - pads, X[:, 1].max() + pads
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, N),
                         np.linspace(y_min, y_max, N))
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray, title: str, outpath: str):
    xx, yy, grid = _meshgrid_from_X(X)
    Z = np.array(model.predict(grid)).reshape(xx.shape)
    acc_tr = accuracy(y.tolist(), model.predict(X.tolist()))
    plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, Z, alpha=0.30)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=15)
    plt.title(f"{title}  (train acc={acc_tr:.2%})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# --------------------------
# Convergence (no API change)
# --------------------------

def _add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])


def lr_loss_history(X: np.ndarray, y: np.ndarray, lr=0.1, max_iters=2000, tol=1e-6):
    """Re-run GD locally to record loss per iteration (matches LinearRegressionGD logic)."""
    Xb = _add_bias(X)
    n, d = Xb.shape
    w = np.zeros(d, dtype=float)
    prev = float("inf")
    losses = []
    for _ in range(1, max_iters+1):
        # preds and loss
        preds = Xb @ w
        e = preds - y
        loss = float((e @ e) / (n or 1))
        losses.append(loss)
        # grad and update
        grad = (2.0 / (n or 1)) * (Xb.T @ e)
        w -= lr * grad
        # stop on small relative improvement
        if prev != float("inf"):
            denom = abs(prev) if abs(prev) > 1e-12 else 1.0
            rel = abs(prev - loss) / denom
            if rel < tol:
                break
        prev = loss
    return losses


def perceptron_mistakes_history(X: np.ndarray, y: np.ndarray, max_iters=2000, shuffle=True, seed=0):
    """Classic perceptron update; record mistakes per epoch."""
    Xb = _add_bias(X)
    n, d = Xb.shape
    w = np.zeros(d, dtype=float)
    ybin = np.where(y == 1, 1, -1)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    mistakes = []
    for _ in range(1, max_iters+1):
        if shuffle:
            rng.shuffle(idx)
        m = 0
        for i in idx:
            pred = 1 if (w @ Xb[i]) >= 0 else -1
            if pred != ybin[i]:
                w += ybin[i] * Xb[i]
                m += 1
        mistakes.append(m)
        if m == 0:
            break
    return mistakes


def plot_lr_loss(losses: list, dataset_name: str, outpath: str):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses)+1), losses)
    plt.title(f"Linear Regression — Loss vs Iterations ({dataset_name})")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_perc_mistakes(mistakes: list, dataset_name: str, outpath: str):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(mistakes)+1), mistakes)
    plt.title(f"Perceptron — Mistakes per Epoch ({dataset_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Mistakes")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# ---------------------
# Accuracy bars (metric)
# ---------------------

def plot_accuracy_bars(metrics_csv: str, outpath: str):
    df = pd.read_csv(metrics_csv)
    datasets = df["dataset"].unique().tolist()
    models = ["linear_regression", "perceptron"]
    width = 0.35
    x = np.arange(len(datasets))
    vals_lr = [df[(df.dataset == ds) & (df.model == "linear_regression")]
               ["accuracy"].values[0] for ds in datasets]
    vals_pc = [df[(df.dataset == ds) & (df.model == "perceptron")]
               ["accuracy"].values[0] for ds in datasets]

    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width/2, vals_lr, width, label="Linear Regression")
    plt.bar(x + width/2, vals_pc, width, label="Perceptron")
    plt.xticks(x, datasets, rotation=0)
    plt.ylabel("Accuracy")
    plt.title("Model Comparison — Accuracy by Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# ------------
# Main routine
# ------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-a", required=True,
                    help="path to binary_classification.csv")
    ap.add_argument("--data-b", required=True,
                    help="path to binary_classification_non_lin.csv")
    ap.add_argument("--metrics", default="metrics.csv")
    ap.add_argument("--standardize", default="true")
    ap.add_argument("--outdir", default="plots")
    args = ap.parse_args()

    std = args.standardize.lower() in ("1", "true", "yes", "y")
    os.makedirs(args.outdir, exist_ok=True)

    # 1) dataset scatter
    plot_dataset_scatter(args.data_a, "Dataset: binary_classification",
                         os.path.join(args.outdir, "dataset_linear.png"))
    plot_dataset_scatter(args.data_b, "Dataset: binary_classification_non_lin",
                         os.path.join(args.outdir, "dataset_nonlin.png"))

    # 2) decision boundaries (train on train split)
    Xa, ya = fit_for_boundary(args.data_a, std)
    Xb, yb = fit_for_boundary(args.data_b, std)

    lr_a = LinearRegressionGD().fit(Xa.tolist(), ya.tolist())
    pc_a = Perceptron().fit(Xa.tolist(), ya.tolist())
    lr_b = LinearRegressionGD().fit(Xb.tolist(), yb.tolist())
    pc_b = Perceptron().fit(Xb.tolist(), yb.tolist())

    plot_decision_boundary(lr_a, Xa, ya,
                           "Linear Regression — binary_classification",
                           os.path.join(args.outdir, "boundary_lr_linear.png"))
    plot_decision_boundary(pc_a, Xa, ya,
                           "Perceptron — binary_classification",
                           os.path.join(args.outdir, "boundary_perc_linear.png"))
    plot_decision_boundary(lr_b, Xb, yb,
                           "Linear Regression — binary_classification_non_lin",
                           os.path.join(args.outdir, "boundary_lr_nonlin.png"))
    plot_decision_boundary(pc_b, Xb, yb,
                           "Perceptron — binary_classification_non_lin",
                           os.path.join(args.outdir, "boundary_perc_nonlin.png"))

    # 3) accuracy bars
    plot_accuracy_bars(args.metrics, os.path.join(
        args.outdir, "accuracy_bars.png"))

    # 4) NEW: convergence curves (train on same train splits, standardized the same way)
    losses_a = lr_loss_history(Xa, ya, lr=0.1, max_iters=2000, tol=1e-6)
    losses_b = lr_loss_history(Xb, yb, lr=0.1, max_iters=2000, tol=1e-6)
    plot_lr_loss(losses_a, "binary_classification",
                 os.path.join(args.outdir, "lr_loss_linear.png"))
    plot_lr_loss(losses_b, "binary_classification_non_lin",
                 os.path.join(args.outdir, "lr_loss_nonlin.png"))

    mistakes_a = perceptron_mistakes_history(
        Xa, ya, max_iters=2000, shuffle=True, seed=0)
    mistakes_b = perceptron_mistakes_history(
        Xb, yb, max_iters=2000, shuffle=True, seed=0)
    plot_perc_mistakes(mistakes_a, "binary_classification",
                       os.path.join(args.outdir, "perc_mistakes_linear.png"))
    plot_perc_mistakes(mistakes_b, "binary_classification_non_lin", os.path.join(
        args.outdir, "perc_mistakes_nonlin.png"))

    print(f"Saved plots to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
