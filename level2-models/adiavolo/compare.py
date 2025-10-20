# Train + evaluate both models on both datasets; save metrics → JSON + CSV.
# Stdlib only.

from typing import Dict, Any, List
import argparse
import os
from utils import (
    load_csv_xy, train_test_split, standardize_fit, standardize_apply,
    accuracy, avg_predict_time_us, save_json, save_csv, now_iso
)
from linear_regression import LinearRegressionGD
# lr=0.1, max_iters=2000, tol=1e-6, threshold 0.5
from perceptron import Perceptron
# max_iters=2000, shuffle=True, seed=0


def run_one(dataset_name: str, path: str, standardize: bool) -> List[Dict[str, Any]]:
    X, y, _ = load_csv_xy(path)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)

    if standardize:
        mu, sd = standardize_fit(Xtr)
        Xtr = standardize_apply(Xtr, mu, sd)
        Xte = standardize_apply(Xte, mu, sd)

    out: List[Dict[str, Any]] = []

    # Linear Regression
    lr = LinearRegressionGD(lr=0.1, max_iters=2000, tol=1e-6)
    lr.fit(Xtr, ytr)
    yp = lr.predict(Xte)
    acc = accuracy(yte, yp)
    tpp = avg_predict_time_us(lr.predict, Xte, loops=100)
    out.append({
        "timestamp": now_iso(),
        "dataset": dataset_name,
        "model": "linear_regression",
        "accuracy": acc,
        "time_to_convergence_sec": lr.train_time_sec_,
        "time_per_prediction_us": tpp,
        "n_iters": lr.n_iters_,
        "converged": lr.converged_,
        "final_loss": lr.final_loss_,
    })

    # Perceptron
    p = Perceptron(max_iters=2000, shuffle=True, seed=0)
    p.fit(Xtr, ytr)
    yp = p.predict(Xte)
    acc = accuracy(yte, yp)
    tpp = avg_predict_time_us(p.predict, Xte, loops=100)
    out.append({
        "timestamp": now_iso(),
        "dataset": dataset_name,
        "model": "perceptron",
        "accuracy": acc,
        "time_to_convergence_sec": p.train_time_sec_,
        "time_per_prediction_us": tpp,
        "n_iters": p.n_iters_,
        "converged": p.converged_,
        "mistakes_last_epoch": p.mistakes_last_epoch_,
    })

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-a", required=False,
                    default="datasets/binary_classification.csv")
    ap.add_argument("--data-b", required=False,
                    default="datasets/binary_classification_non_lin.csv")
    ap.add_argument("--standardize", default="true")      # "true"/"false"
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    std = args.standardize.lower() in ("1", "true", "yes", "y")

    rows: List[Dict[str, Any]] = []
    rows += run_one("binary_classification", args.data_a, std)
    rows += run_one("binary_classification_non_lin", args.data_b, std)

    # Save metrics
    json_obj = {"runs": rows}
    json_path = os.path.join(args.outdir, "metrics.json")
    save_json(json_obj, json_path)

    header = ["timestamp", "dataset", "model", "accuracy", "time_to_convergence_sec",
              "time_per_prediction_us", "n_iters", "converged", "final_loss", "mistakes_last_epoch"]
    csv_rows = []
    for r in rows:
        csv_rows.append([
            r.get("timestamp"), r.get("dataset"), r.get(
                "model"), r.get("accuracy"),
            r.get("time_to_convergence_sec"), r.get("time_per_prediction_us"),
            r.get("n_iters"), r.get("converged"), r.get(
                "final_loss"), r.get("mistakes_last_epoch"),
        ])
    csv_path = os.path.join(args.outdir, "metrics.csv")
    save_csv(csv_rows, header, csv_path)

    # Console summary
    print("\n=== Results ===")
    for r in rows:
        print(f"{r['dataset']:>30} | {r['model']:>18} | acc={r['accuracy']:.3f} | "
              f"train={r['time_to_convergence_sec']:.4f}s | pred={r['time_per_prediction_us']:.2f}us "
              f"| iters={r['n_iters']} | conv={r['converged']}")

    print(f"\nSaved: {json_path}\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
