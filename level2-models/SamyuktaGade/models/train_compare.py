# level2-models/SamyuktaGade/train_compare.py
from __future__ import annotations
import argparse, json, time, os, csv
import numpy as np
from pathlib import Path

from linear_regression import LinearRegressionGD
from perceptron import Perceptron

def generate_binary_dataset(path: str, n: int = 1000, seed: int = 42, 
                            noise_level: float = 1.0, overlap: float = 0.3):
    """Generate a realistic 2D binary classification dataset with noise.
    
    Args:
        path: Output CSV path
        n: Number of samples
        seed: Random seed
        noise_level: Standard deviation of Gaussian noise
        overlap: Amount of class overlap (0=no overlap, 1=complete overlap)
    """
    print(f"[setup] Generating binary dataset at {path}")
    print(f"  Samples: {n}, Noise: {noise_level}, Overlap: {overlap}")
    
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0
    
    # Create two Gaussian blobs with controlled overlap
    # Separation distance based on overlap parameter
    separation = 3.0 * (1 - overlap)
    
    X0 = rng.normal(loc=[-separation/2, -separation/2], 
                    scale=[noise_level, noise_level], size=(n0, 2))
    X1 = rng.normal(loc=[separation/2, separation/2], 
                    scale=[noise_level, noise_level], size=(n1, 2))
    
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    
    # Add feature correlation for complexity
    X[:, 1] += 0.5 * X[:, 0] + rng.normal(0, 0.3, n)
    
    # Shuffle
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]
    
    # Save to CSV
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature1", "feature2", "label"])
        for (x1, x2), yi in zip(X, y):
            writer.writerow([f"{x1:.6f}", f"{x2:.6f}", int(yi)])
    
    print(f"[setup] Dataset saved ({len(y)} samples)")
    return X, y

def load_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset from CSV."""
    print(f"[data] Loading dataset from {path}")
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if len(row)]
    
    # Detect and skip header
    try:
        _ = [float(x) for x in rows[0]]
        start = 0
    except (ValueError, IndexError):
        start = 1
    
    data = np.array([[float(x) for x in row] for row in rows[start:]], dtype=float)
    X, y = data[:, :-1], data[:, -1].astype(int)
    
    print(f"[data] Loaded {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[data] Class balance: {(y==0).sum()} (class 0), {(y==1).sum()} (class 1)")
    
    return X, y

def train_val_test_split(X, y, val_size=0.15, test_size=0.2, seed=42):
    """Split data into train, validation, and test sets."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    
    test_n = int(n * test_size)
    val_n = int(n * val_size)
    train_n = n - test_n - val_n
    
    train_idx = idx[:train_n]
    val_idx = idx[train_n:train_n+val_n]
    test_idx = idx[train_n+val_n:]
    
    return (X[train_idx], X[val_idx], X[test_idx], 
            y[train_idx], y[val_idx], y[test_idx])

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute classification metrics."""
    accuracy = float((y_true == y_pred).mean())
    
    # Precision, Recall, F1 for class 1
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {
            "TP": int(tp), "FP": int(fp),
            "FN": int(fn), "TN": int(tn)
        }
    }

def avg_prediction_time(model_predict, X: np.ndarray, repeats: int = 100) -> float:
    """Compute average per-sample prediction time."""
    # Warmup
    _ = model_predict(X)
    
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = model_predict(X)
    t1 = time.perf_counter()
    
    total_predictions = X.shape[0] * repeats
    return (t1 - t0) / total_predictions

def main():
    parser = argparse.ArgumentParser(
        description="Train and compare Linear Regression vs Perceptron",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Update default paths 
    parser.add_argument("--data", default="../../datasets/binary_classification.csv", 
                       help="Path to binary classification dataset")
    parser.add_argument("--outdir", default="../../results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--generate-size", type=int, default=1000,
                       help="Size of generated dataset if file missing")
    parser.add_argument("--noise", type=float, default=1.2,
                       help="Noise level in generated data (std dev)")
    parser.add_argument("--overlap", type=float, default=0.4,
                       help="Class overlap in generated data (0-1)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print training progress")
    args = parser.parse_args()

    # Generate or load dataset
    if not os.path.exists(args.data):
        print(f"[setup] Dataset not found, generating...")
        generate_binary_dataset(
            args.data, n=args.generate_size, seed=args.seed,
            noise_level=args.noise, overlap=args.overlap
        )
    
    X, y = load_csv(args.data)
    
    # Split data: train / val / test
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, val_size=0.15, test_size=0.2, seed=args.seed
    )
    
    print(f"\n[split] Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    # ============= LINEAR REGRESSION =============
    print("\n" + "="*60)
    print("[Linear Regression] Training...")
    print("="*60)
    
    lr_model = LinearRegressionGD(
        lr=0.01,
        max_epochs=5000,
        tol=1e-6,
        l2=0.05,  # Regularization to prevent overfitting
        fit_intercept=True,
        standardize=True,
        random_state=args.seed,
        verbose=args.verbose
    )
    
    
    t0 = time.perf_counter()
    lr_model.fit(X_train, y_train.astype(float), X_val, y_val.astype(float))
    lr_train_time = time.perf_counter() - t0
    
    # Evaluate on all splits
    lr_train_pred = lr_model.predict_classes(X_train, threshold=0.5)
    lr_val_pred = lr_model.predict_classes(X_val, threshold=0.5)
    lr_test_pred = lr_model.predict_classes(X_test, threshold=0.5)
    
    lr_train_metrics = compute_metrics(y_train, lr_train_pred)
    lr_val_metrics = compute_metrics(y_val, lr_val_pred)
    lr_test_metrics = compute_metrics(y_test, lr_test_pred)
    
    lr_pred_time = avg_prediction_time(
        lambda X: lr_model.predict_classes(X, 0.5), X_test, repeats=100
    )
    
    print(f"  Train Acc: {lr_train_metrics['accuracy']:.4f}")
    print(f"  Val Acc:   {lr_val_metrics['accuracy']:.4f}")
    print(f"  Test Acc:  {lr_test_metrics['accuracy']:.4f}")
    print(f"  Epochs: {lr_model.epochs_run_}, Converged: {lr_model.converged_}")
    print(f"  Training time: {lr_train_time:.4f}s")

    # ============= PERCEPTRON =============
    print("\n" + "="*60)
    print("[Perceptron] Training...")
    print("="*60)
    
    perceptron = Perceptron(
    lr=0.1,
    max_epochs=1000,
    fit_intercept=True,
    shuffle=True,
    random_state=args.seed,
    verbose=args.verbose
    )
 
    
    t0 = time.perf_counter()
    perceptron.fit(X_train, y_train)
    perceptron_train_time = time.perf_counter() - t0
    
    # Evaluate on all splits
    perceptron_train_pred = perceptron.predict(X_train)
    perceptron_val_pred = perceptron.predict(X_val)
    perceptron_test_pred = perceptron.predict(X_test)
    
    perceptron_train_metrics = compute_metrics(y_train, perceptron_train_pred)
    perceptron_val_metrics = compute_metrics(y_val, perceptron_val_pred)
    perceptron_test_metrics = compute_metrics(y_test, perceptron_test_pred)
    
    perceptron_pred_time = avg_prediction_time(
        perceptron.predict, X_test, repeats=100
    )
    
    print(f"  Train Acc: {perceptron_train_metrics['accuracy']:.4f}")
    print(f"  Val Acc:   {perceptron_val_metrics['accuracy']:.4f}")
    print(f"  Test Acc:  {perceptron_test_metrics['accuracy']:.4f}")
    print(f"  Epochs: {perceptron.epochs_run_}, Converged: {perceptron.converged_}")
    print(f"  Training time: {perceptron_train_time:.4f}s")

    # ============= SAVE RESULTS =============
    results = {
        "dataset_path": os.path.abspath(args.data),
        "dataset_shape": [int(X.shape[0]), int(X.shape[1])],
        "splits": {
            "train_size": int(len(y_train)),
            "val_size": int(len(y_val)),
            "test_size": int(len(y_test))
        },
        "linear_regression": {
            "hyperparameters": {
                "learning_rate": lr_model.lr,
                "l2_regularization": lr_model.l2,
                "max_epochs": lr_model.max_epochs
            },
            "training": {
                "epochs_run": lr_model.epochs_run_,
                "converged": lr_model.converged_,
                "final_loss": lr_model.last_loss_,
                "time_sec": lr_train_time
            },
            "metrics": {
                "train": lr_train_metrics,
                "validation": lr_val_metrics,
                "test": lr_test_metrics
            },
            "time_per_prediction_sec": lr_pred_time
        },
        "perceptron": {
            "hyperparameters": {
                "learning_rate": perceptron.lr,
                "max_epochs": perceptron.max_epochs,
                "early_stopping": perceptron.early_stopping
            },
            "training": {
                "epochs_run": perceptron.epochs_run_,
                "converged": perceptron.converged_,
                "time_sec": perceptron_train_time
            },
            "metrics": {
                "train": perceptron_train_metrics,
                "validation": perceptron_val_metrics,
                "test": perceptron_test_metrics
            },
            "time_per_prediction_sec": perceptron_pred_time
        }
    }

    # Save JSON results
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    results_path = os.path.join(args.outdir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Generate analysis report
    analysis_path = os.path.join(args.outdir, "analysis.md")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("# Level 2 Analysis: Linear Regression vs Perceptron\n\n")
        f.write(f"**Dataset:** `{results['dataset_path']}`  \n")
        f.write(f"**Shape:** {results['dataset_shape'][0]} samples × {results['dataset_shape'][1]} features  \n")
        f.write(f"**Split:** {results['splits']['train_size']} train / ")
        f.write(f"{results['splits']['val_size']} val / {results['splits']['test_size']} test  \n\n")
        
        f.write("## Results Summary\n\n")
        f.write("### Linear Regression\n")
        f.write(f"- **Train Accuracy:** {lr_train_metrics['accuracy']:.4f}\n")
        f.write(f"- **Validation Accuracy:** {lr_val_metrics['accuracy']:.4f}\n")
        f.write(f"- **Test Accuracy:** {lr_test_metrics['accuracy']:.4f}\n")
        f.write(f"- **Test F1 Score:** {lr_test_metrics['f1']:.4f}\n")
        f.write(f"- **Epochs:** {lr_model.epochs_run_} (converged: {lr_model.converged_})\n")
        f.write(f"- **Training Time:** {lr_train_time:.4f}s\n\n")
        
        f.write("### Perceptron\n")
        f.write(f"- **Train Accuracy:** {perceptron_train_metrics['accuracy']:.4f}\n")
        f.write(f"- **Validation Accuracy:** {perceptron_val_metrics['accuracy']:.4f}\n")
        f.write(f"- **Test Accuracy:** {perceptron_test_metrics['accuracy']:.4f}\n")
        f.write(f"- **Test F1 Score:** {perceptron_test_metrics['f1']:.4f}\n")
        f.write(f"- **Epochs:** {perceptron.epochs_run_} (converged: {perceptron.converged_})\n")
        f.write(f"- **Training Time:** {perceptron_train_time:.4f}s\n\n")
        
        f.write("## Detailed Analysis\n\n")
        
        # Overfitting analysis
        f.write("### Overfitting Check\n")
        lr_overfit = lr_train_metrics['accuracy'] - lr_test_metrics['accuracy']
        p_overfit = perceptron_train_metrics['accuracy'] - perceptron_test_metrics['accuracy']
        
        f.write(f"- **Linear Regression:** Train-Test gap = {lr_overfit:.4f}\n")
        f.write(f"- **Perceptron:** Train-Test gap = {p_overfit:.4f}\n")
        
        if lr_overfit > 0.1:
            f.write("- ⚠️ Linear Regression shows significant overfitting. Consider increasing L2 regularization.\n")
        if p_overfit > 0.1:
            f.write("- ⚠️ Perceptron shows significant overfitting. This is expected as perceptron has no regularization.\n")
        f.write("\n")
        
        # Convergence analysis
        f.write("### Convergence Behavior\n")
        f.write(f"- **Linear Regression:** ")
        if lr_model.converged_:
            f.write(f"Converged after {lr_model.epochs_run_} epochs using gradient descent with loss={lr_model.last_loss_:.6f}\n")
        else:
            f.write(f"Did not converge after {lr_model.epochs_run_} epochs (may need more epochs or learning rate tuning)\n")
        
        f.write(f"- **Perceptron:** ")
        if perceptron.converged_:
            f.write(f"Converged after {perceptron.epochs_run_} epochs (0 training mistakes)\n")
        else:
            final_mistakes = perceptron.mistake_history_[-1] if perceptron.mistake_history_ else "N/A"
            f.write(f"Did not converge after {perceptron.epochs_run_} epochs (final mistakes: {final_mistakes})\n")
        f.write("\n")
        
        # Performance comparison
        f.write("### Model Comparison\n\n")
        f.write("**Accuracy:**\n")
        if lr_test_metrics['accuracy'] > perceptron_test_metrics['accuracy']:
            diff = lr_test_metrics['accuracy'] - perceptron_test_metrics['accuracy']
            f.write(f"- Linear Regression performs better by {diff:.4f} on test set\n")
        elif perceptron_test_metrics['accuracy'] > lr_test_metrics['accuracy']:
            diff = perceptron_test_metrics['accuracy'] - lr_test_metrics['accuracy']
            f.write(f"- Perceptron performs better by {diff:.4f} on test set\n")
        else:
            f.write("- Both models achieve similar test accuracy\n")
        
        f.write("\n**Speed:**\n")
        f.write(f"- Linear Regression training: {lr_train_time:.4f}s\n")
        f.write(f"- Perceptron training: {perceptron_train_time:.4f}s\n")
        
        speed_ratio = perceptron_train_time / lr_train_time
        if speed_ratio > 1.5:
            f.write(f"- Linear Regression trains {speed_ratio:.1f}x faster\n")
        elif speed_ratio < 0.67:
            f.write(f"- Perceptron trains {1/speed_ratio:.1f}x faster\n")
        else:
            f.write("- Training times are comparable\n")
        
        f.write("\n### Key Insights\n\n")
        f.write("**Linear Regression:**\n")
        f.write("- Uses squared loss and gradient descent for optimization\n")
        f.write("- Feature standardization enables stable training\n")
        f.write("- L2 regularization helps prevent overfitting\n")
        f.write("- Produces continuous outputs (probability-like scores)\n")
        f.write("- More robust to noisy/overlapping data\n\n")
        
        f.write("**Perceptron:**\n")
        f.write("- Uses online learning with threshold-based updates\n")
        f.write("- Only updates on misclassified examples\n")
        f.write("- No built-in regularization (prone to overfitting)\n")
        f.write("- Produces binary outputs directly\n")
        f.write("- Guaranteed to converge only if data is linearly separable\n")
        f.write("- Sensitive to class imbalance and outliers\n\n")
        
        f.write("### Recommendations\n\n")
        f.write("Based on the results:\n")
        
        if not perceptron.converged_ and lr_model.converged_:
            f.write("- The data is likely **not perfectly linearly separable**, explaining why perceptron struggles\n")
            f.write("- Linear Regression's probabilistic approach handles noisy boundaries better\n")
        
        if abs(lr_test_metrics['accuracy'] - perceptron_test_metrics['accuracy']) < 0.05:
            f.write("- Both models perform similarly, suggesting the decision boundary is approximately linear\n")
        
        f.write("- For production use, consider Linear Regression due to better generalization\n")
        f.write("- To improve Perceptron, consider using averaged perceptron or kernel methods\n")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Linear Regression - Test Acc: {lr_test_metrics['accuracy']:.4f}, F1: {lr_test_metrics['f1']:.4f}")
    print(f"Perceptron        - Test Acc: {perceptron_test_metrics['accuracy']:.4f}, F1: {perceptron_test_metrics['f1']:.4f}")
    print(f"\nFiles saved:")
    print(f"  - {results_path}")
    print(f"  - {analysis_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()