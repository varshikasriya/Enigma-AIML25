import json
from linear_regression import LinearRegressionGD
from perceptron import Perceptron
from utils import load_dataset, accuracy_score

datasets = {
    "dataset_A": r"datasets/binary_classification_non_lin.csv",
    "dataset_B": r"datasets/binary_classification.csv"
}

results = {
    "linear_regression": {},
    "perceptron": {}
}

for name, path in datasets.items():
    X, y = load_dataset(path)

    # Linear Regression
    lr_model = LinearRegressionGD()
    lr_model.fit(X, y)
    lr_preds = lr_model.predict(X)

    results["linear_regression"][name] = {
        "accuracy": accuracy_score(y, lr_preds),
        "convergence_time": lr_model.train_time,
        "time_per_prediction": lr_model.pred_time
    }

    # Perceptron
    p_model = Perceptron()
    p_model.fit(X, y)
    p_preds = p_model.predict(X)

    results["perceptron"][name] = {
        "accuracy": accuracy_score(y, p_preds),
        "convergence_time": p_model.train_time,
        "time_per_prediction": p_model.pred_time
    }

# Save results
with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

print("✅ Experiments complete. Results saved to results.json.")
