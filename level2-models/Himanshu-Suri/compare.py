import numpy as np
import pandas as pd
import time
import json
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression
from perceptron import Perceptron

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def evaluate_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lin_model = LinearRegression(learning_rate=0.1)
    start_train = time.time()
    lin_model.fit(X_train, y_train, iterations=1000)
    end_train = time.time()

    start_pred = time.time()
    y_pred_lin = lin_model.predict(X_test) 
    end_pred = time.time()

    acc_lin = accuracy(y_test, y_pred_lin)
    time_to_converge_lin = end_train - start_train
    time_per_pred_lin = (end_pred - start_pred) / len(y_test)

    perc_model = Perceptron(learning_rate=0.01, n_iters=1000)
    start_train = time.time()
    perc_model.fit(X_train, y_train)
    end_train = time.time()

    start_pred = time.time()
    y_pred_perc = perc_model.predict(X_test)
    end_pred = time.time()

    acc_perc = accuracy(y_test, y_pred_perc)
    time_to_converge_perc = end_train - start_train
    time_per_pred_perc = (end_pred - start_pred) / len(y_test)

    return {
        "LinearRegression": {
            "accuracy": float(acc_lin),
            "time_to_convergence": float(time_to_converge_lin),
            "time_per_prediction": float(time_per_pred_lin)
        },
        "Perceptron": {
            "accuracy": float(acc_perc),
            "time_to_convergence": float(time_to_converge_perc),
            "time_per_prediction": float(time_per_pred_perc)
        }
    }

csv_files = ['binary_classification_non_lin(1).csv', 'binary_classification.csv']
all_results = {}

for file in csv_files:
    print(f"Processing {file}...")
    data = pd.read_csv(file)
    X = data[['x1', 'x2']].values
    y = data['label'].values
    all_results[file] = evaluate_models(X, y)

with open("all_model_comparison_results.json", "w") as f:
    json.dump(all_results, f, indent=4)

print("Results saved to all_model_comparison_results.json")
