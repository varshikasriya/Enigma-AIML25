import csv
import time
import json
import math
from linclass import LinearClassifier
from ptron import Perceptron

def csv_to_X_y(filepath):
    """Returns X, y from CSV file."""
    X = []
    y = []
    with open(filepath, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            X.append(list(map(float, row[:-1])))
            y.append(float(row[-1]))
    return X, y

def get_accuracy(y_true, y_pred):
    """Calculates accuracy percentage."""
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / float(len(y_true))

def get_mean_std(X):
    """Calculate mean and std dev for each feature."""
    n_samples = len(X)
    n_features = len(X[0])
    means = [0.0] * n_features
    stds = [0.0] * n_features

    for j in range(n_features):
        col_sum = 0
        for i in range(n_samples):
            col_sum += X[i][j]
        means[j] = col_sum / n_samples

    for j in range(n_features):
        variance_sum = 0
        for i in range(n_samples):
            variance_sum += (X[i][j] - means[j]) ** 2
        
        std_val = math.sqrt(variance_sum / n_samples)
        stds[j] = 1.0 if std_val == 0 else std_val

    return means, stds

def scale_data(X, means, stds):
    """Apply Z-score scaling to the dataset."""
    scaled_X = []
    for i in range(len(X)):
        scaled_row = [(X[i][j] - means[j]) / stds[j] for j in range(len(X[0]))]
        scaled_X.append(scaled_row)
    return scaled_X

def main():
    
    linear_data_path = 'level2-models/datasets/binary_classification.csv'
    nonlinear_data_path = 'level2-models/datasets/binary_classification_non_lin.csv'

    X1, y1 = csv_to_X_y(linear_data_path)
    X2, y2 = csv_to_X_y(nonlinear_data_path)

    datasets = {
        "linearly_separable": (X1, y1),
        "non_linearly_separable": (X2, y2)
    }
    
    models = {
        "linear_classifier": LinearClassifier,
        "perceptron_mlp": Perceptron
    }

    results = {}
    
    for d_name, (X, y) in datasets.items():
        
        means, stds = get_mean_std(X)
        X_scaled = scale_data(X, means, stds)

        for m_name, ModelClass in models.items():
            
            print(f"  > Running {m_name} on {d_name} data...")
            
            model = ModelClass()
            
            start_fit = time.perf_counter()
            model.fit(X_scaled, y)
            end_fit = time.perf_counter()
            fit_time = end_fit - start_fit
            
            start_pred = time.perf_counter()
            y_pred = model.predict(X_scaled)
            end_pred = time.perf_counter()
            
            total_pred_time = end_pred - start_pred
            time_per_prediction = total_pred_time / len(y)
            
            accuracy = get_accuracy(y, y_pred)
            
            key = f"{m_name}_on_{d_name}"
            results[key] = {
                "accuracy": accuracy,
                "time_to_convergence_sec": fit_time,
                "time_per_prediction_sec": time_per_prediction
            }

    with open('level2-models/pilidium/metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Metrics saved to 'metrics.json'")

if __name__ == "__main__":
    main()