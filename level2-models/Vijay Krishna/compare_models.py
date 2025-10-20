import numpy as np
import pandas as pd
import json
import time
from linear_regression_sgd import LinearRegressionSGD # CHANGED
from perceptron import Perceptron


def load_dataset(filepath):
    df = pd.read_csv(filepath)
    X = df[['x1', 'x2']].values
    y = df['label'].values
    return X, y


def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def measure_prediction_time(model, X, n_runs=100):
    """Measure average prediction time by running prediction multiple times."""
    times = []
    # Use a small subset of data for timing to keep the measurement fast and accurate
    X_subset = X[:min(100, len(X))] 
    for _ in range(n_runs):
        start = time.time()
        model.predict(X_subset)
        end = time.time()
        times.append(end - start)
    return np.mean(times) / len(X_subset) # Return time per single prediction


def train_and_evaluate(model, model_name, X, y, dataset_name):
    print(f"\nTraining {model_name} on {dataset_name}")
    
    # Train and measure convergence time
    start_time = time.time()
    model.fit(X, y)
    convergence_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate accuracy
    accuracy = calculate_accuracy(y, y_pred)
    
    # Measure prediction time
    avg_prediction_time = measure_prediction_time(model, X)
    
    # Collect metrics
    metrics = {
        'model': model_name,
        'dataset': dataset_name,
        'accuracy': float(accuracy),
        'time_to_convergence': float(convergence_time),
        'iterations_to_converge': int(model.iterations_to_converge),
        'time_per_prediction_s': float(avg_prediction_time)
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Time to convergence: {convergence_time:.4f} seconds")
    print(f"  Iterations: {model.iterations_to_converge}")
    print(f"  Time per prediction: {avg_prediction_time:.6f} seconds")
    
    return metrics


def main():
    # Define paths assuming you are running this from the contributor's directory
    datasets = {
        'binary_classification_linear': 'C:\\Users\\gvija\\Github\\Enigma-AIML25\\level2-models\\Vijay Krishna\\datasets\\binary_classification.csv',
        'binary_classification_non_lin': 'C:\\Users\\gvija\\Github\\Enigma-AIML25\\level2-models\\Vijay Krishna\\datasets\\binary_classification_non_lin.csv'
    }
    
    all_metrics = []
    
    for dataset_name, dataset_path in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print('='*60)
        
        X, y = load_dataset(dataset_path)
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        
        # Linear Regression (SGD)
        lr_model = LinearRegressionSGD(learning_rate=0.01, max_iterations=1000)
        lr_metrics = train_and_evaluate(lr_model, 'Linear Regression (SGD)', X, y, dataset_name)
        all_metrics.append(lr_metrics)
        
        # Perceptron
        perceptron_model = Perceptron(learning_rate=0.01, max_iterations=1000)
        perceptron_metrics = train_and_evaluate(perceptron_model, 'Perceptron', X, y, dataset_name)
        all_metrics.append(perceptron_metrics)
    
    with open('metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Metrics saved to metrics.json")
    print('='*60)


if __name__ == '__main__':
    main()