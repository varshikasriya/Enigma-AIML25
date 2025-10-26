import numpy as np
import json
import time
from linear_regression import LinearRegression
from perceptron import Perceptron


    
def load_dataset(filepath):
   
    try:
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        X = data[:, :-1]  
        y = data[:, -1]   
        return X, y
    except:
        
        data = np.loadtxt(filepath, skiprows=1)
        X = data[:, :-1]
        y = data[:, -1]
        return X, y


def calculate_accuracy(y_true, y_pred):
    
    unique_true = np.unique(y_true)
    
    if set(unique_true).issubset({0, 1}):
        
        y_pred_binary = (y_pred >= 0.5).astype(int)
        correct = np.sum(y_true == y_pred_binary)
    else:

        threshold = 0.1 * np.std(y_true)
        correct = np.sum(np.abs(y_true - y_pred) < threshold)
    
    accuracy = (correct / len(y_true)) * 100
    return accuracy

def mean_squared_error(y_true, y_pred):
    
    return np.mean((y_true - y_pred) ** 2)

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
 
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name}")
    print(f"{'='*50}")
    
    
    train_start = time.time()
    model.fit(X_train, y_train)
    train_end = time.time()
    training_time = train_end - train_start

    predict_start = time.time()
    predictions = model.predict(X_test)
    predict_end = time.time()
    total_prediction_time = predict_end - predict_start
    
    
    time_per_prediction = total_prediction_time / len(X_test)
    

    accuracy = calculate_accuracy(y_test, predictions)
    
   
    mse = mean_squared_error(y_test, predictions)
    
 
    convergence_iteration = model.convergence_iteration if hasattr(model, 'convergence_iteration') else None
    
 
    print(f"Training Time: {training_time:.6f} seconds")
    print(f"Convergence Iteration: {convergence_iteration}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Time per Prediction: {time_per_prediction:.8f} seconds")
    
    return {
        'model_name': model_name,
        'training_time': training_time,
        'convergence_iteration': convergence_iteration,
        'accuracy': accuracy,
        'mse': mse,
        'total_prediction_time': total_prediction_time,
        'time_per_prediction': time_per_prediction,
        'n_test_samples': len(X_test)
    }

def split_data(X, y, train_ratio=0.8):
  
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    
   
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def normalize_features(X):
   
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
  
    std[std == 0] = 1
    return (X - mean) / std

def main():
   
    np.random.seed(42)
    
 
    dataset_paths = [
    '../datasets/binary_classification.csv',  
    '../datasets/binary_classification_non_lin.csv'   
    ]

    
    all_results = {}
    
    for dataset_idx, dataset_path in enumerate(dataset_paths):
        print(f"\n{'#'*60}")
        print(f"# DATASET {dataset_idx + 1}: {dataset_path}")
        print(f"{'#'*60}")
        
        try:
            
            X, y = load_dataset(dataset_path)
            print(f"Loaded dataset with {len(X)} samples and {X.shape[1]} features")
            
          
            X = normalize_features(X)
            
          
            X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8)
            print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
            
           
            lr_model = LinearRegression(learning_rate=0.01, n_iterations=1000)
            perceptron_model = Perceptron(learning_rate=0.01, n_iterations=1000)
            
            
            lr_results = evaluate_model(
                lr_model, X_train, y_train, X_test, y_test, 
                "Linear Regression"
            )
            
           
            perceptron_results = evaluate_model(
                perceptron_model, X_train, y_train, X_test, y_test, 
                "Perceptron"
            )
            
        
            all_results[f'dataset_{dataset_idx + 1}'] = {
                'dataset_path': dataset_path,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'linear_regression': lr_results,
                'perceptron': perceptron_results
            }
            
        except Exception as e:
            print(f"Error loading or processing {dataset_path}: {e}")
            all_results[f'dataset_{dataset_idx + 1}'] = {
                'error': str(e)
            }
    
   
    output_file = 'metrics.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")
    

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for dataset_name, results in all_results.items():
        if 'error' not in results:
            print(f"\n{dataset_name}:")
            print(f"  Linear Regression Accuracy: {results['linear_regression']['accuracy']:.2f}%")
            print(f"  Perceptron Accuracy: {results['perceptron']['accuracy']:.2f}%")

if __name__ == "__main__":
    main()