import numpy as np
import pandas as pd

def load_dataset(path):
    data = pd.read_csv(path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)
