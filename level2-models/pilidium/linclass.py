class LinearClassifier:

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.n_iters):
            
            # Predicting using current weights and bias
            y_pred = []
            for i in range(n_samples):
                linear_model = self.bias
                for j in range(n_features):
                    linear_model += X[i][j] * self.weights[j]
                y_pred.append(linear_model)

            # Initialising variables for df/dw and df/db
            # f = (1/2*n)*(y_pred-y)^2
            dw = [0.0] * n_features
            db = 0.0

            # Computing df/dw and df/db
            for i in range(n_samples):
                error = y_pred[i] - y[i]
                db += error
                for j in range(n_features):
                    dw[j] += X[i][j] * error
            
            db /= n_samples
            for j in range(n_features):
                dw[j] /= n_samples
                # Updating weights using GD
                self.weights[j] -= self.lr * dw[j]
            
            # Updating bias using GD
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred_continuous = []
        for i in range(len(X)):
            linear_model = self.bias
            for j in range(len(X[0])):
                linear_model += X[i][j] * self.weights[j]
            y_pred_continuous.append(linear_model)
        y_predicted_binary = [1.0 if i > 0.5 else 0.0 for i in y_pred_continuous]
        return y_predicted_binary