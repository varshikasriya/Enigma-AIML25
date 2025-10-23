import math
import random

class Perceptron:

    def __init__(self, lr=0.01, n_iters=1000, hidden=4):
        self.lr = lr
        self.n_iters = n_iters
        self.hidden_n = hidden
        
        self.weights_ih = None
        self.bias_h = None
        self.weights_ho = None
        self.bias_o = None

    def _sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def _sigmoid_derivative(self, sig_output):
        return sig_output * (1.0 - sig_output)

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        
        self.weights_ih = [[random.uniform(-0.5, 0.5) for _ in range(self.hidden_n)] for _ in range(n_features)]
        self.bias_h = [random.uniform(-0.5, 0.5) for _ in range(self.hidden_n)]
        self.weights_ho = [[random.uniform(-0.5, 0.5)] for _ in range(self.hidden_n)]
        self.bias_o = [random.uniform(-0.5, 0.5)]

        for _ in range(self.n_iters):
            for idx in range(n_samples):
                x_i = X[idx]
                y_true = y[idx]

                hidden_inputs = [0.0] * self.hidden_n
                for j in range(self.hidden_n):
                    net_input = self.bias_h[j]
                    for k in range(n_features):
                        net_input += x_i[k] * self.weights_ih[k][j]
                    hidden_inputs[j] = net_input
                
                hidden_outputs = [self._sigmoid(inp) for inp in hidden_inputs]

                output_input = self.bias_o[0]
                for j in range(self.hidden_n):
                    output_input += hidden_outputs[j] * self.weights_ho[j][0]
                
                final_output = self._sigmoid(output_input)

                output_error = y_true - final_output
                output_delta = output_error * self._sigmoid_derivative(final_output)

                hidden_errors = [0.0] * self.hidden_n
                for j in range(self.hidden_n):
                    hidden_errors[j] = output_delta * self.weights_ho[j][0]
                
                hidden_deltas = [0.0] * self.hidden_n
                for j in range(self.hidden_n):
                    hidden_deltas[j] = hidden_errors[j] * self._sigmoid_derivative(hidden_outputs[j])

                for j in range(self.hidden_n):
                    self.weights_ho[j][0] += self.lr * hidden_outputs[j] * output_delta
                self.bias_o[0] += self.lr * output_delta

                for j in range(self.hidden_n):
                    for k in range(n_features):
                        self.weights_ih[k][j] += self.lr * x_i[k] * hidden_deltas[j]
                    self.bias_h[j] += self.lr * hidden_deltas[j]

    def predict(self, X):
        predictions = []
        n_features = len(X[0])

        for i in range(len(X)):
            x_i = X[i]
            
            hidden_inputs = [0.0] * self.hidden_n
            for j in range(self.hidden_n):
                net_input = self.bias_h[j]
                for k in range(n_features):
                    net_input += x_i[k] * self.weights_ih[k][j]
                hidden_inputs[j] = net_input
            
            hidden_outputs = [self._sigmoid(inp) for inp in hidden_inputs]

            output_input = self.bias_o[0]
            for j in range(self.hidden_n):
                output_input += hidden_outputs[j] * self.weights_ho[j][0]
            
            final_output = self._sigmoid(output_input)
            
            prediction = 1.0 if final_output > 0.5 else 0.0
            predictions.append(prediction)
            
        return predictions