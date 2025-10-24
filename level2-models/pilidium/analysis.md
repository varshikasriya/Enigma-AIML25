# ML Model Analysis

## 1. Metrics

```json
{
    "linear_classifier_on_linearly_separable": {
        "accuracy": 0.8716666666666667,
        "time_to_convergence_sec": 0.09880805504508317,
        "time_per_prediction_sec": 1.09383351324747e-07
    },
    "perceptron_mlp_on_linearly_separable": {
        "accuracy": 0.905,
        "time_to_convergence_sec": 0.9932582280016504,
        "time_per_prediction_sec": 7.529867192109426e-07
    },
    "linear_classifier_on_non_linearly_separable": {
        "accuracy": 0.79125,
        "time_to_convergence_sec": 0.13327804603613913,
        "time_per_prediction_sec": 1.0153875336982309e-07
    },
    "perceptron_mlp_on_non_linearly_separable": {
        "accuracy": 0.86625,
        "time_to_convergence_sec": 1.3250223670038395,
        "time_per_prediction_sec": 7.566537533421069e-07
    }
}
```

-----

## 2\. Analysis

This analysis compares the performance of two models: **Linear Regression** (used as a simple linear classifier) and a **Perceptron with one hidden layer** (a Multi-Layer Perceptron, or MLP).

### Performance: Speed

In terms of speed, the MLP took approximately 10 times as much time as the Linear Regression model for convergence and 6-7 times as much time for prediction.

This is an expected trade-off. The MLP has a hidden layer, which means it must perform significantly more computations for each training iteration (both in the forward pass and during backpropagation). The Linear Regression model, by contrast, has a much simpler gradient calculation, making it far faster.

### Performance: Accuracy

In terms of accuracy, the MLP is the clear winner. This is because it has a greater number of trainable parameters compared to the Linear Regression model. This higher "model capacity" means the MLP can learn more complex patterns and non-linear relationships from the data, while the Linear Regression model is fundamentally restricted to learning a single straight line.

### Impact of Dataset Type

The nature of the dataset was the most important factor in model performance.

  * **Linear Regression (Linear Classifier):** This model performed well on the linearly separable dataset but relatively poorly on its non-linear counterpart. This is because the model's entire design is to find the best *straight line* to divide the data. This task is impossible for the non-linearly separable dataset.

  * **Perceptron (MLP):** The MLP performed exceptionally well on *both* datasets. Its accuracy did likely see a minor reduction on the non-linear data, as it is an inherently more difficult problem to solve. However, because the hidden layer and sigmoid activation functions allow the MLP to create complex, *curved* decision boundaries, it was able to successfully classify the non-linear data where the linear model could not.

-----

## 3\. Conclusion

Overall, the Perceptron (MLP) performs better than the Linear Regression classifier on both datasets. The experiment demonstrates that while linear models are fast, they are limited to simple, linearly separable problems. The MLP, while computationally more expensive, is a more powerful and flexible model that can successfully solve both linear and non-linear classification problems.