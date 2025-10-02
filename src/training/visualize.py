import numpy as np
import matplotlib.pyplot as plt

def plot_training_history(
    history,
    metrics=["loss", "accuracy"]
) -> None:
    """
    Plots training and validation metrics from a Keras history object.

    Args:
        history: Keras History object returned by model.fit().
        metrics: List of metrics to plot. Default is ["loss", "accuracy"].
    """
    for metric in metrics:
        plt.figure()
        plt.plot(history.history[metric], label=f"Train {metric}")
        val_metric = f"val_{metric}"
        if val_metric in history.history:
            plt.plot(history.history[val_metric], label=f"Val {metric}")
        plt.title(f"Model {metric.capitalize()}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_confusion_matrix(
    cm,
    class_names,
    normalize=False,
    title="Confusion Matrix",
    cmap=plt.cm.Blues,
) -> None:
    """
    Plots a confusion matrix.

    Args:
        cm: Confusion matrix (2D array).
        class_names: List of class names.
        normalize: Whether to normalize the values.
        title: Title for the plot.
        cmap: Colormap.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")

def show_metric_curves(
    history,
    metrics=("loss", "accuracy")
) -> None:
    """
    Display training and validation metric curves from a training history.

    Args:
        history: Object with .history dict (e.g., Keras History).
        metrics: Tuple of metric names to plot.
    """
    for metric in metrics:
        plt.figure(figsize=(7, 4))
        train = history.history.get(metric)
        val = history.history.get(f"val_{metric}")
        if train is not None:
            plt.plot(train, label=f"Train {metric}")
        if val is not None:
            plt.plot(val, label=f"Validation {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.title(f"{metric.title()} Progression")
        plt.legend()
        plt.tight_layout()
        plt.show()

def draw_confusion(
    cm,
    labels,
    normalize=False,
    title="Confusion",
    cmap="Purples",
) -> None:
    """
    Visualize a confusion matrix.

    Args:
        cm: 2D array-like confusion matrix.
        labels: List of class labels.
        normalize: If True, display proportions.
        title: Plot title.
        cmap: Matplotlib colormap.
    """
    matrix = np.array(cm)
    if normalize:
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(len(labels)), labels, rotation=30)
    plt.yticks(np.arange(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = f"{matrix[i, j]:.2f}" if normalize else f"{int(matrix[i, j])}"
            plt.text(j, i, value, ha="center", va="center",
                     color="white" if matrix[i, j] > matrix.max()/2 else "black")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()
