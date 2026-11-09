"""Classification metrics without an additional scikit-learn dependency."""

import numpy as np


def classification_metrics(y_true, y_pred, num_classes: int = 6) -> dict:
    """Use a fixed class set; rows of the confusion matrix are true labels.

    A class with no true or predicted observations receives F1 = 0, and all
    ``num_classes`` classes participate in the macro average.
    """
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    if truth.ndim != 1 or prediction.ndim != 1 or truth.shape != prediction.shape:
        raise ValueError("Labels must be one-dimensional arrays of equal length")
    if len(truth) == 0 or num_classes < 1:
        raise ValueError("Metrics require at least one observation and one class")
    for values in (truth, prediction):
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("Class labels must be integers")
        if np.any(values < 0) or np.any(values >= num_classes):
            raise ValueError("Class labels are outside the configured class range")
    matrix = np.bincount(
        truth.astype(np.int64) * num_classes + prediction.astype(np.int64),
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    true_positive = matrix.diagonal().astype(np.float64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = np.divide(true_positive, predicted, out=np.zeros(num_classes), where=predicted != 0)
    recall = np.divide(true_positive, support, out=np.zeros(num_classes), where=support != 0)
    denominator = support + predicted
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros(num_classes), where=denominator != 0)
    return {
        "accuracy": float(true_positive.sum() / len(truth)),
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": matrix.tolist(),
        "n_samples": int(len(truth)),
    }
