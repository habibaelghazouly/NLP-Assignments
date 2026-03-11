import numpy as np

def precision_recall_f1(cm: np.ndarray) -> dict:
    """
    Compute per-class and macro-averaged Precision, Recall, and F1
    directly from a confusion matrix.

    Parameters
    ----------
    cm : ndarray of shape (n_classes, n_classes)

    Returns
    -------
    dict with keys:
        precision_per_class, recall_per_class, f1_per_class  (1-D arrays)
        precision_macro, recall_macro, f1_macro               (scalars)
    """
    n_classes = cm.shape[0]

    precision_per = np.zeros(n_classes)
    recall_per = np.zeros(n_classes)
    f1_per = np.zeros(n_classes)

    for i in range(n_classes):
        tp = cm[i, i]  # true positives  for class i
        fp = cm[:, i].sum() - tp  # false positives for class i
        fn = cm[i, :].sum() - tp  # false negatives for class i

        precision_per[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_per[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        denom = precision_per[i] + recall_per[i]
        f1_per[i] = (2 * precision_per[i] * recall_per[i] / denom) if denom > 0 else 0.0

    return {
        "precision_per_class": precision_per,
        "recall_per_class": recall_per,
        "f1_per_class": f1_per,
        "precision_macro": precision_per.mean(),
        "recall_macro": recall_per.mean(),
        "f1_macro": f1_per.mean(),
    }

def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a confusion matrix from ground-truth and predicted labels.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)

    Returns
    -------
    cm     : ndarray of shape (n_classes, n_classes)
             cm[i, j] = number of samples with true class i predicted as class j
    classes: ndarray of unique sorted class labels
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    
    # map label → index
    label_to_idx = {label: idx for idx, label in enumerate(classes)}

    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1

    return cm, classes



