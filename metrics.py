import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, accuracy_score, fowlkes_mallows_score, normalized_mutual_info_score, adjusted_mutual_info_score
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def nmi_score(y, y_pred):
    return normalized_mutual_info_score(y, y_pred)


def ami_score(y, y_pred):
    return adjusted_mutual_info_score(y, y_pred)


def ari_score(y, y_pred):
    return adjusted_rand_score(y, y_pred)


def fms_score(y, y_pred):
    return fowlkes_mallows_score(y, y_pred)


# def cluster_acc(y_true, y_pred):
#     """
#     Calculate unsupervised clustering accuracy. Requires scikit-learn installed
#
#     # Arguments
#         y_true: true labels, numpy.array with shape `(n_samples,)`
#         y_pred: predicted labels, numpy.array with shape `(n_samples,)`
#
#     # Return
#         accuracy, in [0,1]
#     """
#     y_true = y_true.astype(np.int64)
#     assert y_pred.size == y_true.size
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#     row_ind, col_ind = linear_sum_assignment(w.max() - w)
#     return w[row_ind, col_ind].sum() * 1.0 / y_pred.size

def cluster_acc(y_true, y_pred, display_confusion_matrix=True):
    """
    Calculate unsupervised clustering accuracy, accuracy per class, and display confusion matrix.

    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        display_confusion_matrix: bool, whether to display the confusion matrix

    # Returns
        overall_accuracy: accuracy, in [0, 1]
        class_accuracies: dictionary with accuracy per class
        confusion_matrix: confusion matrix
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    # Create a new y_pred aligned with y_true
    y_pred_aligned = np.zeros_like(y_pred)
    for i in range(len(row_ind)):
        y_pred_aligned[y_pred == row_ind[i]] = col_ind[i]

    overall_accuracy = np.mean(y_pred_aligned == y_true)

    # Calculate accuracy per class
    class_accuracies = {}
    for i in range(D):
        if np.sum(y_true == i) > 0:
            class_accuracies[i] = np.mean(y_pred_aligned[y_true == i] == i)
        else:
            class_accuracies[i] = 0.0

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred_aligned)

    if display_confusion_matrix:
        # Plot the confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(D), yticklabels=range(D))
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    return overall_accuracy, class_accuracies, cm


def cluster_purity(y_true, y_pred):
    """
    Calculate clustering purity
    https://en.wikipedia.org/wiki/Cluster_analysis#Evaluation_and_assessment

    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        purity, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    label_mapping = w.argmax(axis=1)
    y_pred_voted = y_pred.copy()
    for i in range(y_pred.size):
        y_pred_voted[i] = label_mapping[y_pred[i]]
    return accuracy_score(y_pred_voted, y_true)
