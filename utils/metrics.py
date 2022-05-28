'''
Compute Metrics from confusion matrix.
Metrics: precision, recall, f1, accuracy
'''
import numpy as np
import warnings
from matplotlib import pyplot as plt

def conf2metrics(conf_mat):
    """ Confusion matrix to performance metrics conversion """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
    precision[np.isnan(precision)] = 0.0

    recall = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
    recall[np.isnan(recall)] = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1 = 2 * (precision * recall) / (precision + recall)
    f1[np.isnan(f1)] = 0.0

    accuracy = np.trace(conf_mat) / np.sum(conf_mat)
    return precision, recall, f1, accuracy


def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    '''
    Visualize confusion matrix
    --------------------------
    cm : confusion matrix
    labels: ground truth of predicted labels
    title: the title of the figure 
    '''
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
