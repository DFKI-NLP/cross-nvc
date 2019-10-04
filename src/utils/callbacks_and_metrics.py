##!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Useful callbacks.

@author lisa.raithel@dfki.de
"""

# import keras
from keras.callbacks import Callback
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, classification_report,
                             precision_recall_fscore_support, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
# from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
# from itertools import cycle

import numpy as np
from inspect import signature

from scipy import interp
import time
import warnings

import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def plot_precision_recall_curve(y_pred, y_true, threshold):
    """..."""
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    # thresholds = dict()
    # n_classes = y_true.shape[1]

    # for i in range(n_classes):
    #     precision[i], recall[i], thresholds[i] = precision_recall_curve(
    #         y_true[:, i], y_pred[:, i])
    #     average_precision[i] = average_precision_score(y_true[:, i],
    #                                                    y_pred[:, i])

    # # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true.ravel(), y_pred.ravel())
    average_precision["micro"] = average_precision_score(
        y_true, y_pred, average="micro")
    # print('Average precision score, micro-averaged over all classes: {0:0.2f}'
    #       .format(average_precision["micro"]))

    # plt.figure()
    # plt.step(
    #     recall['micro'],
    #     precision['micro'],
    #     color='b',
    #     alpha=0.2,
    #     where='post')
    # step_kwargs = ({
    #     'step': 'post'
    # } if 'step' in signature(plt.fill_between).parameters else {})

    # plt.fill_between(
    #     recall["micro"],
    #     precision["micro"],
    #     alpha=0.2,
    #     color='b',
    #     **step_kwargs)

    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title(
    #     'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    #     .format(average_precision["micro"]))
    # plt.show()

    # plot precision-recall curve with iso-F1

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('Micro-average Precision-Recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve (threshold: {:.2f})'.format(threshold))
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    plt.show()


def determine_best_threshold(current_f1, y_pred, y_true):
    """Determine a threshold that returns a better F1 score."""
    y_pred_original = np.copy(y_pred)
    best_threshold = 0.5
    best_predictions = np.zeros(y_pred.shape)
    print("Original threshold: {}, original F1: {}".format(
        best_threshold, current_f1))

    # iterate over thresholds with step size 0.02
    # and determine the threshold that yields the best weighted F1 score
    for threshold in np.arange(0.1, 0.5, 0.02):

        y_pred_original[y_pred_original >= threshold] = 1
        y_pred_original[y_pred_original < threshold] = 0

        f1 = f1_score(y_true, y_pred_original, average="weighted")

        if f1 > current_f1:
            current_f1 = f1
            best_threshold = threshold
            best_predictions = y_pred_original

        y_pred_original = np.copy(y_pred)

    print("\nDetermined best threshold: {} (yielded F1 of {})".format(
        best_threshold, current_f1))

    print("Precision-Recall curve for best threshold (threshold: {:.2f}):".format(best_threshold))
    plot_precision_recall_curve(y_pred=best_predictions, y_true=y_true, threshold=best_threshold)

    return best_threshold


def compute_roc_auc(y_val, predictions, concepts):
    """..."""
    # Compute ROC curve and ROC area for each class
    print("\nROC/AUC for best performing model on dev set (threshold: 0.5):\n")

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()
    n_classes = y_val.shape[1]

    linewidth = 2
    # print("#classes: {}".format(n_classes))

    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_val[:, i],
                                                  predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(
        y_val.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # print(("fpr:\n{}\ntpr:\n{}\nthresholds: {}\n\n").format(
    #     fpr["micro"], tpr["micro"], thresholds["micro"]))
    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label='micro-average ROC curve (area = {0:0.2f})'
        ''.format(roc_auc["micro"]),
        color='deeppink',
        linestyle=':',
        linewidth=4)

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label='macro-average ROC curve (area = {0:0.2f})'
        ''.format(roc_auc["macro"]),
        color='navy',
        linestyle=':',
        linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(
    #         fpr[i],
    #         tpr[i],
    #         color=color,
    #         lw=linewidth,
    #         label='ROC curve of class {0} (area = {1:0.2f})'
    #         ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=linewidth)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


class MetricsCallback(Callback):
    """Custom callbacks for calculating evaluation scores."""

    def __init__(self, labels=[], threshold=0.5):
        """Initialize the callback.

        Args:
            labels: list
                    All available classes
            threshold:  float
                        The classification threshold.
        """
        self.threshold = threshold
        self.labels = labels

    def on_train_begin(self, logs={}):
        """Initialize variables at beginning of training."""
        self.losses = []
        self.aucs = []
        self.report = {}
        self.f1s_weighted = []
        self.f1s_macro = []
        self.f1s_micro = []

        self.recalls_weighted = []
        self.recalls_macro = []
        self.recalls_micro = []

        self.precisions_weighted = []
        self.precisions_macro = []
        self.precisions_micro = []

    def _show_random_results(self):
        """Compare scores to scores of random predictions."""
        print("Classification report for random results:\n")
        y_true = self.validation_data[1]

        random_predictions = np.random.uniform(0., 1.0, size=y_true.shape)
        random_predictions[random_predictions >= self.threshold] = 1
        random_predictions[random_predictions < self.threshold] = 0

        print(classification_report(
            y_true, random_predictions, target_names=self.labels, digits=3))

        _, _, _, support = precision_recall_fscore_support(
            y_true,
            random_predictions,
            average=None,
            labels=range(len(self.labels)))

    def _get_classification_report(self, print_report=True):
        """..."""
        y_true = self.validation_data[1]

        y_pred = self.model.predict(self.validation_data[0])

        y_pred[y_pred >= self.threshold] = 1
        y_pred[y_pred < self.threshold] = 0

        self.report = classification_report(
            y_true,
            y_pred,
            target_names=self.labels,
            digits=3,
            output_dict=True)

        if print_report:
            print("Classification report: \n")
            print(classification_report(
                y_true,
                y_pred,
                target_names=self.labels,
                digits=3,
                output_dict=False))

        _, _, _, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(self.labels)))

    def on_train_end(self, logs={}):
        """Collect the evaluation metrics."""
        self._get_classification_report(print_report=False)
        # print(self.report)
        # self._show_random_results()

        return

    def on_epoch_begin(self, epoch, logs={}):
        """dummy."""
        return

    def _calculate_auc(self, y_true, y_pred):
        """Compute AUC score."""
        try:
            roc_auc = roc_auc_score(y_true, y_pred)
            self.aucs.append(roc_auc)
            return roc_auc

        except ValueError:
            return 0.0

    def _calculate_f1(self, y_true, y_pred):
        """Compute F1 score."""
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_micro = f1_score(y_true, y_pred, average="micro")

        self.f1s_weighted.append(f1_weighted)
        self.f1s_macro.append(f1_macro)
        self.f1s_micro.append(f1_micro)

        return f1_weighted

    def _calculate_precision(self, y_true, y_pred):
        """Compute precision score."""
        precision_weighted = precision_score(
            y_true, y_pred, average="weighted")
        precision_macro = precision_score(y_true, y_pred, average="macro")
        precision_micro = precision_score(y_true, y_pred, average="micro")

        self.precisions_weighted.append(precision_weighted)
        self.precisions_macro.append(precision_macro)
        self.precisions_micro.append(precision_micro)

        return precision_weighted

    def _calculate_recall(self, y_true, y_pred):
        """Compute recall."""
        recall_weighted = recall_score(y_true, y_pred, average="weighted")
        recall_macro = recall_score(y_true, y_pred, average="macro")
        recall_micro = recall_score(y_true, y_pred, average="micro")

        self.recalls_weighted.append(recall_weighted)
        self.recalls_macro.append(recall_macro)
        self.recalls_micro.append(recall_micro)

        return recall_weighted

    def on_epoch_end(self, batch, logs={}):
        """Calculate score at the end of each epoch.

        Can be accessed during training.
        """
        self.losses.append(logs.get('loss'))

        y_true = self.validation_data[1]
        y_pred = self.model.predict(self.validation_data[0])

        y_pred[y_pred >= self.threshold] = 1
        y_pred[y_pred < self.threshold] = 0

        # self._calculate_auc(y_true, y_pred)
        f1_weighted = self._calculate_f1(y_true, y_pred)
        prec_weighted = self._calculate_precision(y_true, y_pred)
        rec_weighted = self._calculate_recall(y_true, y_pred)

        print("- weighted F1 (threshold: {}): {} - weighted precision: {} - "
              "weighted recall: {} ~~ filtering warnings ~~".format(
                  self.threshold, f1_weighted, prec_weighted, rec_weighted))

        # print("Time for calculating scores: {}".format(t1 - t0))
        return

    def on_batch_begin(self, batch, logs={}):
        """dummy."""
        return

    def on_batch_end(self, batch, logs={}):
        """dummy."""
        return
