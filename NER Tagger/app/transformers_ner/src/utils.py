#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""
    __author__: archie
    Created Date: Sun, 12 Jun 2022; 13:18:19
"""
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2


def printf(message):
    tqdm.wriet(message)


def prediction_seqeval(predicted_labels, true_labels, id2labels, scheme=IOB2, verbose=True):

    masks = [[label != -100 for label in true_label]
             for true_label in true_labels]

    true_labels = [labels[masks[i]] for i, labels in true_labels]

    predicted_labels = [[id2labels[i] for i in preds[masks[index]]]
                        for index, preds in enumerate(predicted_labels)]

    report = classification_report(predicted_labels,
                                   true_labels,
                                   scheme=scheme)
    if verbose:
        printf(report)

    return report


def prediction_f1(predicted_labels, true_labels, id2labels, scheme=IOB2):

    masks = [[label != -100 for label in true_label]
             for true_label in true_labels]

    true_labels = [[id2labels[i] for i in labels[masks[idx]]]
                   for idx, labels in enumerate(true_labels)]

    predicted_labels = [[id2labels[i] for i in preds[masks[index]]]
                        for index, preds in enumerate(predicted_labels)]

    score = f1_score(predicted_labels,
                     true_labels,
                     scheme=scheme)

    return score
