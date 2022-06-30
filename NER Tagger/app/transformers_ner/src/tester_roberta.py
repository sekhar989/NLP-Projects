#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""
    __author__: archie
    Created Date: Sun, 12 Jun 2022; 13:16:12
"""

import numpy as np
import torch
from tqdm import tqdm

from src.utils import prediction_f1, printf


def roberta_tester(model, test_loader, id2labels, device):

    with torch.no_grad():
        model.eval()

        test_metrics = []
        every2 = 0

        # perform validation
        # validation loop for each mini-batch
        for test_sentences, test_true_tags in tqdm(test_loader, dynamic_ncols=True, desc='Validation Progress'):

            test_sentences = test_sentences.to(device)
            test_true_tags = test_true_tags.to(device)
            predicted_tags = model(test_sentences)

            predicted_tags = torch.argmax(predicted_tags, dim=-1)

            # calculate validation metrics
            test_accuracy = prediction_f1(predicted_tags,
                                          test_true_tags, id2labels)

            test_metrics.append(test_accuracy.item())

        printf('Test Avg. Accuracy {:>5.4f}'.format(np.mean(test_metrics)))
