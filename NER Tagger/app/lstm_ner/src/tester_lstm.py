#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""
    __author__: archie
    Created Date: Thu, 09 Jun 2022; 14:15:03
"""

import numpy as np
import torch
from tqdm import tqdm

from src.dataloader import convert_sentence2id, convert_id2tokens, convert_id2tag
from src.utils import prediction_evaluation, printf
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def tester(model, test_loader, vocab_map, id2vocab, id2tags, device):

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
            if every2 % 2 == 0:
                print_predictions(test_sentences[0], predicted_tags[0],
                                  test_true_tags[0], id2vocab, id2tags,
                                  vocab_map['<PAD>'])

            # calculate validation metrics
            test_accuracy = prediction_evaluation(predicted_tags,
                                                  test_true_tags,
                                                  pad_value=vocab_map['<PAD>'])

            test_metrics.append(test_accuracy.item())

        printf('Test Avg. Accuracy {:>5.4f}'.format(
            np.mean(test_metrics)))


def print_predictions(encoded_sentence, pred_tags, true_tags, reverse_vocab_map, reverse_tag_map, pad_token):

    red = "\033[1;31m"
    green = "\033[1;32m"
    purple = "\033[1;35m"
    reset = "\033[0m"

    sentence = convert_id2tokens(encoded_sentence.detach().cpu().numpy(),
                                 reverse_vocab_map)

    if len(true_tags) > 0:
        mask = true_tags != pad_token
        pred_tags_converted = convert_id2tag(pred_tags[mask].detach().cpu().numpy(),
                                             reverse_tag_map)

        true_tags_converted = convert_id2tag(true_tags[mask].detach().cpu().numpy(),
                                             reverse_tag_map)

        iterator = zip(sentence, pred_tags_converted, true_tags_converted)
        printf("{:>20} | {:>7} | {:>5}".format(
            "Token", "Pred", "True NER Tag"))
        printf('-'*70)
        for s, p, t in iterator:
            if p == t:
                p = green + p + reset
            else:
                p = red + p + reset
            t = purple + t + reset
            printf("{:>20} | {:>17}  | {:>15}".format(s, p, t))
        return

    pred_tags_converted = convert_id2tag(pred_tags.detach().cpu().numpy(),
                                         reverse_tag_map)
    iterator = zip(sentence, pred_tags_converted)
    printf("{:>20} | {:>7}".format("Token", "Pred"))
    printf('-'*70)
    for s, p in iterator:
        printf("{:>20} | {:>17}".format(s, p))

    return sentence, pred_tags_converted


def raw_input_pred(model, text, vocab2id, id2vocab, id2tag, device):
    encoded_sentence = convert_sentence2id(text, vocab2id)
    encoded_sentence = torch.tensor(encoded_sentence,
                                    dtype=torch.long).to(device)
    predicted_tags = model(encoded_sentence)
    predicted_tags = torch.argmax(predicted_tags, dim=-1)
    print_predictions(encoded_sentence, predicted_tags, [],
                      id2vocab, id2tag, vocab2id['<PAD>'])
    