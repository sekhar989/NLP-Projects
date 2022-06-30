#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""
    __author__: archie
    Created Date: Thu, 09 Jun 2022; 13:03:03
"""

import pickle

from tqdm import tqdm


def create_vocab(words_path, vocab_path, reverse_map_path):
    vocab = {}
    with open(words_path) as f:
        for i, l in enumerate(f.read().splitlines()):
            vocab[l] = i
    vocab['<PAD>'] = len(vocab)

    reverse_vocab_map = {v: k for k, v in vocab.items()}

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(reverse_map_path, 'wb') as f:
        pickle.dump(reverse_vocab_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    return vocab, reverse_map_path


def create_tag_map(tags_path, tag_map_path, reverse_map_path):
    tag_map = {}
    with open(tags_path) as f:
        for i, l in enumerate(f.read().splitlines()):
            tag_map[l] = i

    reverse_tag_map = {v: k for k, v in tag_map.items()}

    with open(tag_map_path, 'wb') as f:
        pickle.dump(tag_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(reverse_map_path, 'wb') as f:
        pickle.dump(reverse_tag_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    return tag_map, reverse_tag_map


def prediction_evaluation(predictions, true_labels, pad_value):
    """
    Inputs:
        pred: prediction array with shape -> (num_examples, max sentence length in batch)
        labels: array of size (batch_size, seq_len)
        pad: integer representing pad character
    Outputs:
        accuracy: float
    """
    # Create mask matrix equal to the shape of the true-labels matrix
    pad_mask = true_labels != pad_value

    # Calculate Accuracy
    accuracy = ((predictions == true_labels) * pad_mask).sum() / pad_mask.sum()

    return accuracy


def printf(message):
    tqdm.write(message)
