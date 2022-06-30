#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""
    __author__: archie
    Created Date: Mon, 06 Jun 2022; 22:14:52
"""

import torch

# Data Paths
data_path = '../data/'
words_path = data_path + 'words.txt'
tags_path = data_path + 'tags.txt'

api_data_path = 'data/'

vocab_map_path = data_path + 'vocab_map.pickle'
reverse_vocab_map_path = data_path + 'reverse_vocab_map.pickle'

tag_map_path = data_path + 'tags_map.pickle'
reverse_tag_map_path = data_path + 'reverse_tags_map.pickle'

train_data = {"sentences": data_path + 'train/sentences.txt',
              "labels": data_path + 'train/labels.txt'}

validation_data = {"sentences": data_path + 'val/sentences.txt',
                   "labels": data_path + 'val/labels.txt'}

test_data = {"sentences": data_path + 'test/sentences.txt',
             "labels": data_path + 'test/labels.txt'}


# Model Training Config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lstm_embedding_size = 50
dense_layer_size = 128
lstm_batch_size = 512
learning_rate = 1e-2
lr_step_size = 2           # every 2 epochs
lr_reduction_pc = 0.95     # 95% reduction of lr

grad_clip_threshold = 0.3

epochs = 25

checkpoint_dir = 'trained_models/'
best_model = "trained_models/NER_Tagger/trained_steps_3300.pt"
