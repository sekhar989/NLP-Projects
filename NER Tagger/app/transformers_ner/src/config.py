#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""
    __author__: archie
    Created Date: Sun, 12 Jun 2022; 13:14:51
"""
import torch
from transformers import RobertaTokenizerFast, TrainingArguments

# PATHs
data_path = '../data/'
words_path = data_path + 'words.txt'
tags_path = data_path + 'tags.txt'

train_data = {"sentences": data_path + 'train/sentences.txt',
              "labels": data_path + 'train/labels.txt'}

validation_data = {"sentences": data_path + 'val/sentences.txt',
                   "labels": data_path + 'val/labels.txt'}

test_data = {"sentences": data_path + 'test/sentences.txt',
             "labels": data_path + 'test/labels.txt'}

vocab_map_path = data_path + 'vocab_map.pickle'
reverse_vocab_map_path = data_path + 'reverse_vocab_map.pickle'

tag_map_path = data_path + 'tags_map.pickle'
reverse_tag_map_path = data_path + 'reverse_tags_map.pickle'

# Config specific to model e.g. RobertaConfig
model_name = "roberta-base"
tokenizer = RobertaTokenizerFast.from_pretrained(model_name,
                                                 add_prefix_space=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# Training parameters i.e training config
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=10,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

epochs = 10
weight_decay=0.01,               # strength of weight decay
num_train_epochs=10,             # total number of training epochs

learning_rate = 1e-2
lr_step_size = 2           # every 2 epochs
lr_reduction_pc = 0.95     # 95% reduction of lr


checkpoint_dir = 'trained_models/'