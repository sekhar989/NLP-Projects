#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""
    __author__: archie
    Created Date: Thu, 09 Jun 2022; 17:45:50
"""
import pickle

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, RobertaTokenizerFast

from models.robert_ner import RobertaNER
from src import config
from src.dataloader import create_transformers_data_loader
from src.trainer_roberta import roberta_custom_trainer
from src.tester_roberta import roberta_tester

# Load vocabs
with open(config.tag_map_path, 'rb') as f:
    tag2id = pickle.load(f)

with open(config.reverse_tag_map_path, 'rb') as f:
    id2tag = pickle.load(f)

# Initialize Model Specific Tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base',
                                                 add_prefix_space=True)

# Data Loaers
train_dataset = create_transformers_data_loader(config.train_data['sentences'],
                                                config.train_data['labels'],
                                                tag2id,
                                                tokenizer)

valid_dataset = create_transformers_data_loader(config.validation_data['sentences'],
                                                config.validation_data['labels'],
                                                tag2id,
                                                tokenizer)

test_dataset = create_transformers_data_loader(config.test_data['sentences'],
                                               config.test_data['labels'],
                                               tag2id,
                                               tokenizer)

# Initialize Model
# ** Make sure to mention model config
roberta_custom_config = AutoConfig.from_pretrained(config.model_name,
                                                   num_labels=len(tag2id),
                                                   id2label=id2tag,
                                                   label2id=tag2id)
# Initialize pre-trained weights
model = RobertaNER.from_pretrained(config.model_name,
                                   config=roberta_custom_config).to(config.device)

# Loss/Cost Function
loss_function = nn.CrossEntropyLoss().to(config.device)

# Optimizer --> model parameters and learning rate
# Class weights can also be specified if necessary
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# learning rate scheduler --> slow reduction of learning rate from a higher value
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                               step_size=config.lr_step_size,
                                               gamma=config.lr_reduction_pc)

checkpoint_dir = config.checkpoint_dir + f'{model._get_name()}/'
logging_dir = checkpoint_dir + "logs/"
writer = SummaryWriter(log_dir=logging_dir)
best_model = roberta_custom_trainer(model, config.epochs, train_dataset, valid_dataset, id2tag,
                                    loss_function, lr_scheduler, optimizer,
                                    config.device, checkpoint_dir, writer)

checkpoint = torch.load(best_model)
model.load_state_dict(checkpoint['model'])

roberta_tester(model, test_dataset, id2tag, config.device)
