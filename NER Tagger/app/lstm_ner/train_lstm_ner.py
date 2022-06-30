#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""
    __author__: archie
    Created Date: Thu, 09 Jun 2022; 13:30:45
"""
import argparse
import os
import pickle
import sys

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from models.lstm_ner import NER_Tagger
from src import config
from src.dataloader import create_ner_data_loader
from src.tester_lstm import raw_input_pred, tester
from src.trainer_lstm import trainer
from src.utils import create_tag_map, create_vocab


def main(args):
    if args.train == "0":
        try:
            assert os.path.exists(args.model_path)
        except AssertionError:
            print("Enter Valid Model Path.. or Train the model")
            sys.exit()
        best_model = args.model_path

    # Load Vocabs
    if not os.path.exists(config.vocab_map_path):
        vocab_map, id2vocab = create_vocab(config.words_path,
                                           config.vocab_map_path,
                                           config.reverse_vocab_map_path)
    else:
        with open(config.vocab_map_path, 'rb') as f:
            vocab_map = pickle.load(f)
        with open(config.reverse_vocab_map_path, 'rb') as f:
            id2vocab = pickle.load(f)

    if not os.path.exists(config.tag_map_path):
        tags_map, id2tag = create_tag_map(config.tags_path,
                                          config.tag_map_path,
                                          config.reverse_tag_map_path)
    else:
        with open(config.tag_map_path, 'rb') as f:
            tags_map = pickle.load(f)
        with open(config.reverse_tag_map_path, 'rb') as f:
            id2tag = pickle.load(f)

    # Prepare Data Loaders
    train_loader = create_ner_data_loader(sentences_path=config.train_data['sentences'], labels_path=config.train_data['labels'],
                                          vocab_map=vocab_map, tags_map=tags_map, batch_size=config.lstm_batch_size)

    valid_loader = create_ner_data_loader(sentences_path=config.validation_data['sentences'], labels_path=config.validation_data['labels'],
                                          vocab_map=vocab_map, tags_map=tags_map, batch_size=config.lstm_batch_size)

    test_loader = create_ner_data_loader(sentences_path=config.test_data['sentences'], labels_path=config.test_data['labels'],
                                         vocab_map=vocab_map, tags_map=tags_map, batch_size=config.lstm_batch_size)

    # Model Init and Model Configs
    model = NER_Tagger(vocab_size=len(vocab_map),
                       embedding_size=config.lstm_embedding_size,
                       dense_output_size=len(tags_map),
                       device=config.device)

    # Loss/Cost Function
    loss_function = nn.CrossEntropyLoss(
        ignore_index=vocab_map['<PAD>']).to(config.device)

    # Optimizer --> model parameters and learning rate
    # Class weights can also be specified if necessary
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=config.learning_rate)

    # learning rate scheduler --> slow reduction of learning rate from a higher value
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                   step_size=config.lr_step_size,
                                                   gamma=config.lr_reduction_pc)

    if args.train == "1":
        checkpoint_dir = config.checkpoint_dir + f'{model._get_name()}/'
        logging_dir = checkpoint_dir + "logs/"
        writer = SummaryWriter(log_dir=logging_dir)
        best_model = trainer(model, config.epochs, vocab_map,
                             train_loader, valid_loader,
                             loss_function, optimizer, lr_scheduler, config.grad_clip_threshold,
                             config.device, checkpoint_dir,
                             writer)

    checkpoint = torch.load(best_model)
    model.load_state_dict(checkpoint['model'])

    if args.train == "1":
        tester(model, test_loader, vocab_map, id2vocab, id2tag, config.device)

    if not args.tagText:
        raw_text = input("Enter Valid Text for Tagging:\n-->\t")
    else:
        raw_text = args.tagText
    raw_input_pred(model, raw_text, vocab_map, id2vocab, id2tag, config.device)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="NER Arguements")
    args.add_argument("-t", "--train", default="1",
                      help="0 or 1. 0 indicates false 1 is true")
    args.add_argument("-g", "--tagText", default="")
    args.add_argument("-m", "--model_path",
                      default="trained_models/NER_Tagger/trained_steps_3300.pt")

    parsed_args = args.parse_args()
    main(parsed_args)
