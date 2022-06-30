#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""
    __author__: archie
    Created Date: Fri, 10 Jun 2022; 19:30:24
"""
import pickle
from itertools import repeat

import src.config as config
import torch
from src.dataloader import convert_id2tag, convert_sentence2id

with open(config.vocab_map_path, 'rb') as f:
    vocab_map = pickle.load(f)
with open(config.reverse_tag_map_path, 'rb') as f:
    id2tag = pickle.load(f)


def ner_pred_api(model, text, vocab2id, id2tag, device):

    # Tokenizes using pre-defined tokeenizer or vocab-map i.e. custom tokenizer
    encoded_sentence = convert_sentence2id(text, vocab2id)

    # prepares the data for model prediction
    encoded_sentence = torch.tensor(encoded_sentence,
                                    dtype=torch.long).to(device)

    # predicts the tags
    predicted_tags = model(encoded_sentence)
    predicted_tags = torch.argmax(predicted_tags, dim=-1)

    # convert tags to ids
    predicted_tags = convert_id2tag(predicted_tags.detach().cpu().numpy(),
                                    id2tag)

    # return tags
    return predicted_tags
