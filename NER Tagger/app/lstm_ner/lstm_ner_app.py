#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""
    __author__: archie
    Created Date: Thu, 09 Jun 2022; 23:59:16
"""

import torch
from fastapi import FastAPI

import src.api_pred as ner_tag
from models.lstm_ner import NER_Tagger

vocab_map = ner_tag.vocab_map
id2tag = ner_tag.id2tag
config = ner_tag.config
device = 'cpu'

model = NER_Tagger(len(vocab_map),
                   config.lstm_embedding_size,
                   len(id2tag))
best_model = torch.load(config.best_model,
                        map_location=torch.device(device))
model.load_state_dict(best_model['model'])
model.to(device)
model.eval()

app = FastAPI()


@app.get("/lstm_ner/only_tags/{sentence}")
def ner_tagger(sentence: str = "", num_tags: int = None):
    # Perform the tagging using predefineed function
    tags = ner_tag.ner_pred_api(model, sentence, vocab_map, id2tag, device)
    if num_tags:
        tags = tags[:num_tags]

    # Combine the data in a json format
    return {'predicted_tags': tags}


@app.get("/lstm_ner/pairs/{sentence}")
def ner_tagger_pairs(sentence: str = ""):
    # Perform the tagging using predefineed function
    tags = ner_tag.ner_pred_api(model, sentence, vocab_map, id2tag, device)
    sentence = sentence.split(' ')
    # Combine the data in a json format
    return {token: tag for token, tag in zip(sentence, tags) if tag != 'O'}


@app.get("/lstm_ner/ner/{sentence}")
def get_ner(sentence: str = ""):
    # Perform the tagging using predefineed function
    tags = ner_tag.ner_pred_api(model, sentence, vocab_map, id2tag, device)
    sentence = sentence.split(' ')
    # Combine the data in a json format
    return {"ner": [sentence[index] for index in range(len(tags)) if tags[index] != 'O']}
