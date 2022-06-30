#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""
    __author__: archie
    Created Date: Sat, 04 Jun 2022; 19:37:26
    
    Two Tasks performed by 2-classes:
        1. Dataset:
                __init__
            - Understands the location of the data
            - Understands features [X] and labels [Y]
            - Understands any specifics related to the data e.g. for NLP tasks knowing the maximum length of the sequences
                __len__
            - Returns the length of the data
            
                __get_item__
            - Performs all necessary preprocessings
            - Returns the preprocessed data
            
        2. Dataloader:
            - Requires a Dataset class as input
            - Manages shuffling of data
            - Returns a batch of data according to the desired batch size    
"""

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def convert_sentence2id(sentence: str, vocab_map: Dict) -> List:
    """
    Description:
    -----------

    Converts input sentence strings into word tokens and then maps them to its
    corresponding ids present in the vocabulary

    Arguments:
    ---------
        sentence {str} -- e.g. The quick brown fox just jumped over the lazy dog
        vocab_map {Dict} -- map (dictionary) of tokens to its corresponding unique ids

    Returns:
    -------
        List -- List of token ids
    """
    sentence_id = []
    for token in sentence.split(' '):
        if token in vocab_map:
            sentence_id.append(vocab_map[token])
        else:
            sentence_id.append(vocab_map['UNK'])
    return sentence_id


def convert_id2tokens(id_list: List, reverse_vocab_map: Dict):
    """
    Description:
    -----------

    Converts a list of word ids, into it's corresponding tokens

    Arguments:
    ---------
        id_list {List} -- list of predicted ids
        reverse_vocab_map {Dict} -- id to token mappings

    Returns:
    -------
        list of tokens
    """
    tokenized = [reverse_vocab_map[i] for i in id_list]
    return tokenized


def convert_tags2id(tag_list: str, tag_map: Dict) -> List:
    """
    Description:
    -----------

    Converts ner tags to its corresponding ids as
    mentioned in the tag map

    Arguments:
    ---------
        tag_list {str} -- a space separated string of NER tags 
                          e.g. B-org I-org B-per I-per I-per O O O O B-tim O O O B-geo O
        tag_map {Dict} -- map (dictionary) of tags to its corresponding unque ids

    Returns:
    -------
        List -- List of tag ids
    """
    tag_id = []
    for label in tag_list.split(' '):
        tag_id.append(tag_map[label])
    return tag_id


def create_ner_data_loader(sentences_path, labels_path, vocab_map, tags_map,
                           batch_size=8, shuffle=True, num_workers=4):
    """
    Description:
    -----------

    Returns a torch data-loader instance capable of shuffling & batching data

    Arguments:
    ---------
        sentences_path {_type_} -- path to sentence file
        labels_path {_type_} -- path to lables file
        vocab_map {_type_} -- token to id mapping
        tags_map {_type_} -- ner tag to id mapping

    Keyword Arguments:
    -----------------
        batch_size {int} -- (default: {8})
        shuffle {bool} -- (default: {True})
        num_workers {int} -- (default: {4})

    Returns:
    -------
        _type_ -- Torch Data Loader
    """

    ner = ner_data(sentences_path, labels_path, vocab_map, tags_map)

    return DataLoader(dataset=ner, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def convert_id2tag(tag_ids: List, reverse_tag_map: Dict):
    """
    Description:
    -----------

    Converts a list of tag ids, into it's corresponding ner tags

    Arguments:
    ---------
        tag_ids {List} -- list of ner tags ids
        reverse_tag_map {Dict} -- id to tag mappings
    """
    tag_list = [reverse_tag_map[i] for i in tag_ids]
    return tag_list


class ner_data(Dataset):
    """
    Description:
    -----------

    Generates a torch data-set class for NER Tagging.
    This dataset can be used for custom model training purposes.
    Not recommended for using to train transformer models as they need further processing.

    """

    def __init__(self, sentences_path: str, labels_path: str, vocab_map: Dict, tags_map: Dict) -> None:
        super().__init__()

        with open(sentences_path, "r") as f:
            self.sentences = f.read().splitlines()

        with open(labels_path, "r") as f:
            self.labels = f.read().splitlines()

        self.max_len = max([len(sentence) for sentence in self.sentences])
        self.vocab_map = vocab_map
        self.tags_map = tags_map

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sentence_padded = np.array(self.max_len * [self.vocab_map["<PAD>"]])
        labels_padded = np.array(self.max_len * [self.vocab_map["<PAD>"]])

        sentence = convert_sentence2id(self.sentences[idx], self.vocab_map)
        labels = convert_tags2id(self.labels[idx], self.tags_map)

        assert len(sentence) == len(labels)

        sentence_padded[:len(sentence)] = sentence
        labels_padded[:len(labels)] = labels

        return torch.tensor(sentence_padded, dtype=torch.long), torch.tensor(labels_padded, dtype=torch.long)


def raw_input_convert(raw_string: str, vocab_map: Dict):
    return convert_sentence2id(raw_string, vocab_map)


def raw_input_convert_transformer(raw_string: str, transformers_tokenizer):
    pass
