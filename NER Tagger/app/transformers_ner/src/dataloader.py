#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""
    __author__: archie
    Created Date: Sun, 12 Jun 2022; 13:12:26
    
    Here the data is to be partially pre-processed before sending it to the model specific tokenizer.
    Not always necessary but this is the approach we're going to follow for this specific task i.e. Token Classification.
    
    - Partial Pre-Processing: Split the sentence at spaces i.e. tokenize by splitting at space
    - Processing: Pass the tokenized words to model specific tokenizer
        - Tokenize the labels
        - Adjust the labels vector according to the model-tokenizer output
    
"""
from typing import Dict, List
import torch

import numpy as np
from torch.utils.data import DataLoader, Dataset
from src.config import tokenizer


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


class transformers_NERDataset(Dataset):
    """
    Description:
    -----------

    Torch dataset to train transformer based models using the hugging face ecosystem.
    The dataset should be initialized with a model specific encoder name.
    """

    def __init__(self, sentences_path: str, labels_path: str,
                 tags_map: Dict, fast_tokenizer):

        self.sentence_encodings = transformers_data_prep(sentences_path,
                                                         labels_path,
                                                         tags_map,
                                                         fast_tokenizer)

    def __len__(self):
        return len(self.sentence_encodings['labels'])

    def __getitem__(self, idx):
        """
        Sentence Encodings are structured in:

            {
            'input_ids': [[...], [...], [...]]
            'attention_mask': [[...], [...], [...]]            
            }

        Return a dictionary with the requested index

        {
            'input_ids': tensor([encoded sentence at idx])
            'attention_mask': tensor([attention at idx])
            'labels': tensor([encoded labels at idx])        
        }

        """
        item = {key: torch.tensor(val[idx], dtype=torch.long)
                for key, val in self.sentence_encodings.items()}
        return item


def transformers_data_prep(sentences_path: str, labels_path: str,
                           tag_map: Dict,
                           fast_tokenizer=tokenizer):
    """
    Description:
    -----------

    _extended_summary_

    Arguments:
    ---------
        sentences {str} -- _description_
        lables {str} -- _description_
        vocab_map {Dict} -- _description_
        tag_map {Dict} -- _description_

    Returns:
    -------
        _type_ -- _description_
    """

    with open(sentences_path, 'r') as f:
        sentences = f.read().splitlines()

    with open(labels_path, "r") as f:
        labels = f.read().splitlines()

    sentences = [sentence.split() for sentence in sentences]
    labels = [convert_tags2id(tags, tag_map) for tags in labels]

    sentence_encodings = fast_tokenizer(sentences,
                                        is_split_into_words=True,
                                        padding=True,
                                        truncation=True)

    encoded_labels_array = []
    for index, label_encoded in enumerate(labels):
        word_ids = sentence_encodings.word_ids(index)
        encoded_labels = align_labels_with_tokens(label_encoded, word_ids)
        assert len(encoded_labels) == len(word_ids)
        encoded_labels_array.append(encoded_labels)

    sentence_encodings["labels"] = encoded_labels_array
    return sentence_encodings


def align_labels_with_tokens(labels, word_ids):
    """
    Description:
    -----------

    Aligning the labels with the tokens to make sure the cross entropy loss is calculated properly.
    This is a copy of the function from the hugging face website.
    Ref: https://huggingface.co/course/en/chapter7/2?fw=pt#processing-the-data

    Arguments:
    ---------
        labels {_type_} -- encoded labels
        word_ids {_type_} -- word id according to the tokenizer

    Returns:
    -------
        _type_ -- list of new labels aligned with the tokenized encoded text input
    """
    new_labels = []
    current_word = None

    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            if word_id is None:
                label = -100
            else:
                if word_id >= len(labels):
                    word_id = len(labels) - 1
                    label = labels[word_id]
                else:
                    label = labels[word_id]
            new_labels.append(label)

        elif word_id is None:
            # Special token
            # generally at the beginning of the sentence
            new_labels.append(-100)
        else:
            # Same word as previous token
            # According to wordpiece or sentencepiece tokenizer which performs
            # sub-word tokenization, breaks a longer word into sub-words with the
            # same word id effectively reducing vocabulary size but maintainng structure
            # for detokenization purpose.
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            # This will only work if B-(begin) tokens are odd
            # and all I-(Intermediate) tokens are even
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def create_transformers_data_loader(sentences_path: str, labels_path: str,
                                    tag_map: Dict, fast_tokenizer=tokenizer,
                                    batch_size=8, shuffle=True, num_workers=4):

    transformers_data_set = transformers_NERDataset(sentences_path,
                                                    labels_path,
                                                    tag_map,
                                                    fast_tokenizer)

    transformers_data_loader = DataLoader(transformers_data_set,
                                          batch_size=batch_size, shuffle=shuffle,
                                          num_workers=num_workers)

    return transformers_data_loader
