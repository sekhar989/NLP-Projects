#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""
    __author__: archie
    Created Date: Sun, 05 Jun 2022; 11:57:01
"""

import torch.nn.functional as F
from torch import nn


class NER_Tagger(nn.Module):

    def __init__(self, vocab_size: int, embedding_size: int, dense_output_size: int, device: str = 'cpu') -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(
            embedding_size, embedding_size//2, batch_first=True)
        self.dense = nn.Linear(embedding_size//2, dense_output_size)

        self.to(device)

    def forward(self, X):
        """
        processed --> will have 3 outputs
            - hidden states for each input sequence
            - final hidden state for each element in the sequence
            - final cell state for each element in the sequence

        processed[0].shape = (batch_size, sequence_length, h_out_size)
        cessed[1][0].shape = (1, batch_size, h_out_size)
        processed[1][1].shape = (1, batch_size, h_out_size)
        """

        embedded = self.embedding(X)
        processed = self.lstm(embedded)
        processed = self.dense(processed[0])
        return F.log_softmax(processed, dim=-1)
