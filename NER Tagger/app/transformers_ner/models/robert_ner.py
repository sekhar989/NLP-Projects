#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""
    __author__: archie
    Created Date: Sun, 05 Jun 2022; 11:57:01
"""
from torch import nn
from transformers import RobertaConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaModel, RobertaPreTrainedModel)


class RobertaNER(RobertaPreTrainedModel):
    config_class = RobertaConfig

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        # Pre-trained model
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # Custom Layers
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):

        outputs = self.roberta(input_ids, attention_mask,
                               token_type_ids, **kwargs)

        sequence_output = self.dropout(outputs[0])

        logits = self.classifier(sequence_output)

        # loss = None

        # if labels is not None:
        #     loss_function = nn.CrossEntropyLoss()
        #     loss = loss_function(
        #         logits.view(-1, logits.shape[2]), labels.view(-1))

        # return TokenClassifierOutput(loss=loss,
        #                              logits=logits,
        #                              hidden_states=outputs.hidden_states,
        #                              attentions=outputs.attentions)

        return logits
