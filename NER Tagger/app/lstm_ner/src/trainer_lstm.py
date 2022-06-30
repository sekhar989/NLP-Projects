#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""
    __author__: archie
    Created Date: Mon, 06 Jun 2022; 22:14:38
"""

import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.utils import prediction_evaluation, printf


def trainer(model, epochs, vocab, train_loader, valid_loader,
            loss_fn, optimizer, lr_scheduler, gradient_clip_threshold,
            device, checkpoint_path, writer):

    last_best = float(-1)
    validation_metric_tracker = []
    steps = 0

    # Epoch Loop --> 1 epoch is one forward pass through all the training data
    for _ in tqdm(range(epochs), dynamic_ncols=True, desc='Epoch Progress'):

        # set model to training mode
        model.train()
        epoch_loss, train_predictions = [], []

        # training loop for each mini-batch
        for sentences, true_tags in tqdm(train_loader, dynamic_ncols=True, desc='Training Progress'):
            # set grads to zero
            optimizer.zero_grad()

            # forward pass
            sentences, true_tags = sentences.to(device), true_tags.to(device)
            predicted_tags = model(sentences)

            # loss calculation
            loss = loss_fn(predicted_tags.view(-1, predicted_tags.shape[2]),
                           true_tags.view(-1))

            # Track Loss for each epoch
            epoch_loss.append(loss.item())

            # calculate training metrics
            train_accuracy = prediction_evaluation(torch.argmax(
                predicted_tags, dim=-1), true_tags, pad_value=vocab['<PAD>'])
            train_predictions.append(train_accuracy.item())

            # backward pass i.e. gradient calculations
            loss.backward()

            # Step 4+: clip the gradient, to avoid gradient explosion
            if gradient_clip_threshold is not None:
                nn.utils.clip_grad_value_(model.parameters(),
                                          clip_value=gradient_clip_threshold)

            # optimizer step
            optimizer.step()

            # increment training steps
            steps += 1

        printf('Training {:>4d} steps  --- Accuracy {:>5.4f}  |  Epoch Loss  {:>5.4f}'.format(steps,
                                                                                              np.mean(train_predictions),
                                                                                              np.mean(epoch_loss)))

        # validation loop after each training epoch
        # set no grad
        with torch.no_grad():
            model.eval()
            validation_predictions = []

            # perform validation
            # validation loop for each mini-batch
            for val_sentences, val_true_tags in tqdm(valid_loader, dynamic_ncols=True, desc='Validation Progress'):

                val_sentences, val_true_tags = val_sentences.to(
                    device), val_true_tags.to(device)
                predicted_tags = model(val_sentences)

                # calculate validation metrics
                validation_accuracy = prediction_evaluation(torch.argmax(
                    predicted_tags, dim=-1), val_true_tags, pad_value=vocab['<PAD>'])
                validation_predictions.append(validation_accuracy.item())

            printf('Validation Accuracy {:>5.4f}'.format(
                np.mean(validation_predictions)))

            validation_metric_tracker.append(np.mean(validation_predictions))

        lr_scheduler.step()

        # save checkpoint (conditioned on performance or time-steps)
        if last_best < validation_metric_tracker[-1]:

            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            printf(f"Saving model @ {checkpoint_path}")

            torch.save({'model': model.state_dict(),
                        'optim': optimizer.state_dict()},
                       checkpoint_path+f'trained_steps_{steps}.pt')
            last_best_name = checkpoint_path+f'trained_steps_{steps}.pt'

        writer.add_scalars('loss', {'loss': np.mean(epoch_loss)}, steps)

        writer.add_scalars('accuracy/train',
                           {'train_accuracy': np.mean(
                               np.mean(train_predictions))},
                           steps)
        writer.add_scalars('accuracy/validation',
                           {'validation_accuracy': np.mean(
                               validation_predictions)},
                           steps)

    return last_best_name
