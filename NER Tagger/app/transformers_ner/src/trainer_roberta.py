#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""
    __author__: archie
    Created Date: Sun, 12 Jun 2022; 13:15:24
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from src.utils import prediction_f1, printf


def roberta_custom_trainer(model, epochs, train_loader, validation_loader, id2label,
                           loss_fn, lr_scheduler, optimizer,
                           device, checkpoint_path, writer):

    last_best = float(-1)
    validation_metric_tracker = []
    steps = 0

    # Epoch Loop --> 1 epoch is one forward pass through all the training data
    for _ in tqdm(range(epochs), dynamic_ncols=True, desc='Epoch Progress'):

        # set model to training mode
        model.train()
        train_metrics = []
        epoch_loss = []
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # training loop for each mini-batch
        for train_data in tqdm(train_loader, dynamic_ncols=True, desc='Training Progress:'):

            # set grads to zero
            optimizer.zero_grad()

            # forward pass
            true_tags = train_data['labels'].to(device)

            train_data['input_ids'] = train_data['input_ids'].to(device)
            train_data['attention_mask'] = train_data['attention_mask'].to(
                device)

            predicted_tags = model(input_ids=train_data['input_ids'],
                                   attention_mask=train_data['attention_mask'])

            # loss calculation
            loss = loss_fn(predicted_tags.view(-1, predicted_tags.shape[2]),
                           true_tags.view(-1))

            # Track Loss for each epoch
            epoch_loss.append(loss.item())

            predicted_labels = torch.argmax(predicted_tags,
                                            dim=-1)

            # backward pass i.e. gradient calculations
            loss.backward()

            train_score = prediction_f1(predicted_labels.detach().cpu().numpy(),
                                        true_tags.detach().cpu().numpy(),
                                        id2label)

            train_metrics.append(train_score)

            # optimizer step
            optimizer.step()

            # increment training steps
            steps += 1

        printf('Training {:>4d} steps  --- Avg. F1-Score {:>5.4f}  |  Epoch Loss  {:>5.4f}'.format(steps,
                                                                                                   np.mean(
                                                                                                       train_metrics),
                                                                                                   np.mean(epoch_loss)))

        # validation loop after each training epoch
        # set no grad
        with torch.no_grad():
            model.eval()
            validation_predictions = []

            # perform validation
            # validation loop for each mini-batch
            for valid_data in validation_loader:

                true_labels = valid_data['labels']
                output = model(input_ids=valid_data['input_ids'],
                               attention_mask=valid_data['attention_masks'])
                predicted_tags = output["logits"]

                # calculate validation metrics
                validation_accuracy = prediction_f1(torch.argmax(predicted_tags, dim=-1),
                                                    true_labels,
                                                    )
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
                               np.mean(train_metrics))},
                           steps)
        writer.add_scalars('accuracy/validation',
                           {'validation_accuracy': np.mean(
                               validation_predictions)},
                           steps)

    return last_best_name
