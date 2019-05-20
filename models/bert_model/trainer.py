# coding=utf-8
import os
from typing import List, Dict

import numpy as np
import torch
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import trange, tqdm

from models.bert_model.processors import DataProcessor
from models.bert_model.utils import convert_examples_to_features, convert_example_to_feature, logger, create_dirs
from models.bert_model.utils import warmup_linear, accuracy

SEED = 42


class Trainer:
    def __init__(self, model, optimizer, epochs, batch_size, num_train_steps, device, train_loader, val_loader=None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_train_steps = num_train_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = 1.0
        self.warmup_proportion = 0.1
        self.learning_rate = 2e-5
        self.device = device
        self.loss_fnc = CrossEntropyLoss()

    def run_train_loop(self):
        global_step = 0
        logger.info("==> 🏋 Training:")
        logger.info("  Num examples = %d", len(self.train_loader))
        logger.info("  Batch size = %d", self.batch_size)
        logger.info("  Num steps = %d", self.num_train_steps)
        for _ in trange(int(self.epochs), desc="Epoch"):
            self.model.train()

            running_loss = 0.
            total_acc, total_num = 0, 0
            # nb_tr_examples, nb_tr_steps = 0, 0

            for i, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # Forward pass
                logits = self.model(input_ids, segment_ids, input_mask)

                # Compute loss
                # Compute loss
                loss = cross_entropy(logits.view(-1, self.model.num_labels), label_ids.view(-1))
                loss.backward()

                # Add mini-batch loss to epoch loss
                running_loss += loss.item()
                running_loss += loss
                global_step += 1
                # gradient clipping
                lr_this_step = self.learning_rate * warmup_linear(global_step / self.num_train_steps,
                                                                  self.warmup_proportion)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                # Update weights
                self.optimizer.step()

                # Compute accuracy
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                total_acc += accuracy(logits, label_ids)
                total_num += input_ids.size(0)

            logger.info("==> 💯 Train performance:")
            logger.info("Train loss: {:.4f}".format(running_loss / len(self.train_loader)))
            logger.info("Train Accuracy: {:.4f}".format(total_acc / total_num))

            # Validation
            if self.val_loader:
                val_acc, val_loss = self.validate()

    def validate(self):
        self.model.eval()  # Set model to eval mode due to Dropout
        running_acc = 0.
        running_loss = 0.
        total_num = 0

        for input_ids, input_mask, segment_ids, label_ids in tqdm(self.val_loader, desc="Evaluating"):
            # Set all tensors to use cuda
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)

            # Make prediction with model
            with torch.no_grad():
                tmp_eval_loss = self.model(input_ids, segment_ids, input_mask, label_ids)
                logits = self.model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            # Compute accuracy
            running_loss += tmp_eval_loss.mean().item()
            running_acc += tmp_eval_accuracy

            total_num += input_ids.size(0)

            # self.model.train()  # Set model back to training mode
        running_loss = running_loss / len(self.val_loader)
        running_acc = running_acc / total_num

        logger.info("==> 💯 Validation performance:")
        logger.info("Validation loss: {:.4f}".format(running_loss))
        logger.info("Validation Accuracy: {:.4f}%".format(running_acc))
        return running_acc, running_loss

