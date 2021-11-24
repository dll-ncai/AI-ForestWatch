# Copyright (c) 2021, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, test_data_loader=None,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.test_data_loader = test_data_loader
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, item in enumerate(self.data_loader):
            data, target = item['input'], item['label']
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            out_x, logits = self.model(data)
            pred = torch.argmax(logits, dim=1)
            not_one_hot_target = torch.argmax(target, dim=1)
            loss = self.criterion(logits, not_one_hot_target)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 0.05)
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(logits, not_one_hot_target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch()
            log.update(**{'val_'+k : v for k, v in val_log.items()})
        if self.do_test and epoch == self.config['trainer']['epochs']:
            test_log = self._test_epoch()
            log.update(**{'test_'+k : v for k, v in test_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, item in enumerate(self.valid_data_loader):
                data, target = item['input'], item['label']
                data, target = data.to(self.device), target.to(self.device)

                out_x, logits = self.model(data)
                pred = torch.argmax(logits, dim=1)
                not_one_hot_target = torch.argmax(target, dim=1)
                loss = self.criterion(logits, not_one_hot_target)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(logits, not_one_hot_target))
        return self.valid_metrics.result()
    
    def _test_epoch(self):
        """
        Test after training

        :return: A log that contains information about testing
        """
        self.model.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for batch_idx, item in enumerate(self.test_data_loader):
                data, target = item['input'], item['label']
                data, target = data.to(self.device), target.to(self.device)

                out_x, logits = self.model(data)
                pred = torch.argmax(logits, dim=1)
                not_one_hot_target = torch.argmax(target, dim=1)
                loss = self.criterion(logits, not_one_hot_target)

                self.test_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(logits, not_one_hot_target))
        return self.test_metrics.result()
    

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
