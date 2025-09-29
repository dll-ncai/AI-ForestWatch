# Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from base import BaseTrainer
from torch.nn.utils import clip_grad_norm_
from utils import MetricTracker, inf_loop
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os


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

        self.train_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns])
        
        # Initialize TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=os.path.join(config.models_dir, 'tensorboard_logs'))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        
        # Create progress bar for training
        pbar = tqdm(self.data_loader, desc=f'Training Epoch {epoch}', 
                   leave=False, ncols=100, unit='batch', 
                   disable=False, dynamic_ncols=True)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            out_x, softmaxed = self.model(data)

            # Normalize outputs to support deep supervision (list of outputs)
            # Use highest-resolution output for metrics; average loss over all outputs
            logits_list = out_x if isinstance(out_x, (list, tuple)) else [out_x]
            probs_list = softmaxed if isinstance(softmaxed, (list, tuple)) else [softmaxed]
            main_logits = logits_list[-1]
            main_probs = probs_list[-1] if len(probs_list) == len(logits_list) else None

            if main_probs is None:
                main_probs = torch.softmax(main_logits, dim=1)

            pred = torch.argmax(main_probs, dim=1)
            loss_target = target.clone()
            loss_target[loss_target != 0] -= 1
            loss_target = loss_target.squeeze(1)

            # Compute loss: for focal_loss2d use logits, otherwise use probs
            if hasattr(self.criterion, '__name__') and self.criterion.__name__ == 'focal_loss2d':
                losses = [self.criterion(logit, loss_target) for logit in logits_list]
            else:
                # ensure probs_list matches logits_list length
                if len(probs_list) != len(logits_list):
                    probs_list = [torch.softmax(l, dim=1) for l in logits_list]
                losses = [self.criterion(prob, loss_target) for prob in probs_list]
            loss = sum(losses) / len(losses)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 0.05)
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(
                    met.__name__, met(main_probs, loss_target))

            # Update progress bar with current loss
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

            # Disable the original progress logging since we have tqdm
            # if batch_idx % self.log_step == 0:
            #     self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
            #         epoch,
            #         self._progress(batch_idx),
            #         loss.item()))

            if batch_idx == self.len_epoch:
                break
        
        pbar.close()
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(self.valid_data_loader)
            log.update(**{'val_'+k: v for k, v in val_log.items()})
        if self.do_test and epoch == self.config['trainer']['epochs']:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            self._resume_checkpoint(best_path)
            self.logger.info("Testing current best: model_best.pth ...")
            test_log = self._valid_epoch(self.test_data_loader)
            log.update(**{'test_'+k: v for k, v in test_log.items()})

        # Log metrics to TensorBoard
        self._log_to_tensorboard(log, epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, data_loader):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """
        preds = np.array([])
        targets = np.array([])

        self.model.eval()
        self.valid_metrics.reset()
        
        # Create progress bar for validation
        pbar = tqdm(data_loader, desc='Validation', leave=False, ncols=100, unit='batch', 
                   dynamic_ncols=True)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)

                out_x, softmaxed = self.model(data)

                logits_list = out_x if isinstance(out_x, (list, tuple)) else [out_x]
                probs_list = softmaxed if isinstance(softmaxed, (list, tuple)) else [softmaxed]
                main_logits = logits_list[-1]
                main_probs = probs_list[-1] if len(probs_list) == len(logits_list) else None
                if main_probs is None:
                    main_probs = torch.softmax(main_logits, dim=1)

                pred = torch.argmax(main_probs, dim=1)
                loss_target = target.clone()
                loss_target[loss_target != 0] -= 1
                loss_target = loss_target.squeeze(1)
                if hasattr(self.criterion, '__name__') and self.criterion.__name__ == 'focal_loss2d':
                    losses = [self.criterion(logit, loss_target) for logit in logits_list]
                else:
                    if len(probs_list) != len(logits_list):
                        probs_list = [torch.softmax(l, dim=1) for l in logits_list]
                    losses = [self.criterion(prob, loss_target) for prob in probs_list]
                loss = sum(losses) / len(losses)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(
                        met.__name__, met(main_probs, loss_target))

                # Update progress bar with current loss
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

                label_valid_indices = (target.view(-1) != 0)
                valid_pred = pred.view(-1)[label_valid_indices]
                valid_label = target.view(-1)[label_valid_indices] - 1
                preds = np.concatenate((preds, valid_pred.view(-1).cpu()), axis=0)
                targets = np.concatenate((targets, valid_label.view(-1).cpu()), axis=0)
        
        pbar.close()
        log = self.valid_metrics.result()
        log['classification_report'] = "\n" + classification_report(targets, preds, target_names=('Non-Forest', 'Forest'))
        return log

    def _log_to_tensorboard(self, log, epoch):
        """
        Log metrics to TensorBoard
        
        :param log: Dictionary containing metrics to log
        :param epoch: Current epoch number
        """
        for key, value in log.items():
            if isinstance(value, (int, float)) and key != 'epoch':
                self.tb_writer.add_scalar(key, value, epoch)
        
        # Log learning rate if available
        if self.lr_scheduler is not None:
            self.tb_writer.add_scalar('learning_rate', 
                                    self.lr_scheduler.get_last_lr()[0], epoch)
    
    def close_tensorboard(self):
        """
        Close TensorBoard writer
        """
        self.tb_writer.close()

    def _save_checkpoint(self, epoch, save_best=False, is_last=False):
        """
        Saving checkpoints with lr_scheduler state

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        :param is_last: if True, save as 'model_last_epoch{}.pth' and clean up old last checkpoints
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        
        # Add lr_scheduler state if available
        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
        
        if is_last:
            # Clean up old model_last_epoch*.pth files
            import glob
            import os
            old_last_files = glob.glob(str(self.checkpoint_dir / 'model_last_epoch*.pth'))
            for old_file in old_last_files:
                try:
                    os.remove(old_file)
                except OSError:
                    pass
            
            # Save new last checkpoint with epoch number
            filename = str(self.checkpoint_dir / 'model_last_epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            self.logger.info("Saving last checkpoint: {} ...".format(filename))
        else:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
            
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints with lr_scheduler state

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        if not 'epoch' in checkpoint:
            self.model.load_state_dict(torch.load(resume_path), strict=False)
        else:
            self.start_epoch = checkpoint['epoch'] + 1
            self.mnt_best = checkpoint['monitor_best']
            # load architecture params from checkpoint.
            if checkpoint['config']['arch'] != self.config['arch']:
                self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                    "checkpoint. This may yield an exception while state_dict is being loaded.")
            self.model.load_state_dict(checkpoint['state_dict'])

            # load optimizer state from checkpoint only when optimizer type is not changed.
            if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
                self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                    "Optimizer parameters not being resumed.")
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            # load lr_scheduler state from checkpoint if available
            if 'lr_scheduler' in checkpoint and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                self.logger.info("Learning rate scheduler state loaded from checkpoint.")

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
