import logging
import torch
from .base import BaseTrainer
from src.utils import inf_loop
from src.evaluate import evaluate
from src.utils.logger import BatchMetrics


logger = logging.getLogger('trainer')


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, round=0):
        super().__init__(model, criterion, metric_ftns, optimizer, config, round)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler

        self.train_metrics = BatchMetrics('loss',
                                          *[type(m).__name__ for m in self.metric_ftns],
                                          postfix='/train',
                                          writer=self.writer)
        self.valid_metrics = BatchMetrics('loss',
                                          *[type(m).__name__ for m in self.metric_ftns],
                                          postfix='/valid',
                                          writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target, basin) in enumerate(self.data_loader):
            target = target.to(self.device)
            if isinstance(data, torch.Tensor):
                data = data.to(self.device)
            else:
                data = tuple(i.to(self.device) for i in data)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target, basin)
            loss.backward()
            self.optimizer.step()

            # step = (epoch - 1) * self.len_epoch + batch_idx
            self.train_metrics.update('loss', loss.item())

            if batch_idx % self.log_step == 0:
                logger.info('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if (self.valid_data_loader is not None and
                len(self.valid_data_loader) > 0):
            val_log = self._valid_epoch(epoch)
            log.update(**val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # add result metrics on entire epoch to tensorboard
        if self.writer is not None:
            for k, v in log.items():
                self.writer.add_scalar(k, v, epoch)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.valid_metrics.reset()
        metrics = evaluate(self.model, self.valid_data_loader, [self.criterion, *self.metric_ftns])
        self.valid_metrics.update('loss', metrics[0])
        for metrict_ftn, metric in zip(self.metric_ftns, metrics[1:]):
            self.valid_metrics.update(type(metrict_ftn).__name__, metric)

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        try:
            # epoch-based training
            total = len(self.data_loader.dataset)
            current = batch_idx * self.data_loader.batch_size
        except AttributeError:
            # iteration-based training
            total = self.len_epoch
            current = batch_idx
        return base.format(current, total, 100.0 * current / total)
