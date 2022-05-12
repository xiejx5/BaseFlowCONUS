import torch
import logging
from numpy import inf
from pathlib import Path
from shutil import copyfile
from src.utils import write_conf
from src.utils.logger import EpochMetrics
from abc import abstractmethod, ABCMeta
from hydra.utils import to_absolute_path, get_original_cwd


logger = logging.getLogger('base-trainer')


class BaseTrainer(metaclass=ABCMeta):
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, round):
        self.config = config

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.log_step = cfg_trainer['logging_step']

        # setup metric monitoring for monitoring model performance and saving best-checkpoint
        self.monitor = cfg_trainer.get('monitor', 'off')

        metric_names = ['loss'] + [type(met).__name__ for met in self.metric_ftns]
        self.ep_metrics = EpochMetrics(metric_names, phases=('train', 'valid'), monitoring=self.monitor)
        self.result_dir = f'epoch-results-{round}.csv' if round else f'epoch-results.csv'

        self.checkpt_top_k = cfg_trainer.get('save_topk', -1)
        self.early_stop = cfg_trainer.get('early_stop', inf)
        self.early_stop = inf if self.early_stop is None else self.early_stop

        write_conf(self.config, 'config.yaml')

        self.start_epoch = 1
        self.checkpt_dir = Path(self.config.save_dir).absolute()
        self.checkpt_dir.mkdir(exist_ok=True)

        # setup visualization writer instance
        self.writer = None
        if cfg_trainer['tensorboard']:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = Path(self.config.log_dir).absolute()
            log_dir.mkdir(exist_ok=True)
            self.writer = SummaryWriter(log_dir)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            self.ep_metrics.update(epoch, result)

            # print result metrics of this epoch
            max_line_width = max(len(line) for line in str(self.ep_metrics).splitlines())
            # divider ---
            logger.info('-' * max_line_width)
            logger.info(self.ep_metrics.latest().to_string(float_format=lambda x: f'{x:.4f}') + '\n')

            # check if model performance improved or not, for early stopping and topk saving
            is_best = False
            improved = self.ep_metrics.is_improved()
            if improved:
                not_improved_count = 0
                is_best = True
            else:
                not_improved_count += 1

            if not_improved_count > self.early_stop:
                logger.info("Validation performance didn\'t improve for {} epochs. "
                            "Training stops.".format(self.early_stop))
                break

            using_topk_save = self.checkpt_top_k > 0
            self._save_checkpoint(epoch, save_best=is_best, save_latest=using_topk_save)

            # keep top-k checkpoints only, using monitoring metrics
            if using_topk_save:
                self.ep_metrics.keep_topk_checkpt(self.checkpt_dir, self.checkpt_top_k)

            self.ep_metrics.to_csv(self.result_dir)

            # divider ===
            logger.info('=' * max_line_width)

        # Return metric score for hyperparameter optimization
        return self.ep_metrics.latest_metric()

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            logger.warning("Warning: There\'s no GPU available on this machine,"
                           "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                           "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False, save_latest=True):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, save a copy of current checkpoint file as 'model_best.pth'
        :param save_latest: if True, save a copy of current checkpoint file as 'model_latest.pth'
        """
        state = {
            'model': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch_metrics': self.ep_metrics,
            'config': self.config
        }
        abs_path = self.checkpt_dir / 'checkpoint-epoch{}.pth'.format(epoch)
        torch.save(state, abs_path)

        rel_path = (abs_path.relative_to(get_original_cwd())
                    if abs_path.is_relative_to(get_original_cwd())
                    else abs_path)
        logger.info(f"Model checkpoint saved at: \n    {rel_path}")
        if save_latest:
            latest_path = self.checkpt_dir / 'model_latest.pth'
            copyfile(abs_path, latest_path)
        if save_best:
            best_path = self.checkpt_dir / 'model_best.pth'
            copyfile(abs_path, best_path)
            rel_best_path = (best_path.relative_to(Path.cwd())
                             if best_path.is_relative_to(Path.cwd())
                             else best_path)
            logger.info(f"Renewing best checkpoint: \n    ...\\{rel_best_path}")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(to_absolute_path(resume_path))
        logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        # TODO: support overriding monitor-metric config
        self.ep_metrics = checkpoint['epoch_metrics']

        # load architecture params from checkpoint.
        if checkpoint['config']['model'] != self.config['model']:
            logger.warning("Warning: Architecture configuration given in config file is different from that of "
                           "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['trainer']['optimizer'] != self.config['trainer']['optimizer']:
            logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                           "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
