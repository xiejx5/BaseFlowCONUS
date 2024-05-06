import os
import logging
import pandas as pd
from itertools import product


logger = logging.getLogger('logger')


class BatchMetrics:
    def __init__(self, *keys, postfix='', writer=None):
        self.writer = writer
        self.postfix = postfix
        if postfix:
            keys = [k + postfix for k in keys]
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1, step=None):
        if self.postfix:
            key = key + self.postfix
        if self.writer is not None and isinstance(step, int):
            self.writer.add_scalar(key, value, step)
        self._data.loc[key, 'total'] += value * n
        self._data.loc[key, 'counts'] += n
        self._data.loc[key, 'average'] = self._data.loc[key, 'total'] / self._data.loc[key, 'counts']

    def avg(self, key):
        if self.postfix:
            key = key + self.postfix
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class EpochMetrics:
    def __init__(self, metric_names, phases=('train', 'valid'), monitoring='off'):
        # setup pandas DataFrame with hierarchical columns
        columns = tuple(product(metric_names, phases))
        self._data = pd.DataFrame(columns=columns)  # TODO: add epoch duration
        self.monitor_mode, self.monitor_metric = self._parse_monitoring_mode(monitoring)
        self.topk_idx = []

    def minimizing_metric(self, idx):
        if self.monitor_mode == 'off':
            return 0
        try:
            metric = self._data[self.monitor_metric].loc[idx]
        except KeyError:
            logger.warning("Warning: Metric '{}' is not found. "
                           "Model performance monitoring is disabled.".format(self.monitor_metric))
            self.monitor_mode = 'off'
            return 0
        if self.monitor_mode == 'min':
            return metric
        else:
            return - metric

    def _parse_monitoring_mode(self, monitor_mode):
        if monitor_mode == 'off':
            return 'off', None
        else:
            monitor_mode, monitor_metric = monitor_mode.split()
            monitor_metric = tuple(monitor_metric.split('/'))
            assert monitor_mode in ['min', 'max']
        return monitor_mode, monitor_metric

    def is_improved(self):
        if self.monitor_mode == 'off':
            return True

        last_epoch = self._data.index[-1]
        best_epoch = self.topk_idx[0]
        return last_epoch == best_epoch

    def keep_topk_checkpt(self, checkpt_dir, k=3):
        """
        Keep top-k checkpoints by deleting k+1'th best epoch index from dataframe for every epoch.
        """
        if len(self.topk_idx) > k and self.monitor_mode != 'off':
            last_epoch = self._data.index[-1]
            self.topk_idx = self.topk_idx[:(k + 1)]
            if last_epoch not in self.topk_idx:
                to_delete = last_epoch
            else:
                to_delete = self.topk_idx[-1]

            # delete checkpoint having out-of topk metric
            filename = str(checkpt_dir / 'checkpoint-epoch{}.pth'.format(to_delete.split('-')[1]))
            os.remove(filename)

    def update(self, epoch, result):
        epoch_idx = f'epoch-{epoch}'
        self._data.loc[epoch_idx] = {tuple(k.split('/')): v for k, v in result.items()}

        self.topk_idx.append(epoch_idx)
        self.topk_idx = sorted(self.topk_idx, key=self.minimizing_metric)

    def latest(self):
        return self._data[-1:]

    def latest_metric(self, best=False):
        if best:
            idx = self.topk_idx[0]
        else:
            idx = self._data.index[-1]

        metric = self.minimizing_metric(idx)
        if self.monitor_mode == 'min':
            return metric
        else:
            return - metric

    def to_csv(self, save_path=None):
        self._data.to_csv(save_path)

    def __str__(self):
        return str(self._data)
