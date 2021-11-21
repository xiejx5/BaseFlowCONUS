import numpy as np
from src.datasplits.base_split import BaseSplit
from sklearn.model_selection import train_test_split


def multi_arange(starts, stops, steps=1):
    lens = ((stops - starts) + steps - np.sign(steps)) // steps
    if isinstance(steps, np.ndarray):
        res = np.repeat(steps, lens)
    else:
        res = np.full(lens.sum(), steps)
    ends = (lens - 1) * steps + starts
    res[0] = starts[0]
    res[lens[:-1].cumsum()] = starts[1:] - ends[:-1]
    return res.cumsum()


def train_test_stratify(labels, test_size):
    bin_counts = np.bincount(labels)
    counts = bin_counts[np.nonzero(bin_counts)[0]]
    end = np.cumsum(counts)
    test_counts = np.rint(counts * test_size).astype('int')
    split = end - test_counts

    test_idx = multi_arange(split, end).astype(int)
    train_idx = np.setdiff1d(np.arange(len(labels)), test_idx)

    return train_idx, test_idx


class TrainTestSplit(BaseSplit):
    def __init__(self, dataset, batch_size=32, num_workers=0,
                 pin_memory=False, test_size=0.2) -> None:
        super().__init__(batch_size, num_workers, pin_memory)

        # Split dataset into train and test
        if test_size:
            # split = train_test_stratify(dataset.basin, test_size)
            split = train_test_split(
                np.arange(len(dataset.basin)),
                test_size=test_size,
                random_state=1,
                shuffle=True,
                stratify=dataset.basin)
        else:
            split = (np.arange(len(dataset)), np.array([]))

        self.dataset = dataset
        self.splits = [split]
