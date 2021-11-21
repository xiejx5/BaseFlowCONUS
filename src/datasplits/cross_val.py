import numpy as np
from src.datasplits.base_split import BaseSplit
from src.datasplits.train_test import TrainTestSplit
from sklearn.model_selection import StratifiedKFold


class CrossValSplit(BaseSplit):
    def __init__(self, dataset, batch_size=64, num_workers=0, pin_memory=False,
                 test_size=0.2, fold=4, train_loader=None) -> None:
        super().__init__(batch_size, num_workers, pin_memory)

        if train_loader is None:
            train_loader = next(iter(TrainTestSplit(dataset,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    pin_memory=pin_memory,
                                                    test_size=test_size)))[0]

        # k fold cross validation using stratified sampling
        train_dataset = train_loader.dataset
        skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=1)

        self.dataset = train_dataset
        self.splits = list(skf.split(np.zeros(len(train_dataset)),
                                     train_dataset[:][-1]))
