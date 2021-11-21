from torch.utils.data import Subset, DataLoader


def gen_loader(dataset, splits, **loader_args):
    for train_idx, test_idx in splits:
        train_loader = DataLoader(Subset(dataset, train_idx),
                                  **loader_args, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx),
                                 **loader_args, shuffle=False)
        yield train_loader, test_loader


class BaseSplit:
    def __init__(self, batch_size=32, num_workers=0, pin_memory=False) -> None:
        super().__init__()
        self.loader_args = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }

    def __iter__(self):
        return gen_loader(self.dataset, self.splits, **self.loader_args)
