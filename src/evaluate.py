import torch


def evaluate(model, test_loader, metric_ftns):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        outputs = []
        targets = []
        labels = []
        for input, target, label in test_loader:
            target = target.to(device)
            if isinstance(input, torch.Tensor):
                input = input.to(device)
            else:
                input = tuple(i.to(device) for i in input)
            outputs.append(model(input))
            targets.append(target)
            labels.append(label)

        # concatenate in dim 0
        y_pred = torch.cat(outputs)
        y_true = torch.cat(targets)
        basin = torch.cat(labels)

        metrics = []
        for metric_ftn in metric_ftns:
            metrics.append(metric_ftn(y_pred, y_true, basin).item())

    return metrics


def basin_evaluate(checkpoint, save_csv=True):
    import pandas as pd
    from pathlib import Path
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from src.utils import consistent_split
    from src.trainer.metric import seqKGE

    pth = torch.load(checkpoint)
    pth['config']['work_dir'] = '.'
    cfg = OmegaConf.create(pth['config'])

    dataset = instantiate(cfg.dataset)
    datasplit = instantiate(cfg.datasplit, dataset)
    consistent_split(cfg, dataset, datasplit)
    test_loader = next(iter(datasplit))[1]

    # restore network architecture
    model = instantiate(cfg.model)

    # load trained weights
    if cfg['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(pth['state_dict'])

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # simulate
    with torch.no_grad():
        outputs = []
        targets = []
        labels = []
        for input, target, label in test_loader:
            target = target.to(device)
            if isinstance(input, torch.Tensor):
                input = input.to(device)
            else:
                input = tuple(i.to(device) for i in input)
            outputs.append(model(input))
            targets.append(target)
            labels.append(label)

        # concatenate in dim 0
        y_pred = torch.cat(outputs)
        y_true = torch.cat(targets)
        basin = torch.cat(labels)

    df = pd.DataFrame()
    df['STAID'] = dataset.gages['STAID']
    df['NSE'] = seqKGE(y_pred, y_true, basin).cpu().numpy()
    if save_csv:
        df.to_csv(Path(checkpoint).parents[1] / 'basin-results.csv', index=None)
    return df
