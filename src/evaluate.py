import torch
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate
from src.trainer.metric import seqKGE
from src.utils import consistent_split
from captum.attr import IntegratedGradients


def evaluate(model, test_loader, metric_ftns):
    y_pred, y_true, basin = predict_test(model, test_loader)

    metrics = []
    for metric_ftn in metric_ftns:
        metrics.append(metric_ftn(y_pred, y_true, basin).item())

    return metrics


def basin_evaluate(checkpoint, save_csv=True):
    model, cfg = instantiate_model(model)
    dataset = instantiate(cfg.dataset)
    datasplit = instantiate(cfg.datasplit, dataset)
    consistent_split(cfg, dataset, datasplit)
    test_loader = next(iter(datasplit))[1]
    y_pred, y_true, basin = predict_test(model, test_loader)

    df = pd.DataFrame()
    df['STAID'] = dataset.gages['STAID']
    df[['KGE', 'r', 'alpha', 'beta']] = seqKGE(y_pred, y_true, basin).cpu().numpy().T
    if save_csv:
        df.to_csv(Path(checkpoint).parents[1] / 'basin-results.csv', index=None)
    return df


def importance(checkpoint):
    model, cfg = instantiate_model(checkpoint)
    device = next(model.parameters()).device
    ig = IntegratedGradients(model)
    dataset = instantiate(cfg.dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=500,
                                         pin_memory=cfg.datasplit.pin_memory,
                                         num_workers=cfg.datasplit.num_workers)

    # cudnn cannot return gradients in eval mode
    if device.type == 'cuda':
        model.train()
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
            elif isinstance(module, torch.nn.LSTM):
                module.dropout = 0

    # calculate integrated gradients
    total = 0
    for input, _, _ in loader:
        if isinstance(input, torch.Tensor):
            input = tuple([input.to(device)])
        else:
            input = tuple(i.to(device) for i in input)
        attrs = [abs(attr) for attr in ig.attribute(input)]
        total += torch.concat([attr.reshape(-1, attr.shape[-1]).sum(axis=0)
                               for attr in attrs])

    # create dataframe header
    station = Path(cfg.dataset.X_dir) / (dataset.gages['STAID'][0] + '.csv')
    sta = (pd.Index([]) if 'S_dir' not in cfg.dataset else
           pd.read_csv(cfg.dataset.S_dir, nrows=0).columns[-dataset.S.shape[1]:])
    dyn = 'dyn_' + pd.read_csv(station, nrows=0).columns
    importance = (total / torch.abs(total).sum() * 100).detach().cpu().numpy()

    return pd.DataFrame(importance.reshape(1, -1), columns=sta.append(dyn))


def predict_test(model, test_loader):
    # cuda or cpu
    device = next(model.parameters()).device

    # simulate
    with torch.no_grad():
        outputs = []
        targets = []
        labels = []
        for input, target, label in test_loader:
            target = target.to(device)
            if isinstance(input, torch.Tensor):
                input = tuple([input.to(device)])
            else:
                input = tuple(i.to(device) for i in input)
            outputs.append(model(*input))
            targets.append(target)
            labels.append(label)

        # concatenate in dim 0
        y_pred = torch.cat(outputs)
        y_true = torch.cat(targets)
        basin = torch.cat(labels)

    return y_pred, y_true, basin


def instantiate_model(checkpoint, device=None):
    pth = torch.load(checkpoint)
    pth['config']['work_dir'] = '.'
    cfg = OmegaConf.create(pth['config'])

    # restore network architecture
    model = instantiate(cfg.model)

    # load trained weights
    if cfg['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(pth['state_dict'])

    # set device
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if device is None else device)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    return model, cfg
