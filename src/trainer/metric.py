import torch
import numpy as np


class NSE(torch.nn.Module):
    def __init__(self):
        super(NSE, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor, basin: torch.Tensor):
        SS_res = torch.sum(torch.square(target - output))
        SS_tot = torch.sum(torch.square(target - torch.mean(target)))
        return (1 - SS_res / (SS_tot + 1e-10))


class MeanNSE(torch.nn.Module):
    def __init__(self):
        super(MeanNSE, self).__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, basin: torch.Tensor):
        y_pred, y_true = torch.flatten(y_pred), torch.flatten(y_true)
        _, idx, counts = torch.unique(basin.to(y_true.device), sorted=True,
                                      return_inverse=True, return_counts=True)
        y_mean = torch.bincount(idx, weights=y_true) / counts
        SS_res = torch.bincount(idx, weights=torch.square(y_true - y_pred))
        SS_tot = torch.bincount(idx, torch.square(y_true - y_mean[idx]))
        return torch.mean(1 - SS_res / (SS_tot + 1e-10))


class MedianNSE(torch.nn.Module):
    def __init__(self):
        super(MedianNSE, self).__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, basin: torch.Tensor):
        y_pred, y_true = torch.flatten(y_pred), torch.flatten(y_true)
        _, idx, counts = torch.unique(basin.to(y_true.device), sorted=True,
                                      return_inverse=True, return_counts=True)
        y_mean = torch.bincount(idx, weights=y_true) / counts
        SS_res = torch.bincount(idx, weights=torch.square(y_true - y_pred))
        SS_tot = torch.bincount(idx, torch.square(y_true - y_mean[idx]))
        return torch.median(1 - SS_res / (SS_tot + 1e-10))


def singleNSE(output: torch.Tensor, target: torch.Tensor):
    SS_res = torch.sum(torch.square(target - output))
    SS_tot = torch.sum(torch.square(target - torch.mean(target)))
    return (1 - SS_res / (SS_tot + 1e-10))


def seqNSE(y_pred, y_true, basin):
    unique = torch.unique(basin, sorted=True)
    NSEs = []
    for u in unique:
        NSEs.append(singleNSE(y_pred[basin == u], y_true[basin == u]).item())
    NSEs = np.array(NSEs)
    return NSEs


def seqNSE_torch(y_pred: torch.Tensor, y_true: torch.Tensor, basin: torch.Tensor):
    y_pred, y_true = torch.flatten(y_pred), torch.flatten(y_true)
    _, idx, counts = torch.unique(basin.to(y_true.device), sorted=True,
                                  return_inverse=True, return_counts=True)
    y_mean = torch.bincount(idx, weights=y_true) / counts
    SS_res = torch.bincount(idx, weights=torch.square(y_true - y_pred))
    SS_tot = torch.bincount(idx, torch.square(y_true - y_mean[idx]))
    return (1 - SS_res / (SS_tot + 1e-10))


def seqLogNSE_torch(y_pred: torch.Tensor, y_true: torch.Tensor, basin: torch.Tensor):
    y_pred, y_true = torch.log(torch.flatten(y_pred) + 1), torch.log(torch.flatten(y_true) + 1)
    _, idx, counts = torch.unique(basin.to(y_true.device), sorted=True,
                                  return_inverse=True, return_counts=True)
    y_mean = torch.bincount(idx, weights=y_true) / counts
    SS_res = torch.bincount(idx, weights=torch.square(y_true - y_pred))
    SS_tot = torch.bincount(idx, torch.square(y_true - y_mean[idx]))
    return (1 - SS_res / (SS_tot + 1e-10))


def singleMARE(output: torch.Tensor, target: torch.Tensor):
    return torch.sum(torch.abs(target - output)) / (torch.sum(target) + 1e-10)


def seqMARE(y_pred, y_true, basin):
    unique = torch.unique(basin, sorted=True)
    NSEs = []
    for u in unique:
        NSEs.append(singleMARE(y_pred[basin == u], y_true[basin == u]))
    return torch.hstack(NSEs)


def singleBias(output: torch.Tensor, target: torch.Tensor):
    return torch.sum(target - output) / (torch.sum(target) + 1e-10)


def seqBias(y_pred, y_true, basin):
    unique = torch.unique(basin, sorted=True)
    NSEs = []
    for u in unique:
        NSEs.append(singleBias(y_pred[basin == u], y_true[basin == u]))
    return torch.hstack(NSEs)


def seqKGE(y_pred, y_true, basin):
    unique = torch.unique(basin, sorted=True)
    KGEs = []
    for u in unique:
        KGEs.append(singleKGE(y_pred[basin == u], y_true[basin == u]))
    return torch.hstack(KGEs)


def singleKGE(simulations, evaluation):
    """Original Kling-Gupta Efficiency (KGE) and its three components
    (r, α, β) as per `Gupta et al., 2009
    <https://doi.org/10.1016/j.jhydrol.2009.08.003>`_.
    Note, all four values KGE, r, α, β are returned, in this order.
    :Calculation Details:
        .. math::
           E_{\\text{KGE}} = 1 - \\sqrt{[r - 1]^2 + [\\alpha - 1]^2
           + [\\beta - 1]^2}
        .. math::
           r = \\frac{\\text{cov}(e, s)}{\\sigma({e}) \\cdot \\sigma(s)}
        .. math::
           \\alpha = \\frac{\\sigma(s)}{\\sigma(e)}
        .. math::
           \\beta = \\frac{\\mu(s)}{\\mu(e)}
        where *e* is the *evaluation* series, *s* is (one of) the
        *simulations* series, *cov* is the covariance, *σ* is the
        standard deviation, and *μ* is the arithmetic mean.
    """
    # calculate error in timing and dynamics r
    # (Pearson's correlation coefficient)
    sim_mean = torch.mean(simulations, axis=0, dtype=torch.float64)
    obs_mean = torch.mean(evaluation, dtype=torch.float64)

    r_num = torch.sum((simulations - sim_mean) * (evaluation - obs_mean),
                      axis=0, dtype=torch.float64)
    r_den = torch.sqrt(torch.sum((simulations - sim_mean) ** 2,
                                 axis=0, dtype=torch.float64)
                       * torch.sum((evaluation - obs_mean) ** 2,
                                   dtype=torch.float64))
    r = r_num / (r_den + 1e-10)
    # calculate error in spread of flow alpha
    alpha = torch.std(simulations) / (torch.std(evaluation) + 1e-10)
    # calculate error in volume beta (bias of mean discharge)
    beta = (torch.sum(simulations, axis=0, dtype=torch.float64)
            / (torch.sum(evaluation, dtype=torch.float64) + 1e-10))
    # calculate the Kling-Gupta Efficiency KGE
    kge_ = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return torch.vstack((kge_, r, alpha, beta))
