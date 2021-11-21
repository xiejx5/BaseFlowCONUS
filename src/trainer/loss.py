import torch
import numpy as np


class NMSELoss(torch.nn.Module):
    """Calculate (batch-wise) NSE Loss from (Kratzert et al., 2019).
    Each sample i is weighted by 1 / (std_i + eps)^2, where std_i is the standard deviation of the
    discharge from the basin, to which the sample belongs.
    Parameters:
    -----------
    eps : float
        Constant, added to the weight for numerical stability and smoothing, default to 0.1
    """

    def __init__(self, dataset=None, eps=0.001):
        super().__init__()
        counts = np.bincount(dataset.basin)
        y_sum = np.bincount(dataset.basin, weights=dataset.y.flatten())
        y_mean = torch.from_numpy((y_sum / counts).astype('float32'))
        self.weights = 1 / (y_mean + eps) ** 2

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, basin: torch.Tensor):
        """Calculate the batch-wise NSE Loss function.
        Parameters
        ----------
        y_pred : torch.Tensor
            Tensor containing the network prediction.
        y_true : torch.Tensor
            Tensor containing the true discharge values
        basin : torch.Tensor
            numpy ndarray containing the basin index
        Returns
        -------
        torch.Tenor
            The (batch-wise) NSE Loss
        """
        squared_error = (y_pred - y_true)**2
        scaled_loss = self.weights[basin.cpu().numpy()].to(y_pred.device) * squared_error
        return torch.mean(scaled_loss)


class NMSLELoss(torch.nn.Module):
    def __init__(self, dataset=None, eps=0.001):
        super().__init__()
        counts = np.bincount(dataset.basin)
        y_sum = np.bincount(dataset.basin, weights=dataset.y.flatten())
        y_mean = torch.from_numpy((y_sum / counts).astype('float32'))
        self.weights = 1 / (y_mean + eps) ** 2

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, basin: torch.Tensor):
        squared_error = (torch.log(y_pred + 1) - torch.log(y_true + 1))**2
        scaled_loss = self.weights[basin.cpu().numpy()].to(y_pred.device) * squared_error
        return torch.mean(scaled_loss)


class RMSLELoss(torch.nn.Module):
    def __init__(self, dataset=None):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, basin: torch.Tensor):
        return torch.sqrt(self.mse(torch.log(y_pred + 1), torch.log(y_true + 1)))


class MSLELoss(torch.nn.Module):
    def __init__(self, dataset=None):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, basin: torch.Tensor):
        return self.mse(torch.log(y_pred + 1), torch.log(y_true + 1))


class NRMSLELoss(torch.nn.Module):
    def __init__(self, dataset=None, eps=0.001):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        counts = np.bincount(dataset.basin)
        y_sum = np.bincount(dataset.basin, weights=dataset.y.flatten())
        y_mean = torch.from_numpy((y_sum / counts).astype('float32'))
        self.weights = 1 / (torch.log(y_mean + 1) + 1) ** 2

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, basin: torch.Tensor):
        squared_error = (torch.log(y_pred + 1) - torch.log(y_true + 1))**2
        scaled_loss = self.weights[basin.cpu().numpy()].to(y_pred.device) * squared_error
        return torch.sqrt(torch.mean(scaled_loss))


class MSELoss(torch.nn.Module):
    def __init__(self, dataset=None):
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor, basin: torch.Tensor):
        loss_func = torch.nn.MSELoss()
        return loss_func(output, target)
