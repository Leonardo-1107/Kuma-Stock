import torch
import torch.nn as nn
import torch.nn.functional as F


# information coefficient loss
class ICLoss(nn.Module):
    def __init__(self):
        super(ICLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        
        pred_mean = torch.mean(y_pred)
        true_mean = torch.mean(y_true)
        
        numerator = torch.sum((y_pred - pred_mean) * (y_true - true_mean))
        denominator = torch.sqrt(torch.sum((y_pred - pred_mean)**2) * torch.sum((y_true - true_mean)**2))
        
        ic = numerator / denominator
        
        # To make it a loss function, we want to maximize IC (which means minimizing -IC)
        return -ic


# scale-independent loss function
class MeanPercentageErrorLoss(nn.Module):
    def __init__(self):
        super(MeanPercentageErrorLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Calculate the Mean Percentage Error (MPE) loss.

        Returns:
            torch.Tensor: Mean Percentage Error loss.
        """
        y_true = y_true.float()
        y_pred = y_pred.float()

        # Avoid division by zero
        non_zero_mask = (y_true != 0)
        mpe = torch.mean(torch.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        
        return mpe

# mean absolute error of naive forecast
class MeanAbsoluteScaledErrorLoss(nn.Module):
    def __init__(self):
        super(MeanAbsoluteScaledErrorLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        MASE is useful for understanding model performance in time series forecasting

        Returns:
            torch.Tensor: Mean Absolute Scaled Error loss.
        """
        y_true = y_true.float()
        y_pred = y_pred.float()

        abs_errors = torch.abs(y_true - y_pred)
        # Compute naive forecast errors
        naive_forecast_errors = torch.abs(y_true[:, 1:] - y_true[:, :-1])  # shape (batch_size, L-1)
        
        scaling_factor = torch.mean(naive_forecast_errors, dim=1).mean()
        mase = torch.mean(abs_errors) / scaling_factor if scaling_factor != 0 else torch.tensor(float('inf'))
        
        return mase