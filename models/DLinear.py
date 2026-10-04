"""DLinear baseline using the repository's common forecast interface."""

import torch
from torch import nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("moving_avg must be a positive odd integer")
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x):
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        padded = torch.cat([front, x, end], dim=1)
        return self.pool(padded.permute(0, 2, 1)).permute(0, 2, 1)


class Model(nn.Module):
    """Series decomposition followed by linear seasonal/trend projections."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.downstream_task = getattr(configs, "downstream_task", None)
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.individual = bool(getattr(configs, "individual", 0))
        self.decomposition = MovingAverage(getattr(configs, "moving_avg", 25))
        if self.individual:
            self.seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)]
            )
            self.trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)]
            )
        else:
            self.seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.trend = nn.Linear(self.seq_len, self.pred_len)

    def _is_forecast_task(self):
        return self.task_name in {"long_term_forecast", "short_term_forecast"} or (
            self.task_name == "finetune" and self.downstream_task == "forecast"
        )

    def forecast(self, x):
        trend = self.decomposition(x)
        seasonal = x - trend
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)
        if self.individual:
            seasonal_out = torch.stack(
                [self.seasonal[i](seasonal[:, i, :]) for i in range(self.channels)],
                dim=1,
            )
            trend_out = torch.stack(
                [self.trend[i](trend[:, i, :]) for i in range(self.channels)],
                dim=1,
            )
        else:
            seasonal_out = self.seasonal(seasonal)
            trend_out = self.trend(trend)
        return (seasonal_out + trend_out).permute(0, 2, 1)

    def forward(self, x_enc, *args, **kwargs):
        if not self._is_forecast_task():
            raise ValueError(
                "DLinear currently supports forecasting only; received "
                f"task_name={self.task_name!r}, downstream_task={self.downstream_task!r}"
            )
        return self.forecast(x_enc)[:, -self.pred_len :, :]
