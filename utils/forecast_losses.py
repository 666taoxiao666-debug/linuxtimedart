import torch
import torch.nn as nn
import torch.nn.functional as F


class ForecastLoss(nn.Module):
    """Element-wise forecasting loss with optional horizon/power weighting.

    Defaults reproduce ordinary MSE. ``MIXED`` combines MSE and MAE so large
    errors remain strongly penalised without making the objective entirely
    quadratic. Horizon weights grow linearly from 1 to
    ``horizon_weight_end``. Power weighting uses the standardized target and
    increases emphasis smoothly for high-power observations.
    """

    def __init__(
        self,
        loss_name="MSE",
        huber_delta=1.0,
        mix_mse_weight=0.8,
        horizon_weight_end=1.0,
        power_weight_alpha=0.0,
    ):
        super().__init__()
        self.loss_name = str(loss_name).upper()
        self.huber_delta = float(huber_delta)
        self.mix_mse_weight = float(mix_mse_weight)
        self.horizon_weight_end = float(horizon_weight_end)
        self.power_weight_alpha = float(power_weight_alpha)

        if self.loss_name not in {"MSE", "MAE", "HUBER", "SMOOTHL1", "MIXED"}:
            raise ValueError("loss_name must be MSE, MAE, Huber, or MIXED")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if not 0.0 <= self.mix_mse_weight <= 1.0:
            raise ValueError("mix_mse_weight must be in [0, 1]")
        if self.horizon_weight_end <= 0:
            raise ValueError("horizon_weight_end must be positive")
        if self.power_weight_alpha < 0:
            raise ValueError("power_weight_alpha cannot be negative")

    def _point_loss(self, prediction, target):
        error = prediction - target

        if self.loss_name == "MSE":
            return error.square()

        if self.loss_name == "MAE":
            return error.abs()

        if self.loss_name in {"HUBER", "SMOOTHL1"}:
            return F.huber_loss(
                prediction,
                target,
                delta=self.huber_delta,
                reduction="none",
            )

        mse = error.square()
        mae = error.abs()

        return self.mix_mse_weight * mse + (1.0 - self.mix_mse_weight) * mae

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction/target shapes differ: "
                f"{prediction.shape} != {target.shape}"
            )

        point_loss = self._point_loss(prediction, target)
        weights = torch.ones_like(point_loss)

        # Forecast tensors are [batch, horizon, channel].
        if point_loss.ndim >= 2 and self.horizon_weight_end != 1.0:
            horizon = point_loss.shape[1]

            horizon_weights = torch.linspace(
                1.0,
                self.horizon_weight_end,
                steps=horizon,
                device=point_loss.device,
                dtype=point_loss.dtype,
            )

            shape = [1] * point_loss.ndim
            shape[1] = horizon
            weights = weights * horizon_weights.view(*shape)

        if self.power_weight_alpha > 0:
            # SDWPF target values are standardized. sigmoid(target) is bounded,
            # monotonic, and avoids unstable weights near zero power.
            power_weights = (
                1.0
                + self.power_weight_alpha
                * torch.sigmoid(target.detach())
            )
            weights = weights * power_weights

        # Normalisation keeps the overall learning-rate scale comparable across
        # weighted and unweighted experiments.
        return (
            (point_loss * weights).sum()
            / weights.sum().clamp_min(1e-12)
        )