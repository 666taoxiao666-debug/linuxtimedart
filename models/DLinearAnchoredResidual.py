"""v26: frozen DLinear forecast anchor plus a zero-initialised TSMixerBasicSmall residual branch.

Standardized power space, direct 4-step forecast:

    y_h(x) = b_h(x) + r_h(x)

  b  frozen DLinear (the same-fold v20 seed-2024 checkpoint, loaded with ``load_anchor_state``); power slot only;
     always in eval mode, ``requires_grad=False`` and never handed to an optimiser.
  r  the unchanged v25 ``TSMixerBasicSmall`` (2 blocks, ff 32, dropout 0.1, LayerNorm over (24, 7)) whose final
     ``Linear(24 -> 4)`` weight AND bias are set to 0. Nothing else is zeroed: there is no gate or scale, so the
     head receives a non-zero gradient at the first step and the mixer blocks from the second step on.

For finite inputs r == 0 exactly at construction, so the initial forecast is bitwise the DLinear forecast.

``input_mode='power_only'`` sets the auxiliary standardized channels 0..5 to exactly 0 before BOTH b and r. The
DLinear power output is channel independent (individual=0, per-channel moving average and shared linear maps), so
b is identical across arms; only the residual sees a different input.

RNG: the residual branch is constructed first and consumes exactly the draws of ``TSMixerBasicSmall`` (its blocks
start from the same tensors as the v25 model for the same seed); the anchor is constructed inside
``torch.random.fork_rng`` so it consumes no global draws before being overwritten by the frozen checkpoint.
This module adds no future covariate, label, turbine embedding, prompt, Wiki or persistence input.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from models.DLinear import Model as DLinearModel
from models.TSMixerBasicSmall import INPUT_MODES, Model as TSMixerModel, power_only_input

ANCHOR_DEFAULTS = {"task_name": "finetune", "downstream_task": "forecast", "individual": 0, "moving_avg": 25}


class Model(nn.Module):
    """[B, L, C] standardized history -> [B, H, 1] standardized power = frozen DLinear + learned residual."""

    def __init__(self, *, seq_len: int = 24, channels: int = 7, pred_len: int = 4, blocks: int = 2,
                 ff_dim: int = 32, dropout: float = 0.1, norm_eps: float = 1e-3, target_slot: int = 6,
                 input_mode: str = "full", anchor_args: dict | None = None):
        super().__init__()
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}")
        if not 0 <= int(target_slot) < int(channels):
            raise ValueError("target_slot outside the channel axis")
        self.seq_len, self.channels, self.pred_len = int(seq_len), int(channels), int(pred_len)
        self.target_slot, self.input_mode = int(target_slot), input_mode
        # Masking happens once, here, for both branches; the residual itself is built in 'full' mode.
        self.residual = TSMixerModel(seq_len=self.seq_len, channels=self.channels, pred_len=self.pred_len,
                                     blocks=int(blocks), ff_dim=int(ff_dim), dropout=float(dropout),
                                     norm_eps=float(norm_eps), target_slot=self.target_slot, input_mode="full")
        nn.init.zeros_(self.residual.head.weight)
        nn.init.zeros_(self.residual.head.bias)
        args = {**ANCHOR_DEFAULTS, "seq_len": self.seq_len, "pred_len": self.pred_len, "enc_in": self.channels,
                **(anchor_args or {})}
        if (int(args["seq_len"]), int(args["pred_len"]), int(args["enc_in"])) != (self.seq_len, self.pred_len, self.channels):
            raise ValueError("Anchor and residual must share seq_len, pred_len and channels")
        if int(args.get("individual", 0)) != 0:
            raise ValueError("The anchor must be the channel-independent shared DLinear (individual=0)")
        with torch.random.fork_rng(devices=[]):
            self.anchor = DLinearModel(SimpleNamespace(**args))
        self.anchor_args = args
        self._freeze_anchor()

    # ------------------------------------------------------------------ frozen anchor
    def _freeze_anchor(self) -> None:
        self.anchor.requires_grad_(False)
        self.anchor.eval()

    def load_anchor_state(self, state) -> None:
        """Load the frozen DLinear checkpoint (strict) and re-assert that it is frozen."""
        self.anchor.load_state_dict(state, strict=True)
        self._freeze_anchor()

    def train(self, mode: bool = True):
        super().train(mode)
        self.anchor.eval()  # the frozen executor never enters training mode
        return self

    # ------------------------------------------------------------------ forward
    def _inputs(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or tuple(history.shape[1:]) != (self.seq_len, self.channels):
            raise ValueError(f"Expected [B, {self.seq_len}, {self.channels}] history, got {tuple(history.shape)}")
        return power_only_input(history, self.target_slot) if self.input_mode == "power_only" else history

    def _anchor(self, x: torch.Tensor) -> torch.Tensor:
        return self.anchor(x)[..., self.target_slot:self.target_slot + 1]

    def anchor_forecast(self, history: torch.Tensor) -> torch.Tensor:
        """b(x): the frozen DLinear power forecast [B, H, 1] (standardized)."""
        return self._anchor(self._inputs(history))

    def correction(self, history: torch.Tensor) -> torch.Tensor:
        """r(x): the learned residual [B, H, 1] (standardized); exactly 0 at construction."""
        return self.residual(self._inputs(history))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        x = self._inputs(history)
        return self._anchor(x) + self.residual(x)
