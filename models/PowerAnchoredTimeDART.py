"""Opt-in power-feature identity bypass; original TimeDART remains unchanged."""
import torch
from torch import nn

from models.TimeDART import PromptGuidedModel


class PowerAnchoredMixer(nn.Module):
    """Preserve the pretrained power token; learn a zero-initialized correction.

    Reuse the original mixer object, parameters and RNG state. Only its final
    correction projections are zero initialized. No parameters are added.
    """
    def __init__(self, correction):
        super().__init__()
        self.correction = correction
        nn.init.zeros_(correction.proj[-1].weight)
        nn.init.zeros_(correction.proj[-1].bias)
        if correction.context_proj is not None:
            nn.init.zeros_(correction.context_proj.weight)
            nn.init.zeros_(correction.context_proj.bias)

    def forward(self, x, context=None):
        if x.ndim != 4 or x.shape[1] != self.correction.num_features:
            raise ValueError("Expected [batch, channels, patches, hidden] tokens")
        # The registered Kelmarsh feature order has power in the final slot.
        return x[:, -1:, :, :] + self.correction(x, context=context)


class Model(PromptGuidedModel):
    def __init__(self, args):
        super().__init__(args)
        if args.task_name == "finetune":
            if (getattr(args, "features", None) != "MS" or not self.mix_channels or
                    not self.feature_columns or self.feature_columns[-1] != "power"):
                raise ValueError("Power anchoring requires registered MS power-last features")
            if self.channel_mixer is None:
                raise ValueError("A multivariate mixer is required")
            self.channel_mixer = PowerAnchoredMixer(self.channel_mixer)
