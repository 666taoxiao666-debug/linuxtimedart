"""Opt-in gradient-mask ablation of the power-anchored numerical model."""
from models.PowerAnchoredTimeDART import Model as PowerAnchoredModel


class Model(PowerAnchoredModel):
    """Keep pretrained representations fixed; fit only forecast-side parameters.

    This changes gradient eligibility, not architecture, forward computation,
    dropout mode, initialization, parameter count, or random-number consumption.
    """
    TRAINABLE_PREFIXES = ("head.", "channel_mixer.", "residual_gate")

    def __init__(self, args):
        super().__init__(args)
        if args.task_name != "finetune":
            raise ValueError("Frozen transfer reuses a matched pretrained checkpoint")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(self.TRAINABLE_PREFIXES))

    def trainability_audit(self):
        named = list(self.named_parameters())
        return {
            "policy": "freeze pretrained representation; train forecast-side parameters only",
            "total_parameters": sum(p.numel() for _, p in named),
            "trainable_parameters": sum(p.numel() for _, p in named if p.requires_grad),
            "frozen_parameters": sum(p.numel() for _, p in named if not p.requires_grad),
            "trainable_names": [name for name, p in named if p.requires_grad],
            "frozen_names": [name for name, p in named if not p.requires_grad],
        }
