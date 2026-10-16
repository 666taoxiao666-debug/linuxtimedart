"""Matched early/late cross-channel attention ablation of PowerAnchoredTimeDART."""
from torch import nn

from models.PowerAnchoredTimeDART import Model as AnchoredModel


class PatchChannelAttention(nn.Module):
    """Mix only variables of the same observed patch, never windows or patches."""
    def __init__(self, channels, width, heads, dropout):
        super().__init__()
        self.channels, self.width = channels, width
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        # Exact identity at initialization, preserving the matched pretrained path.
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[0] % self.channels or tokens.shape[-1] != self.width:
            raise ValueError("Expected [batch*channels, historical_patches, width]")
        flat, patches, width = tokens.shape
        batch = flat // self.channels
        local = tokens.reshape(batch, self.channels, patches, width).permute(0, 2, 1, 3)
        local = local.reshape(batch * patches, self.channels, width)
        query = self.norm(local)
        correction, _ = self.attention(query, query, query, need_weights=False)
        mixed = local + correction
        return mixed.reshape(batch, patches, self.channels, width).permute(0, 2, 1, 3).reshape(flat, patches, width)


class _TimingModel(AnchoredModel):
    position = None

    def __init__(self, args):
        super().__init__(args)
        if args.task_name != "finetune" or self.utility_wiki or self.ramp_residual:
            raise ValueError("This matched ablation is numerical, finetune-only, without Wiki/ramp")
        # Register under the existing new-module LR group without renaming any
        # pretrained key. Identical allocation/RNG in both positional arms.
        self.channel_mixer.cross_attention = PatchChannelAttention(
            args.enc_in, args.d_model, args.n_heads, args.dropout)
        if self.position == "early":
            self.encoder.register_forward_pre_hook(self._before_encoder)
        elif self.position == "late":
            self.encoder.register_forward_hook(self._after_encoder)
        else:
            raise ValueError("Explicit early or late timing required")

    def _before_encoder(self, module, inputs):
        return (self.channel_mixer.cross_attention(inputs[0]), *inputs[1:])

    def _after_encoder(self, module, inputs, output):
        return self.channel_mixer.cross_attention(output)


class EarlyModel(_TimingModel):
    position = "early"


class LateModel(_TimingModel):
    position = "late"
