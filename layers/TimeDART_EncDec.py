import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import generate_causal_mask, generate_self_only_mask, generate_partial_mask


class ChannelIndependence(nn.Module):
    def __init__(
        self,
    ):
        super(ChannelIndependence, self).__init__()

    def forward(self, x):
        """
        :param x: [batch_size, input_len, num_features]
        :return: [batch_size * num_features, input_len, 1]
        """
        _, input_len, _ = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, input_len, 1)
        return x


class ChannelMixer(nn.Module):
    """Fuse every channel token at each patch into one power token.

    TimeDART encodes channels independently, so wind speed never reaches the
    power head unless the patch tokens are mixed here.  The mixer is used only
    in MS fine-tuning; pre-training still reconstructs each channel on its own.

    ``channel_prior`` scales each channel before concat so Wspd/power start
    larger than weak SCADA.  ``context`` is a pre-norm operating-point vector
    (last/mean wind, last power, yaw, pitch) added to every patch token.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int,
        dropout: float,
        channel_prior=None,
        context_dim: int = 0,
    ):
        super().__init__()
        if num_features < 1:
            raise ValueError("num_features must be positive")
        self.num_features = int(num_features)
        if channel_prior is None:
            prior = torch.ones(self.num_features)
        else:
            prior = torch.as_tensor(channel_prior, dtype=torch.float32).reshape(-1)
            if prior.numel() != self.num_features:
                raise ValueError(
                    "channel_prior length must equal num_features: "
                    f"{prior.numel()} != {self.num_features}"
                )
        self.log_channel_scale = nn.Parameter(torch.log(prior.clamp(min=1e-4)))
        self.proj = nn.Sequential(
            nn.Linear(self.num_features * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.context_dim = int(context_dim)
        self.context_proj = (
            nn.Linear(self.context_dim, d_model) if self.context_dim > 0 else None
        )
        self.context_gate = (
            nn.Linear(self.context_dim, self.num_features)
            if self.context_dim > 0
            else None
        )
        if self.context_gate is not None:
            # 2*sigmoid(0)=1: training starts exactly from the physics prior,
            # then learns regime-specific channel importance per sample.
            nn.init.zeros_(self.context_gate.weight)
            nn.init.zeros_(self.context_gate.bias)

    def channel_scale(self) -> torch.Tensor:
        return self.log_channel_scale.exp()

    def effective_channel_scale(self, context: torch.Tensor = None) -> torch.Tensor:
        """Return static physics priors or per-sample context-adaptive scales."""
        base = self.channel_scale()
        if self.context_gate is None:
            return base
        if context is None:
            raise ValueError("ChannelMixer was built with context_dim>0 but context is None")
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError(
                "operating context width does not match mixer: "
                f"{tuple(context.shape)} vs context_dim={self.context_dim}"
            )
        dynamic = 2.0 * torch.sigmoid(self.context_gate(context))
        return dynamic * base.unsqueeze(0)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """
        :param x: [batch_size, num_features, num_patches, d_model]
        :param context: optional [batch_size, context_dim] operating point
        :return: [batch_size, 1, num_patches, d_model]
        """
        if x.ndim != 4:
            raise ValueError(f"ChannelMixer expects 4D input, got {tuple(x.shape)}")
        batch, num_features, num_patches, d_model = x.shape
        if num_features != self.num_features:
            raise ValueError(
                "ChannelMixer feature count does not match the data: "
                f"encoder={num_features}, mixer={self.num_features}. "
                "Set --enc_in to the SDWPF feature count."
            )
        scale = self.effective_channel_scale(context)
        if scale.ndim == 1:
            scale = scale.unsqueeze(0)
        if scale.shape[0] not in (1, batch):
            raise ValueError(
                f"context batch size {scale.shape[0]} does not match input batch {batch}"
            )
        scale = scale.view(scale.shape[0], self.num_features, 1, 1)
        x = x * scale
        mixed = x.permute(0, 2, 1, 3).reshape(
            batch, num_patches, num_features * d_model
        )
        mixed = self.proj(mixed)
        if self.context_proj is not None:
            mixed = mixed + self.context_proj(context).unsqueeze(1)
        return mixed.unsqueeze(1)


class AddSosTokenAndDropLast(nn.Module):
    def __init__(self, sos_token: torch.Tensor):
        super(AddSosTokenAndDropLast, self).__init__()
        assert sos_token.dim() == 3
        self.sos_token = sos_token

    def forward(self, x):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        sos_token_expanded = self.sos_token.expand(
            x.size(0), -1, -1
        )  # [batch_size * num_features, 1, d_model]
        x = torch.cat(
            [sos_token_expanded, x], dim=1
        )  # [batch_size * num_features, seq_len + 1, d_model]
        x = x[:, :-1, :]  # [batch_size * num_features, seq_len, d_model]
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float
    ):
        super(TransformerEncoderBlock, self).__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=feedforward_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=feedforward_dim, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :param mask: [1, 1, seq_len, seq_len]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        # Self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(x)
        output = self.norm2(x + self.dropout(ff_output))

        return output


class CausalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ):
        super(CausalTransformer, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, is_mask=True):
        # x: [batch_size * num_features, seq_len, d_model]
        seq_len = x.size(1)
        mask = generate_causal_mask(seq_len).to(x.device) if is_mask else None
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return x


class Diffusion(nn.Module):
    def __init__(
        self,
        time_steps: int,
        device: torch.device,
        scheduler: str = "cosine",
    ):
        super(Diffusion, self).__init__()
        self.device = device
        self.time_steps = time_steps

        if scheduler == "cosine":
            self.betas = self._cosine_beta_schedule().to(self.device)
        elif scheduler == "linear":
            self.betas = self._linear_beta_schedule().to(self.device)
        else:
            raise ValueError(f"Invalid scheduler: {scheduler=}")

        self.alpha = 1 - self.betas
        self.gamma = torch.cumprod(self.alpha, dim=0).to(self.device)

    def _cosine_beta_schedule(self, s=0.008):
        steps = self.time_steps + 1
        x = torch.linspace(0, self.time_steps, steps)
        alphas_cumprod = (
            torch.cos(((x / self.time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def _linear_beta_schedule(self, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, self.time_steps)
        return betas

    def sample_time_steps(self, shape):
        return torch.randint(0, self.time_steps, shape, device=self.device)

    def noise(self, x, t):
        noise = torch.randn_like(x)
        gamma_t = self.gamma[t].unsqueeze(-1)  # [batch_size * num_features, seq_len, 1]
        # x_t = sqrt(gamma_t) * x + sqrt(1 - gamma_t) * noise
        noisy_x = torch.sqrt(gamma_t) * x + torch.sqrt(1 - gamma_t) * noise
        return noisy_x, noise

    def forward(self, x):
        # x: [batch_size * num_features, seq_len, patch_len]
        t = self.sample_time_steps(x.shape[:2])  # [batch_size * num_features, seq_len]
        noisy_x, noise = self.noise(x, t)
        return noisy_x, noise, t


class AdaLN_Modulation(nn.Module):
    """AdaLN-Zero: prompt-conditioned FiLM scale/shift on decoder features."""

    def __init__(self, prompt_dim, d_model):
        super(AdaLN_Modulation, self).__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(prompt_dim, d_model * 2),
        )
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)

    def forward(self, x, prompt_emb):
        """
        :param x: [batch_size, seq_len, d_model]
        :param prompt_emb: [batch_size, prompt_dim]
        :return: modulated x [batch_size, seq_len, d_model]
        """
        params = self.mlp(prompt_emb)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return x * (1 + gamma) + beta


class PromptEncoder(nn.Module):
    """Encode encoder hidden states into a global prompt vector P."""

    def __init__(self, d_model, prompt_dim, dropout=0.1):
        super(PromptEncoder, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, prompt_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_out):
        """
        :param x_out: [batch_size, seq_len, d_model]
        :return: prompt_emb [batch_size, prompt_dim]
        """
        pooled = x_out.mean(dim=1)
        return self.proj(pooled)


class RegimePredictor(nn.Module):
    """Auxiliary regime predictor: encoder output -> mode probability p."""

    def __init__(self, d_model, num_modes, dropout=0.1):
        super(RegimePredictor, self).__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_modes),
        )

    def forward(self, x_out):
        """
        :param x_out: [batch_size, seq_len, d_model]
        :return: logits [batch_size, num_modes], probs [batch_size, num_modes]
        """
        logits = self.classifier(x_out.transpose(1, 2))
        probs = F.softmax(logits, dim=-1)
        return logits, probs


class SoftPromptGenerator(nn.Module):
    """Learnable prompt dictionary E; soft prompt P = p @ E."""

    def __init__(self, num_modes, d_model):
        super(SoftPromptGenerator, self).__init__()
        self.prompt_embeddings = nn.Parameter(torch.randn(num_modes, d_model) * 0.02)

    def forward(self, probs):
        """
        :param probs: [batch_size, num_modes]
        :return: soft_prompt [batch_size, d_model]
        """
        return torch.matmul(probs, self.prompt_embeddings)


class SceneWikiPromptRouter(nn.Module):
    """Retrieve and mix frozen LLM scene anchors from a causal state query.

    The LLM is used offline to encode the Wiki text. During training and
    inference this module contains only tensors and small trainable projections,
    so runs are deterministic and do not depend on an online service.
    """

    def __init__(
        self,
        semantic_embeddings,
        d_model,
        *,
        num_features=None,
        channel_prior=None,
        top_k=2,
        temperature=0.2,
        rule_weight=2.0,
        prompt_gate_init=-2.2,
        null_scene_index=None,
        dropout=0.1,
    ):
        super().__init__()
        keys = torch.as_tensor(semantic_embeddings, dtype=torch.float32)
        if keys.ndim != 2 or keys.size(0) < 2 or keys.size(1) < 1:
            raise ValueError(
                "semantic_embeddings must have shape [num_scenes, hidden_size]"
            )
        if not torch.isfinite(keys).all():
            raise ValueError("semantic_embeddings contain non-finite values")
        if not 1 <= int(top_k) <= int(keys.size(0)):
            raise ValueError("scene_wiki_top_k must be between 1 and num_scenes")
        if float(temperature) <= 0:
            raise ValueError("scene_wiki_temperature must be positive")
        if float(rule_weight) < 0:
            raise ValueError("scene_wiki_rule_weight cannot be negative")
        self.num_scenes = int(keys.size(0))
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.rule_weight = float(rule_weight)
        self.null_scene_index = (
            None if null_scene_index is None else int(null_scene_index)
        )
        if self.null_scene_index is not None and not (
            0 <= self.null_scene_index < self.num_scenes
        ):
            raise ValueError("null_scene_index must identify a valid Wiki scene")
        self.prompt_gate_logit = nn.Parameter(
            torch.tensor(float(prompt_gate_init), dtype=torch.float32)
        )
        self.register_buffer("semantic_keys", F.normalize(keys, dim=-1))
        self.num_features = int(num_features) if num_features is not None else None
        if self.num_features is not None:
            prior = torch.as_tensor(channel_prior, dtype=torch.float32).reshape(-1)
            if prior.numel() != self.num_features:
                raise ValueError("Wiki router channel prior must match num_features")
            self.log_channel_weight = nn.Parameter(torch.log(prior.clamp_min(1e-4)))
        else:
            self.log_channel_weight = None
        self.semantic_projection = nn.Linear(keys.size(1), d_model, bias=False)
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.prompt_delta = nn.Parameter(torch.zeros(self.num_scenes, d_model))
        self.prompt_norm = nn.LayerNorm(d_model)

    def forward(self, x_out, rule_logits=None):
        if x_out.ndim == 4:
            if self.log_channel_weight is None or x_out.size(1) != self.num_features:
                raise ValueError(
                    "Wiki router multichannel input does not match its configured features"
                )
            weights = F.softmax(self.log_channel_weight, dim=0).view(1, -1, 1, 1)
            x_out = (x_out * weights).sum(dim=1)
        if x_out.ndim != 3:
            raise ValueError(
                f"Wiki router expects [batch,patch,d_model] or multichannel 4D, got {x_out.shape}"
            )
        state = torch.cat([x_out.mean(dim=1), x_out[:, -1]], dim=-1)
        query = F.normalize(self.query_encoder(state), dim=-1)
        projected_keys = self.semantic_projection(self.semantic_keys)
        retrieval_keys = F.normalize(projected_keys, dim=-1)
        retrieval_logits = torch.matmul(query, retrieval_keys.transpose(0, 1))
        retrieval_logits = retrieval_logits / self.temperature
        routing_logits = retrieval_logits
        if rule_logits is not None:
            if rule_logits.shape != retrieval_logits.shape:
                raise ValueError(
                    "rule logits must match retrieval logits: "
                    f"{rule_logits.shape} != {retrieval_logits.shape}"
                )
            routing_logits = routing_logits + self.rule_weight * rule_logits
        if self.top_k < self.num_scenes:
            keep = routing_logits.topk(self.top_k, dim=-1).indices
            masked = torch.full_like(routing_logits, float("-inf"))
            masked.scatter_(1, keep, routing_logits.gather(1, keep))
            routing_logits = masked
        probabilities = F.softmax(routing_logits, dim=-1)
        prompt_values = self.prompt_norm(projected_keys + self.prompt_delta)
        prompt_probabilities = probabilities
        if self.null_scene_index is not None:
            prompt_probabilities = probabilities.clone()
            prompt_probabilities[:, self.null_scene_index] = 0.0
        prompt = torch.sigmoid(self.prompt_gate_logit) * torch.matmul(
            prompt_probabilities, prompt_values
        )
        return prompt, retrieval_logits, probabilities


class CompositionalEventWikiRouter(nn.Module):
    """Compose sparse event-factor residuals over the ordinary trend prompt.

    Each frozen semantic anchor represents one observable physical event.  The
    router uses independent sigmoid scores (not a mutually exclusive softmax),
    keeps at most ``top_k`` factors, and requires both semantic confidence and
    positive physical-rule margin.  Consequently an all-absent or ambiguous
    event vector produces an exact zero residual rather than a learned null
    prompt that could silently perturb normal operation.
    """

    def __init__(
        self,
        semantic_embeddings,
        d_model,
        *,
        factor_reliability=None,
        num_features=None,
        channel_prior=None,
        top_k=2,
        temperature=0.2,
        rule_weight=2.0,
        prompt_gate_init=-2.2,
        activation_threshold=0.55,
        confidence_power=1.0,
        dropout=0.1,
    ):
        super().__init__()
        keys = torch.as_tensor(semantic_embeddings, dtype=torch.float32)
        if keys.ndim != 2 or keys.size(0) < 1 or keys.size(1) < 1:
            raise ValueError(
                "semantic_embeddings must have shape [num_factors, hidden_size]"
            )
        if not torch.isfinite(keys).all():
            raise ValueError("semantic_embeddings contain non-finite values")
        if not 1 <= int(top_k) <= int(keys.size(0)):
            raise ValueError("scene_wiki_top_k must be between 1 and num_factors")
        if float(temperature) <= 0:
            raise ValueError("scene_wiki_temperature must be positive")
        if not 0.0 <= float(activation_threshold) < 1.0:
            raise ValueError("Wiki activation threshold must be in [0, 1)")
        if float(confidence_power) <= 0:
            raise ValueError("Wiki confidence power must be positive")

        self.num_factors = int(keys.size(0))
        # Keep the generic attribute for audit/checkpoint utilities shared with
        # the legacy single-label router.
        self.num_scenes = self.num_factors
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.rule_weight = float(rule_weight)
        self.activation_threshold = float(activation_threshold)
        self.confidence_power = float(confidence_power)
        self.prompt_gate_logit = nn.Parameter(
            torch.tensor(float(prompt_gate_init), dtype=torch.float32)
        )
        self.register_buffer("semantic_keys", F.normalize(keys, dim=-1))
        reliability = torch.as_tensor(
            (
                torch.ones(self.num_factors, dtype=torch.float32)
                if factor_reliability is None
                else factor_reliability
            ),
            dtype=torch.float32,
        ).reshape(-1)
        if reliability.numel() != self.num_factors:
            raise ValueError("factor_reliability must match num_factors")
        if not torch.isfinite(reliability).all() or (
            (reliability < 0.0) | (reliability > 1.0)
        ).any():
            raise ValueError("factor_reliability must be finite and in [0, 1]")
        # Frozen lifecycle confidence.  It is estimated offline from train-only
        # OOF evidence; optimization must not learn it from validation/test loss.
        self.register_buffer("factor_reliability", reliability)
        self.num_features = int(num_features) if num_features is not None else None
        if self.num_features is not None:
            prior = torch.as_tensor(channel_prior, dtype=torch.float32).reshape(-1)
            if prior.numel() != self.num_features:
                raise ValueError("Event Wiki channel prior must match num_features")
            self.log_channel_weight = nn.Parameter(torch.log(prior.clamp_min(1e-4)))
        else:
            self.log_channel_weight = None
        self.semantic_projection = nn.Linear(keys.size(1), d_model, bias=False)
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.prompt_delta = nn.Parameter(torch.zeros(self.num_factors, d_model))
        self.prompt_norm = nn.LayerNorm(d_model)

    def forward(self, x_out, rule_logits):
        if x_out.ndim == 4:
            if self.log_channel_weight is None or x_out.size(1) != self.num_features:
                raise ValueError(
                    "Event Wiki multichannel input does not match its configured features"
                )
            weights = F.softmax(self.log_channel_weight, dim=0).view(1, -1, 1, 1)
            x_out = (x_out * weights).sum(dim=1)
        if x_out.ndim != 3:
            raise ValueError(
                "Event Wiki expects [batch,patch,d_model] or multichannel 4D, "
                f"got {x_out.shape}"
            )

        state = torch.cat([x_out.mean(dim=1), x_out[:, -1]], dim=-1)
        query = F.normalize(self.query_encoder(state), dim=-1)
        projected_keys = self.semantic_projection(self.semantic_keys)
        retrieval_keys = F.normalize(projected_keys, dim=-1)
        retrieval_logits = torch.matmul(query, retrieval_keys.transpose(0, 1))
        retrieval_logits = retrieval_logits / self.temperature
        if rule_logits.shape != retrieval_logits.shape:
            raise ValueError(
                "event rule logits must match retrieval logits: "
                f"{rule_logits.shape} != {retrieval_logits.shape}"
            )
        if not torch.isfinite(rule_logits).all():
            raise ValueError("event rule logits contain non-finite values")

        # Semantic confidence is evaluated independently.  A strong physical
        # rule is not allowed to manufacture semantic confidence by being added
        # to the retrieval logit; both branches must support intervention.
        probabilities = torch.sigmoid(retrieval_logits)
        confidence = (
            (probabilities - self.activation_threshold)
            / max(1.0 - self.activation_threshold, 1e-6)
        ).clamp(0.0, 1.0)
        confidence = confidence.pow(self.confidence_power)
        # Positive signed margin is mandatory.  Rule weight controls how fast
        # physical support saturates, while zero/negative evidence remains an
        # exact zero and can never be rescued by semantic similarity alone.
        positive_margin = rule_logits.clamp(0.0, 1.0)
        physical_support = 1.0 - torch.exp(-self.rule_weight * positive_margin)
        activations = (
            confidence
            * physical_support
            * self.factor_reliability.view(1, -1)
        )
        if self.top_k < self.num_factors:
            keep = activations.topk(self.top_k, dim=-1).indices
            support = torch.zeros_like(activations, dtype=torch.bool)
            support.scatter_(1, keep, True)
            activations = activations.masked_fill(~support, 0.0)

        prompt_values = self.prompt_norm(projected_keys + self.prompt_delta)
        active_count = (activations > 0).sum(dim=-1, keepdim=True).to(activations.dtype)
        composition_scale = active_count.clamp_min(1.0).sqrt()
        prompt = torch.matmul(activations, prompt_values) / composition_scale
        prompt = torch.sigmoid(self.prompt_gate_logit) * prompt
        return prompt, retrieval_logits, probabilities, activations


class PromptGuidedDecoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float, prompt_dim: int
    ):
        super(PromptGuidedDecoderBlock, self).__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.encoder_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.adaLN1 = AdaLN_Modulation(prompt_dim, d_model)
        self.adaLN2 = AdaLN_Modulation(prompt_dim, d_model)
        self.adaLN3 = AdaLN_Modulation(prompt_dim, d_model)

    def forward(self, query, key, value, tgt_mask, src_mask, prompt_emb):
        attn_output, _ = self.self_attention(query, query, query, attn_mask=tgt_mask)
        query_norm = self.norm1(query + self.dropout(attn_output))
        query = self.adaLN1(query_norm, prompt_emb)

        attn_output, _ = self.encoder_attention(query, key, value, attn_mask=src_mask)
        query_norm = self.norm2(query + self.dropout(attn_output))
        query = self.adaLN2(query_norm, prompt_emb)

        ff_output = self.ff(query)
        x_norm = self.norm3(query + self.dropout(ff_output))
        x = self.adaLN3(x_norm, prompt_emb)

        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float
    ):
        super(TransformerDecoderBlock, self).__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.encoder_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, tgt_mask, src_mask):
        """
        :param query: [batch_size * num_features, seq_len, d_model]
        :param key: [batch_size * num_features, seq_len, d_model]
        :param value: [batch_size * num_features, seq_len, d_model]
        :param mask: [1, 1, seq_len, seq_len]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        # Self-attention
        attn_output, _ = self.self_attention(query, query, query, attn_mask=tgt_mask)
        query = self.norm1(query + self.dropout(attn_output))

        # Encoder attention
        attn_output, _ = self.encoder_attention(query, key, value, attn_mask=src_mask)
        query = self.norm2(query + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(query)
        x = self.norm3(query + self.dropout(ff_output))

        return x


class DenoisingPatchDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
        mask_ratio: float,
        use_prompt_adaln: bool = False,
        prompt_dim: int = None,
    ):
        super(DenoisingPatchDecoder, self).__init__()

        self.use_prompt_adaln = use_prompt_adaln
        prompt_dim = prompt_dim or d_model

        if use_prompt_adaln:
            self.layers = nn.ModuleList(
                [
                    PromptGuidedDecoderBlock(
                        d_model, num_heads, feedforward_dim, dropout, prompt_dim
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    TransformerDecoderBlock(d_model, num_heads, feedforward_dim, dropout)
                    for _ in range(num_layers)
                ]
            )
        self.norm = nn.LayerNorm(d_model)
        self.mask_ratio = mask_ratio

    def forward(
        self,
        query,
        key,
        value,
        is_tgt_mask=True,
        is_src_mask=True,
        prompt_emb=None,
    ):
        seq_len = query.size(1)
        tgt_mask = (
            generate_partial_mask(seq_len, self.mask_ratio).to(query.device) if is_tgt_mask else None
        )
        src_mask = (
            generate_partial_mask(seq_len, self.mask_ratio).to(query.device) if is_src_mask else None
        )
        for layer in self.layers:
            if self.use_prompt_adaln:
                query = layer(query, key, value, tgt_mask, src_mask, prompt_emb)
            else:
                query = layer(query, key, value, tgt_mask, src_mask)
        x = self.norm(query)
        return x


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(self.receptive_field - 1),  # 左填充
            dilation=dilation,
            groups=groups
        )
        
    def forward(self, x):
        out = self.conv(x)
        # 裁剪掉多余的未来时间步，确保与输入长度一致
        return out[:, :, :x.size(2)]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class CausalTCN(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, kernel_size=3):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        
        # First linear layer to map input_dims to hidden_dims
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        
        # Create a dilated causal convolutional encoder
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * (depth - 1) + [output_dims],
            kernel_size=kernel_size
        )
        
    def forward(self, x):
        # Input x is of shape [batch_size, seq_len, input_dims]
        
        # Flatten input (batch_size, seq_len, input_dims) -> (batch_size, seq_len, hidden_dims)
        x = self.input_fc(x)
        
        # Transpose for the convolution (batch_size, seq_len, hidden_dims) -> (batch_size, hidden_dims, seq_len)
        x = x.transpose(1, 2)
        
        # Apply dilated convolutions
        x = self.feature_extractor(x)  # [batch_size, hidden_dims, seq_len] -> [batch_size, output_dims, seq_len]
        
        # Transpose back to [batch_size, seq_len, output_dims]
        x = x.transpose(1, 2)
        
        return x


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):   
        """
        :param x: [batch_size, seq_len, input_dims]
        :return: [batch_size, seq_len, output_dims]
        """
        x = x.transpose(1, 2)
        return self.net(x).transpose(1, 2)


class ClsHead(nn.Module):
    def __init__(self, seq_len, d_model, num_classes, dropout):
        super(ClsHead, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(seq_len * d_model, num_classes)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)


class OldClsHead(nn.Module):
    def __init__(self, seq_len, d_model, num_classes, dropout):
        super(OldClsHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.fc(torch.max(x, dim=1)[0])


class ClsEmbedding(nn.Module):
    def __init__(self, num_features, d_model, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=num_features, 
            out_channels=d_model, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        return self.conv(x).transpose(1, 2) 


class ClsFlattenHead(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, num_features, dropout):
        super(ClsFlattenHead, self).__init__()
        self.pred_len = pred_len
        self.num_features = num_features
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecast_head = nn.Linear(seq_len * d_model, pred_len * num_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        :param x: [batch_size, seq_len, d_model]
        :return: [batch_size, pred_len, num_features]
        """
        x = self.flatten(x)  # [batch_size, seq_len * d_model]
        x = self.dropout(x)  # [batch_size, seq_len * d_model]
        x = self.forecast_head(x)  # [batch_size, pred_len * num_features]
        return x.reshape(x.size(0), self.pred_len, self.num_features)


class ARFlattenHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch_len: int,
        dropout: float,
    ):
        super(ARFlattenHead, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecast_head = nn.Linear(d_model, patch_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, num_features, seq_len, d_model]
        :return: [batch_size, seq_len * patch_len, num_features]
        """
        x = self.forecast_head(x)  # (batch_size, num_features, seq_len, patch_len)
        x = self.dropout(x)  # (batch_size, num_features, seq_len, patch_len)
        x = self.flatten(x)  # (batch_size, num_features, seq_len * patch_len)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len * patch_len, num_features)
        return x
