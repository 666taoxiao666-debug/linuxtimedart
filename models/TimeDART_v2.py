import torch
import torch.nn as nn
try:
    from transformers import AutoModel
except ImportError:  # TimeDART/SimMTM should remain usable without the v2 extra.
    AutoModel = None
from layers.TimeDART_EncDec import (
    ChannelIndependence,
    AddSosTokenAndDropLast,
    CausalTransformer,
    Diffusion,
    DenoisingPatchDecoder,
    PromptEncoder,
    RegimePredictor,
    SoftPromptGenerator,
    DilatedConvEncoder,
    ClsEmbedding,
    ClsHead,
    OldClsHead,
    ClsFlattenHead,
    ARFlattenHead,
)
from layers.Embed import Patch, PatchEmbedding, PositionalEncoding
from utils.regime_labels import compute_regime_pseudo_labels_from_series


class FlattenHead(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        pred_len: int,
        dropout: float,
    ):
        super(FlattenHead, self).__init__()
        self.pred_len = pred_len
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecast_head = nn.Linear(seq_len * d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, num_features, seq_len, d_model]
        :return: [batch_size, pred_len, num_features]
        """
        x = self.flatten(x)  # (batch_size, num_features, seq_len * d_model)
        x = self.forecast_head(x)  # (batch_size, num_features, pred_len)
        x = self.dropout(x)  # (batch_size, num_features, pred_len)
        x = x.permute(0, 2, 1)  # (batch_size, pred_len, num_features)
        return x


class Model(nn.Module):
    """
    TimeDART v2 with Qwen2.5-0.5B encoder
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.input_len = args.input_len
        self.args = args

        # For Model Hyperparameters
        self.d_model = args.d_model
        self.num_heads = args.n_heads
        self.feedforward_dim = args.d_ff
        self.dropout = args.dropout
        self.device = args.device
        self.task_name = args.task_name
        self.pred_len = args.pred_len
        self.use_norm = args.use_norm
        self.regime_target_index = getattr(args, "regime_target_index", -1)
        self.channel_independence = ChannelIndependence()

        # Patch
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.patch = Patch(
            patch_len=self.patch_len,
            stride=self.stride,
        )
        self.seq_len = int((self.input_len - self.patch_len) / self.stride) + 1

        # Embedding
        self.enc_embedding = PatchEmbedding(
            patch_len=self.patch_len,
            d_model=self.d_model,
        )

        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout,
        )

        sos_token = torch.randn(1, 1, self.d_model, device=self.device)
        self.sos_token = nn.Parameter(sos_token, requires_grad=True)

        self.add_sos_token_and_drop_last = AddSosTokenAndDropLast(
            sos_token=self.sos_token,
        )

        # Initialize encoder. Exp_Basic owns device selection and moves the
        # complete model after construction.
        self.encoder = self.__init_encoder()

        # Diffusion
        self.diffusion = Diffusion(
            time_steps=args.time_steps,
            device=self.device,
            scheduler=args.scheduler,
        )

        self.use_soft_prompt = getattr(args, "use_soft_prompt", False)
        self.use_prompt_adaln = getattr(args, "use_prompt_adaln", False) or self.use_soft_prompt
        self.regime_stable_thresh = getattr(args, "regime_stable_thresh", 0.15)
        self.regime_ramp_thresh = getattr(args, "regime_ramp_thresh", 0.25)

        if self.use_soft_prompt:
            self.num_modes = getattr(args, "num_modes", 3)
            self.prompt_dim = args.d_model
            self.regime_predictor = RegimePredictor(
                d_model=args.d_model,
                num_modes=self.num_modes,
                dropout=args.dropout,
            )
            self.soft_prompt_generator = SoftPromptGenerator(
                num_modes=self.num_modes,
                d_model=args.d_model,
            )
        else:
            self.prompt_dim = (
                args.prompt_dim if args.prompt_dim is not None else args.d_model
            )
            if self.use_prompt_adaln:
                self.prompt_encoder = PromptEncoder(
                    d_model=args.d_model,
                    prompt_dim=self.prompt_dim,
                    dropout=args.dropout,
                )

        if self.task_name == "pretrain":
            self.denoising_patch_decoder = DenoisingPatchDecoder(
                d_model=args.d_model,
                num_layers=args.d_layers,
                num_heads=args.n_heads,
                feedforward_dim=args.d_ff,
                dropout=args.dropout,
                mask_ratio=args.mask_ratio,
                use_prompt_adaln=self.use_prompt_adaln,
                prompt_dim=self.prompt_dim,
            )
            self.projection = FlattenHead(
                seq_len=self.seq_len,
                d_model=self.d_model,
                pred_len=self.input_len,
                dropout=args.head_dropout,
            )
        elif self.task_name == "finetune":
            self.head = FlattenHead(
                seq_len=self.seq_len,
                d_model=self.d_model,
                pred_len=self.pred_len,
                dropout=args.head_dropout,
            )

    def __init_encoder(self):
        """Initialize Qwen2.5-0.5B encoder"""
        if self.args.backbone == "Qwen2.5-0.5B":
            if AutoModel is None:
                raise ImportError(
                    "TimeDART_v2 with a Qwen backbone requires the transformers package"
                )
            encoder = AutoModel.from_pretrained(
                self.args.llm_path,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            hidden_size = getattr(encoder.config, "hidden_size", None)
            if hidden_size != self.d_model:
                raise ValueError(
                    f"d_model={self.d_model} must match the backbone hidden_size={hidden_size}"
                )
        elif self.args.backbone == "Transformer":
            encoder = CausalTransformer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                num_layers=self.args.e_layers,
                feedforward_dim=self.feedforward_dim,
                dropout=self.dropout,
            )
        else:
            raise ValueError(f"Backbone {self.args.backbone} not supported")
        return encoder

    def _encode(self, embedding, causal):
        if self.args.backbone == "Qwen2.5-0.5B":
            return self.encoder(inputs_embeds=embedding, return_dict=True).last_hidden_state
        return self.encoder(embedding, is_mask=causal)

    def _build_prompt(self, x_out, label_source, batch_size, num_features):
        if getattr(self, "use_soft_prompt", False):
            target_index = self.regime_target_index % num_features
            target_hidden = x_out.reshape(
                batch_size, num_features, x_out.size(1), x_out.size(2)
            )[:, target_index]
            regime_logits, regime_probs = self.regime_predictor(target_hidden)
            sample_prompt = self.soft_prompt_generator(regime_probs)
            soft_prompt = sample_prompt.repeat_interleave(num_features, dim=0)
            pseudo_labels = compute_regime_pseudo_labels_from_series(
                label_source[:, :, target_index],
                stable_thresh=self.regime_stable_thresh,
                ramp_thresh=self.regime_ramp_thresh,
            )
            return soft_prompt, regime_logits, pseudo_labels

        prompt_emb = (
            self.prompt_encoder(x_out)
            if getattr(self, "use_prompt_adaln", False)
            and hasattr(self, "prompt_encoder")
            else None
        )
        return prompt_emb, None, None

    def pretrain(self, x):
        # [batch_size, input_len, num_features]
        batch_size, input_len, num_features = x.size()
        label_source = x
        if self.use_norm:
            # Instance Normalization
            means = torch.mean(
                x, dim=1, keepdim=True
            ).detach()  # [batch_size, 1, num_features], detach from gradient
            x = x - means  # [batch_size, input_len, num_features]
            stdevs = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()  # [batch_size, 1, num_features]
            x = x / stdevs  # [batch_size, input_len, num_features]

        # Channel Independence
        x = self.channel_independence(x)  # [batch_size * num_features, input_len, 1]
        # Patch
        x_patch = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]

        # For Qwen2.5-0.5B Encoder
        x_embedding = self.enc_embedding(
            x_patch
        )  # [batch_size * num_features, seq_len, d_model]
        x_embedding_bias = self.add_sos_token_and_drop_last(
            x_embedding
        )  # [batch_size * num_features, seq_len, d_model]
        x_embedding_bias = self.positional_encoding(x_embedding_bias)

        # Get encoder outputs from Qwen2.5-0.5B
        x_out = self._encode(x_embedding_bias, causal=True)

        # Noising Diffusion
        noise_x_patch, noise, t = self.diffusion(
            x_patch
        )  # [batch_size * num_features, seq_len, patch_len]
        noise_x_embedding = self.enc_embedding(
            noise_x_patch
        )  # [batch_size * num_features, seq_len, d_model]
        noise_x_embedding = self.positional_encoding(noise_x_embedding)

        # For Denoising Patch Decoder
        prompt_emb, regime_logits, pseudo_labels = self._build_prompt(
            x_out, label_source, batch_size, num_features
        )
        predict_x = self.denoising_patch_decoder(
            query=noise_x_embedding,
            key=x_out,
            value=x_out,
            is_tgt_mask=True,
            is_src_mask=True,
            prompt_emb=prompt_emb,
        )  # [batch_size * num_features, seq_len, d_model]

        # For Decoder
        predict_x = predict_x.reshape(
            batch_size, num_features, -1, self.d_model
        )  # [batch_size, num_features, seq_len, d_model]
        predict_x = self.projection(predict_x)  # [batch_size, input_len, num_features]

        # Instance Denormalization
        if self.use_norm:
            predict_x = predict_x * (stdevs[:, 0, :].unsqueeze(1)).repeat(
                1, input_len, 1
            )  # [batch_size, input_len, num_features]
            predict_x = predict_x + (means[:, 0, :].unsqueeze(1)).repeat(
                1, input_len, 1
            )  # [batch_size, input_len, num_features]

        if getattr(self, "use_soft_prompt", False):
            return {
                "pred": predict_x,
                "regime_logits": regime_logits,
                "pseudo_labels": pseudo_labels,
            }
        return predict_x

    def forecast(self, x):
        """Forecast from the history only; no future target is an input."""
        batch_size, _, num_features = x.size()
        label_source = x
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            centered = x - means
            stdevs = torch.sqrt(
                torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x = centered / stdevs

        x = self.channel_independence(x)
        x = self.patch(x)
        x = self.enc_embedding(x)
        x = self.positional_encoding(x)
        x = self._encode(x, causal=False)

        if self.use_soft_prompt:
            prompt_emb, _, _ = self._build_prompt(
                x, label_source, batch_size, num_features
            )
            x = x + prompt_emb.unsqueeze(1)

        x = x.reshape(batch_size, num_features, self.seq_len, self.d_model)
        prediction = self.head(x)
        if self.use_norm:
            prediction = prediction * stdevs[:, 0, :].unsqueeze(1)
            prediction = prediction + means[:, 0, :].unsqueeze(1)
        return prediction

    def forward(self, x):
        if self.task_name == "pretrain":
            return self.pretrain(x)
        elif self.task_name == "finetune":
            return self.forecast(x)
        else:
            raise ValueError(f"Task name {self.task_name} not supported")
