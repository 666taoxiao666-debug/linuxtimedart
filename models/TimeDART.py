import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
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
from layers.Embed import Patch, PatchEmbedding, PositionalEncoding, TokenEmbedding_TimeDART
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
        x = self.flatten(x)
        x = self.forecast_head(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x


class Model(nn.Module):
    """
    TimeDART
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.input_len = args.input_len

        # For Model Hyperparameters
        self.d_model = args.d_model
        self.num_heads = args.n_heads
        self.feedforward_dim = args.d_ff
        self.dropout = args.dropout
        self.device = args.device
        self.task_name = args.task_name
        self.pred_len = args.pred_len
        self.use_norm = args.use_norm
        self.residual_forecast = getattr(args, "residual_forecast", False)
        self.zero_init_residual_head = getattr(
            args,
            "zero_init_residual_head",
            False,
        )
        self.use_soft_prompt = getattr(args, "use_soft_prompt", False)
        self.use_prompt_adaln = (
            getattr(args, "use_prompt_adaln", False)
            or self.use_soft_prompt
        )
        self.regime_target_index = getattr(
            args,
            "regime_target_index",
            -1,
        )
        self.regime_stable_thresh = getattr(
            args,
            "regime_stable_thresh",
            0.15,
        )
        self.regime_ramp_thresh = getattr(
            args,
            "regime_ramp_thresh",
            0.25,
        )
        self.channel_independence = ChannelIndependence()

        # Patch
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.patch = Patch(
            patch_len=self.patch_len,
            stride=self.stride,
        )
        self.seq_len = (
            int(
                (self.input_len - self.patch_len)
                / self.stride
            )
            + 1
        )

        # Embedding
        self.enc_embedding = PatchEmbedding(
            patch_len=self.patch_len,
            d_model=self.d_model,
        )

        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout,
        )

        sos_token = torch.randn(
            1,
            1,
            self.d_model,
            device=self.device,
        )
        self.sos_token = nn.Parameter(
            sos_token,
            requires_grad=True,
        )

        self.add_sos_token_and_drop_last = (
            AddSosTokenAndDropLast(
                sos_token=self.sos_token,
            )
        )

        # Encoder (Causal Transformer)
        self.diffusion = Diffusion(
            time_steps=args.time_steps,
            device=self.device,
            scheduler=args.scheduler,
        )
        self.encoder = CausalTransformer(
            d_model=args.d_model,
            num_heads=args.n_heads,
            feedforward_dim=args.d_ff,
            dropout=args.dropout,
            num_layers=args.e_layers,
        )

        # self.encoder = DilatedConvEncoder(
        #     in_channels=self.d_model,
        #     channels=[self.d_model] * args.e_layers,
        #     kernel_size=3,
        # )

        # Prompt modules are also built for finetuning, so a
        # PromptTimeDART checkpoint does not discard the learned
        # regime classifier/dictionary.
        if self.use_soft_prompt:
            self.num_modes = getattr(args, "num_modes", 3)
            if self.num_modes != 3:
                raise ValueError(
                    "The pseudo-label definition requires "
                    "num_modes=3"
                )

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
                args.prompt_dim
                if args.prompt_dim is not None
                else args.d_model
            )

            if self.use_prompt_adaln:
                self.prompt_encoder = PromptEncoder(
                    d_model=args.d_model,
                    prompt_dim=self.prompt_dim,
                    dropout=args.dropout,
                )

        # Decoder
        if self.task_name == "pretrain":
            self.denoising_patch_decoder = (
                DenoisingPatchDecoder(
                    d_model=args.d_model,
                    num_layers=args.d_layers,
                    num_heads=args.n_heads,
                    feedforward_dim=args.d_ff,
                    dropout=args.dropout,
                    mask_ratio=args.mask_ratio,
                    use_prompt_adaln=(
                        self.use_prompt_adaln
                    ),
                    prompt_dim=self.prompt_dim,
                )
            )

            self.projection = FlattenHead(
                seq_len=self.seq_len,
                d_model=self.d_model,
                pred_len=args.input_len,
                dropout=args.head_dropout,
            )

            # self.projection = ARFlattenHead(
            #     d_model=self.d_model,
            #     patch_len=self.patch_len,
            #     dropout=args.head_dropout,
            # )

        elif self.task_name == "finetune":
            self.head = FlattenHead(
                seq_len=self.seq_len,
                d_model=args.d_model,
                pred_len=args.pred_len,
                dropout=args.head_dropout,
            )

            if (
                self.residual_forecast
                and self.zero_init_residual_head
            ):
                # The initial forecast is exactly persistence.
                # The head then learns only the correction
                # relative to the last observation.
                nn.init.zeros_(
                    self.head.forecast_head.weight
                )
                nn.init.zeros_(
                    self.head.forecast_head.bias
                )

    def _build_prompt(
        self,
        x_out,
        label_source,
        batch_size,
        num_features,
    ):
        if getattr(
            self,
            "use_soft_prompt",
            False,
        ):
            target_index = (
                self.regime_target_index
                % num_features
            )

            target_hidden = x_out.reshape(
                batch_size,
                num_features,
                x_out.size(1),
                x_out.size(2),
            )[:, target_index]

            regime_logits, regime_probs = (
                self.regime_predictor(
                    target_hidden
                )
            )

            sample_prompt = (
                self.soft_prompt_generator(
                    regime_probs
                )
            )

            # Channel-independent encoder rows belonging
            # to the same sample receive the same
            # power-regime prompt.
            soft_prompt = (
                sample_prompt.repeat_interleave(
                    num_features,
                    dim=0,
                )
            )

            pseudo_labels = (
                compute_regime_pseudo_labels_from_series(
                    label_source[
                        :,
                        :,
                        target_index,
                    ],
                    stable_thresh=(
                        self.regime_stable_thresh
                    ),
                    ramp_thresh=(
                        self.regime_ramp_thresh
                    ),
                )
            )

            return (
                soft_prompt,
                regime_logits,
                pseudo_labels,
            )

        prompt_emb = (
            self.prompt_encoder(x_out)
            if (
                getattr(
                    self,
                    "use_prompt_adaln",
                    False,
                )
                and hasattr(
                    self,
                    "prompt_encoder",
                )
            )
            else None
        )

        return prompt_emb, None, None

    def pretrain(self, x):
        # [batch_size, input_len, num_features]
        batch_size, input_len, num_features = (
            x.size()
        )
        label_source = x

        if self.use_norm:
            # Instance Normalization
            means = torch.mean(
                x,
                dim=1,
                keepdim=True,
            ).detach()

            x = x - means

            stdevs = torch.sqrt(
                torch.var(
                    x,
                    dim=1,
                    keepdim=True,
                    unbiased=False,
                )
                + 1e-5
            ).detach()

            x = x / stdevs

        # Channel Independence
        x = self.channel_independence(x)

        # Patch
        x_patch = self.patch(x)

        # For Causal Transformer
        x_embedding = self.enc_embedding(
            x_patch
        )

        x_embedding_bias = (
            self.add_sos_token_and_drop_last(
                x_embedding
            )
        )

        x_embedding_bias = (
            self.positional_encoding(
                x_embedding_bias
            )
        )

        x_out = self.encoder(
            x_embedding_bias,
            is_mask=True,
        )

        # Noising Diffusion
        noise_x_patch, noise, t = (
            self.diffusion(x_patch)
        )

        noise_x_embedding = (
            self.enc_embedding(
                noise_x_patch
            )
        )

        noise_x_embedding = (
            self.positional_encoding(
                noise_x_embedding
            )
        )

        # For Denoising Patch Decoder
        (
            prompt_emb,
            regime_logits,
            pseudo_labels,
        ) = self._build_prompt(
            x_out,
            label_source,
            batch_size,
            num_features,
        )

        predict_x = (
            self.denoising_patch_decoder(
                query=noise_x_embedding,
                key=x_out,
                value=x_out,
                is_tgt_mask=True,
                is_src_mask=True,
                prompt_emb=prompt_emb,
            )
        )

        # For Decoder
        predict_x = predict_x.reshape(
            batch_size,
            num_features,
            -1,
            self.d_model,
        )

        predict_x = self.projection(
            predict_x
        )

        # Instance Denormalization
        if self.use_norm:
            predict_x = (
                predict_x
                * (
                    stdevs[
                        :,
                        0,
                        :,
                    ].unsqueeze(1)
                ).repeat(
                    1,
                    input_len,
                    1,
                )
            )

            predict_x = (
                predict_x
                + (
                    means[
                        :,
                        0,
                        :,
                    ].unsqueeze(1)
                ).repeat(
                    1,
                    input_len,
                    1,
                )
            )

        if self.use_soft_prompt:
            return {
                "pred": predict_x,
                "regime_logits": regime_logits,
                "pseudo_labels": pseudo_labels,
            }

        return predict_x

    def forecast(self, x):
        batch_size, _, num_features = (
            x.size()
        )
        label_source = x

        last_observation = (
            x[:, -1:, :].detach()
        )

        if self.use_norm:
            means = torch.mean(
                x,
                dim=1,
                keepdim=True,
            ).detach()

            x = x - means

            stdevs = torch.sqrt(
                torch.var(
                    x,
                    dim=1,
                    keepdim=True,
                    unbiased=False,
                )
                + 1e-5
            ).detach()

            x = x / stdevs

        x = self.channel_independence(x)
        x = self.patch(x)
        x = self.enc_embedding(x)
        x = self.positional_encoding(x)

        x = self.encoder(
            x,
            is_mask=False,
        )

        if self.use_soft_prompt:
            prompt_emb, _, _ = (
                self._build_prompt(
                    x,
                    label_source,
                    batch_size,
                    num_features,
                )
            )
            x = x + prompt_emb.unsqueeze(1)

        x = x.reshape(
            batch_size,
            num_features,
            -1,
            self.d_model,
        )

        # Forecast
        x = self.head(x)

        # Denormalise either as an absolute forecast
        # or as a correction to the persistence baseline.
        # In residual mode the mean must not be added:
        # the last observation is already on the
        # original input scale.
        if self.residual_forecast:
            if self.use_norm:
                x = (
                    x
                    * stdevs[
                        :,
                        0,
                        :,
                    ].unsqueeze(1)
                )

            x = last_observation + x

        elif self.use_norm:
            x = (
                x
                * stdevs[
                    :,
                    0,
                    :,
                ].unsqueeze(1)
            )
            x = (
                x
                + means[
                    :,
                    0,
                    :,
                ].unsqueeze(1)
            )

        return x

    def forward(self, batch_x):
        if self.task_name == "pretrain":
            return self.pretrain(batch_x)

        elif self.task_name == "finetune":
            dec_out = self.forecast(batch_x)
            return dec_out[
                :,
                -self.pred_len:,
                :,
            ]

        else:
            raise ValueError(
                "task_name should be "
                "'pretrain' or 'finetune'"
            )


class ClsModel(nn.Module):
    def __init__(self, args):
        super(ClsModel, self).__init__()
        self.input_len = args.input_len

        # For Model Hyperparameters
        self.d_model = args.d_model
        self.num_heads = args.n_heads
        self.feedforward_dim = args.d_ff
        self.dropout = args.dropout
        self.device = args.device
        self.task_name = args.task_name
        self.num_classes = args.num_classes

        # Patch
        self.patch_len = args.patch_len
        self.stride = args.stride

        nominal_len = (
            int(
                (
                    self.input_len
                    - self.patch_len
                )
                / self.stride
            )
            + 2
        )

        padding = max(
            0,
            nominal_len * self.stride
            - self.input_len,
        )

        self.seq_len = (
            (
                self.input_len
                + 2 * padding
                - self.patch_len
            )
            // self.stride
        ) + 1

        # Embedding
        self.enc_embedding = ClsEmbedding(
            num_features=args.enc_in,
            d_model=args.d_model,
            kernel_size=args.patch_len,
            stride=args.stride,
            padding=padding,
        )

        self.positional_encoding = (
            PositionalEncoding(
                d_model=self.d_model,
                dropout=self.dropout,
            )
        )

        sos_token = torch.randn(
            1,
            1,
            self.d_model,
            device=self.device,
        )

        self.sos_token = nn.Parameter(
            sos_token,
            requires_grad=True,
        )

        self.add_sos_token_and_drop_last = (
            AddSosTokenAndDropLast(
                sos_token=self.sos_token,
            )
        )

        # Encoder (Causal Transformer)
        self.diffusion = Diffusion(
            time_steps=args.time_steps,
            device=self.device,
            scheduler=args.scheduler,
        )

        self.encoder = CausalTransformer(
            d_model=args.d_model,
            num_heads=args.n_heads,
            feedforward_dim=args.d_ff,
            dropout=args.dropout,
            num_layers=args.e_layers,
        )

        # self.encoder = DilatedConvEncoder(
        #     in_channels=self.d_model,
        #     channels=[self.d_model] * args.e_layers,
        #     kernel_size=3,
        # )

        # Decoder
        if self.task_name == "pretrain":
            self.use_soft_prompt = getattr(
                args,
                "use_soft_prompt",
                False,
            )

            self.use_prompt_adaln = (
                getattr(
                    args,
                    "use_prompt_adaln",
                    False,
                )
                or self.use_soft_prompt
            )

            self.regime_stable_thresh = getattr(
                args,
                "regime_stable_thresh",
                0.15,
            )
            self.regime_ramp_thresh = getattr(
                args,
                "regime_ramp_thresh",
                0.25,
            )

            if self.use_soft_prompt:
                self.num_modes = getattr(
                    args,
                    "num_modes",
                    3,
                )
                self.prompt_dim = args.d_model

                self.regime_predictor = (
                    RegimePredictor(
                        d_model=args.d_model,
                        num_modes=self.num_modes,
                        dropout=args.dropout,
                    )
                )

                self.soft_prompt_generator = (
                    SoftPromptGenerator(
                        num_modes=self.num_modes,
                        d_model=args.d_model,
                    )
                )

            else:
                self.prompt_dim = (
                    args.prompt_dim
                    if args.prompt_dim
                    is not None
                    else args.d_model
                )

                if self.use_prompt_adaln:
                    self.prompt_encoder = (
                        PromptEncoder(
                            d_model=args.d_model,
                            prompt_dim=(
                                self.prompt_dim
                            ),
                            dropout=args.dropout,
                        )
                    )

            self.denoising_patch_decoder = (
                DenoisingPatchDecoder(
                    d_model=args.d_model,
                    num_layers=args.d_layers,
                    num_heads=args.n_heads,
                    feedforward_dim=args.d_ff,
                    dropout=args.dropout,
                    mask_ratio=args.mask_ratio,
                    use_prompt_adaln=(
                        self.use_prompt_adaln
                    ),
                    prompt_dim=self.prompt_dim,
                )
            )

            self.projection = ClsFlattenHead(
                seq_len=self.seq_len,
                d_model=self.d_model,
                pred_len=args.input_len,
                num_features=args.c_out,
                dropout=args.head_dropout,
            )

        elif self.task_name == "finetune":
            self.head = OldClsHead(
                seq_len=self.seq_len,
                d_model=args.d_model,
                num_classes=args.num_classes,
                dropout=args.head_dropout,
            )

    def _build_prompt(
        self,
        x_out,
        label_source,
    ):
        if getattr(
            self,
            "use_soft_prompt",
            False,
        ):
            (
                regime_logits,
                regime_probs,
            ) = self.regime_predictor(
                x_out
            )

            soft_prompt = (
                self.soft_prompt_generator(
                    regime_probs
                )
            )

            target_index = (
                getattr(
                    self,
                    "regime_target_index",
                    -1,
                )
                % label_source.size(-1)
            )

            pseudo_labels = (
                compute_regime_pseudo_labels_from_series(
                    label_source[
                        :,
                        :,
                        target_index,
                    ],
                    stable_thresh=(
                        self.regime_stable_thresh
                    ),
                    ramp_thresh=(
                        self.regime_ramp_thresh
                    ),
                )
            )

            return (
                soft_prompt,
                regime_logits,
                pseudo_labels,
            )

        prompt_emb = (
            self.prompt_encoder(x_out)
            if (
                getattr(
                    self,
                    "use_prompt_adaln",
                    False,
                )
                and hasattr(
                    self,
                    "prompt_encoder",
                )
            )
            else None
        )

        return prompt_emb, None, None

    def pretrain(self, x):
        # [batch_size, input_len, num_features]
        batch_input = x

        # batch_size, input_len, num_features = x.size()
        # means = torch.mean(
        #     x, dim=1, keepdim=True
        # ).detach()
        # x = x - means
        # stdevs = torch.sqrt(
        #     torch.var(
        #         x,
        #         dim=1,
        #         keepdim=True,
        #         unbiased=False,
        #     )
        #     + 1e-5
        # ).detach()
        # x = x / stdevs

        # For Causal Transformer
        x_embedding = self.enc_embedding(x)

        x_embedding_bias = (
            self.add_sos_token_and_drop_last(
                x_embedding
            )
        )

        x_embedding_bias = (
            self.positional_encoding(
                x_embedding_bias
            )
        )

        x_out = self.encoder(
            x_embedding_bias,
            is_mask=True,
        )

        # Noising Diffusion
        noise_x_patch, noise, t = (
            self.diffusion(x)
        )

        noise_x_embedding = (
            self.enc_embedding(
                noise_x_patch
            )
        )

        noise_x_embedding = (
            self.positional_encoding(
                noise_x_embedding
            )
        )

        # For Denoising Patch Decoder
        (
            prompt_emb,
            regime_logits,
            pseudo_labels,
        ) = self._build_prompt(
            x_out,
            batch_input,
        )

        predict_x = (
            self.denoising_patch_decoder(
                query=noise_x_embedding,
                key=x_out,
                value=x_out,
                is_tgt_mask=True,
                is_src_mask=True,
                prompt_emb=prompt_emb,
            )
        )

        # For Decoder
        predict_x = self.projection(
            predict_x
        )

        # Instance Denormalization
        # predict_x = predict_x * (
        #     stdevs[:, 0, :].unsqueeze(1)
        # ).repeat(
        #     1,
        #     input_len,
        #     1,
        # )
        #
        # predict_x = predict_x + (
        #     means[:, 0, :].unsqueeze(1)
        # ).repeat(
        #     1,
        #     input_len,
        #     1,
        # )

        if getattr(
            self,
            "use_soft_prompt",
            False,
        ):
            return {
                "pred": predict_x,
                "regime_logits": regime_logits,
                "pseudo_labels": pseudo_labels,
            }

        return predict_x

    def forecast(self, x):
        # batch_size, _, num_features = x.size()
        # means = torch.mean(
        #     x,
        #     dim=1,
        #     keepdim=True,
        # ).detach()
        # x = x - means
        # stdevs = torch.sqrt(
        #     torch.var(
        #         x,
        #         dim=1,
        #         keepdim=True,
        #         unbiased=False,
        #     )
        #     + 1e-5
        # ).detach()
        # x = x / stdevs

        x = self.enc_embedding(x)
        x = self.positional_encoding(x)

        x = self.encoder(
            x,
            is_mask=False,
        )

        # Forecast
        x = self.head(x)

        return x

    def forward(self, batch_x):
        if self.task_name == "pretrain":
            return self.pretrain(batch_x)

        elif self.task_name == "finetune":
            return self.forecast(batch_x)

        else:
            raise ValueError(
                "task_name should be "
                "'pretrain' or 'finetune'"
            )


class PromptGuidedModel(Model):
    """
    TimeDART with regime predictor, soft prompt,
    and AdaLN denoising decoder.
    """

    def __init__(self, args):
        args.use_soft_prompt = True
        args.use_prompt_adaln = True
        super().__init__(args)

    def forward(self, batch_x):
        if self.task_name == "pretrain":
            result = self.pretrain(batch_x)

            if isinstance(result, dict):
                return (
                    result["pred"],
                    result["regime_logits"],
                    result["pseudo_labels"],
                )

            return result

        return super().forward(batch_x)


class PromptGuidedClsModel(ClsModel):
    """
    Classification variant with
    prompt-guided pretraining.
    """

    def __init__(self, args):
        args.use_soft_prompt = True
        args.use_prompt_adaln = True
        super().__init__(args)

    def forward(self, batch_x):
        if self.task_name == "pretrain":
            result = self.pretrain(batch_x)

            if isinstance(result, dict):
                return (
                    result["pred"],
                    result["regime_logits"],
                    result["pseudo_labels"],
                )

            return result

        return super().forward(batch_x)