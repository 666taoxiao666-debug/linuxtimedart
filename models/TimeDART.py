import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.TimeDART_EncDec import (
    ChannelIndependence,
    ChannelMixer,
    AddSosTokenAndDropLast,
    CausalTransformer,
    Diffusion,
    DenoisingPatchDecoder,
    PromptEncoder,
    RegimePredictor,
    SoftPromptGenerator,
    SceneWikiPromptRouter,
    CompositionalEventWikiRouter,
    DilatedConvEncoder,
    ClsEmbedding,
    ClsHead,
    OldClsHead,
    ClsFlattenHead,
    ARFlattenHead,
)
from layers.Embed import Patch, PatchEmbedding, PositionalEncoding, TokenEmbedding_TimeDART
from utils.regime_labels import compute_regime_pseudo_labels_from_series
from utils.wind_regime_wiki import (
    compute_event_factor_rule_logits,
    compute_scene_wiki_rule_logits,
    load_wind_regime_wiki_bundle,
)
from data_provider.sdwpf_features import (
    channel_prior_vector,
    operating_context_dim,
    revin_keep_indices,
)


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
        self.residual_gate_init = float(getattr(args, "residual_gate_init", -4.0))
        self.residual_gate_logit = (
            nn.Parameter(torch.tensor(self.residual_gate_init, dtype=torch.float32))
            if self.residual_forecast
            else None
        )
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
        self.regime_label_method = getattr(
            args,
            "regime_label_method",
            "legacy_volatility",
        )
        self.prompt_router = getattr(args, "prompt_router", "trend")
        self.features = getattr(args, "features", "M")
        self.enc_in = int(getattr(args, "enc_in", 1))
        self.feature_columns = list(getattr(args, "feature_columns", None) or [])
        if not self.feature_columns:
            self.feature_columns = [f"f{i}" for i in range(self.enc_in)]
        if len(self.feature_columns) != self.enc_in:
            raise ValueError(
                "feature_columns length must equal enc_in: "
                f"{len(self.feature_columns)} != {self.enc_in}"
            )
        self.mix_channels = bool(getattr(args, "mix_channels", False)) and (
            self.features == "MS"
        )
        self.use_channel_prior = bool(getattr(args, "channel_prior", False))
        self.use_op_context = bool(getattr(args, "op_context", False))
        self.revin_keep_indices = (
            revin_keep_indices(self.feature_columns)
            if bool(getattr(args, "revin_keep_wind", False))
            else []
        )
        self.op_context_dim = (
            operating_context_dim(self.feature_columns) if self.use_op_context else 0
        )
        self.channel_independence = ChannelIndependence()
        self.channel_mixer = (
            ChannelMixer(
                num_features=self.enc_in,
                d_model=self.d_model,
                dropout=self.dropout,
                channel_prior=channel_prior_vector(
                    self.feature_columns,
                    physics_init=self.use_channel_prior,
                ),
                context_dim=self.op_context_dim,
            )
            if self.mix_channels and self.task_name == "finetune"
            else None
        )

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
            if self.prompt_router in (
                "trend",
                "hybrid_wiki",
                "compositional_wiki",
            ) and self.num_modes != 3:
                raise ValueError(
                    "The pseudo-label definition requires "
                    "num_modes=3"
                )

            self.prompt_dim = args.d_model
            if self.prompt_router in ("trend", "hybrid_wiki", "compositional_wiki"):
                # This is the project's original train-calibrated prompt branch.
                # In the hybrid model it remains the base representation rather
                # than being replaced by a generic semantic retriever.
                self.regime_predictor = RegimePredictor(
                    d_model=args.d_model,
                    num_modes=self.num_modes,
                    dropout=args.dropout,
                )
                self.soft_prompt_generator = SoftPromptGenerator(
                    num_modes=self.num_modes,
                    d_model=args.d_model,
                )
                down = getattr(args, "regime_down_thresh", None)
                up = getattr(args, "regime_up_thresh", None)
                self.register_buffer(
                    "regime_down_thresh",
                    torch.tensor(float("nan") if down is None else float(down)),
                )
                self.register_buffer(
                    "regime_up_thresh",
                    torch.tensor(float("nan") if up is None else float(up)),
                )

            if self.prompt_router in (
                "scene_wiki",
                "hybrid_wiki",
                "compositional_wiki",
            ):
                bundle = load_wind_regime_wiki_bundle(
                    args.scene_wiki_embeddings,
                    expected_scene_ids=getattr(args, "scene_wiki_scene_ids", None),
                )
                if self.prompt_router == "scene_wiki" and self.num_modes != len(
                    bundle["scene_ids"]
                ):
                    raise ValueError(
                        "num_modes must equal the Wiki scene count: "
                        f"{self.num_modes} != {len(bundle['scene_ids'])}"
                    )
                if self.prompt_router == "hybrid_wiki" and tuple(
                    bundle["scene_ids"]
                )[:1] != ("no_exception",):
                    raise ValueError(
                        "hybrid_wiki requires an exception Wiki whose first scene is "
                        "no_exception"
                    )
                self.scene_wiki_scene_ids = tuple(bundle["scene_ids"])
                self.scene_wiki_num_modes = len(self.scene_wiki_scene_ids)
                self.scene_wiki_config_sha256 = args.scene_wiki_config_sha256
                self.scene_wiki_bundle_sha256 = bundle["sha256"]
                self.scene_wiki_encoder_name = bundle["encoder_name"]
                router_kwargs = {
                    "d_model": args.d_model,
                    "num_features": self.enc_in,
                    "channel_prior": channel_prior_vector(
                        self.feature_columns,
                        physics_init=True,
                    ),
                    "top_k": args.scene_wiki_top_k,
                    "temperature": args.scene_wiki_temperature,
                    "rule_weight": args.scene_wiki_rule_weight,
                    "prompt_gate_init": args.scene_wiki_prompt_gate_init,
                    "dropout": args.dropout,
                }
                if self.prompt_router == "compositional_wiki":
                    self.scene_wiki_router = CompositionalEventWikiRouter(
                        bundle["embeddings"],
                        factor_reliability=bundle["factor_reliability"],
                        activation_threshold=args.scene_wiki_activation_threshold,
                        confidence_power=args.scene_wiki_confidence_power,
                        **router_kwargs,
                    )
                else:
                    self.scene_wiki_router = SceneWikiPromptRouter(
                        bundle["embeddings"],
                        null_scene_index=(
                            0 if self.prompt_router == "hybrid_wiki" else None
                        ),
                        **router_kwargs,
                    )
                self.register_buffer(
                    "scene_scaler_mean",
                    torch.full((self.enc_in,), float("nan")),
                )
                self.register_buffer(
                    "scene_scaler_scale",
                    torch.full((self.enc_in,), float("nan")),
                )
                self.scene_wiki_rule_kwargs = {
                    "recent_steps": int(args.scene_wiki_recent_steps),
                    "low_wind_max_mps": float(args.scene_wiki_low_wind_max_mps),
                    "active_wind_min_mps": float(args.scene_wiki_active_wind_min_mps),
                    "low_power_max_ratio": float(args.scene_wiki_low_power_max_ratio),
                    "rated_power_min_ratio": float(args.scene_wiki_rated_power_min_ratio),
                    "gust_std_min_mps": float(args.scene_wiki_gust_std_min_mps),
                    "gust_step_min_mps": float(args.scene_wiki_gust_step_min_mps),
                }
                if self.prompt_router == "compositional_wiki":
                    self.scene_wiki_rule_kwargs.update(
                        {
                            "factor_rules": list(args.scene_wiki_factor_rules),
                            "rule_defaults": dict(args.scene_wiki_rule_defaults),
                        }
                    )
                if self.prompt_router != "compositional_wiki":
                    self.scene_wiki_rule_kwargs["ramp_delta_min_ratio"] = float(
                        args.scene_wiki_ramp_delta_min_ratio
                    )
                self.rated_power = float(args.rated_power)
            if self.prompt_router not in (
                "trend",
                "scene_wiki",
                "hybrid_wiki",
                "compositional_wiki",
            ):
                raise ValueError(f"Unknown prompt router: {self.prompt_router}")
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
                # A zero residual head makes the initial forecast exactly
                # persistence, including when channel mixing is enabled. The
                # head learns first; subsequent steps also update the mixer.
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

            channel_hidden = x_out.reshape(
                batch_size,
                num_features,
                x_out.size(1),
                x_out.size(2),
            )
            target_hidden = channel_hidden[:, target_index]

            scene_logits = None
            scene_labels = None
            event_activations = None
            if self.prompt_router in (
                "scene_wiki",
                "hybrid_wiki",
                "compositional_wiki",
            ):
                if not torch.isfinite(self.scene_scaler_mean).all() or not torch.isfinite(
                    self.scene_scaler_scale
                ).all():
                    raise RuntimeError(
                        "Scene Wiki scaler statistics are unset. Pretrain with a train-only "
                        "SDWPF split or load its matched pretraining checkpoint."
                    )
                raw_history = (
                    label_source * self.scene_scaler_scale.view(1, 1, -1)
                    + self.scene_scaler_mean.view(1, 1, -1)
                )
                if self.prompt_router == "compositional_wiki":
                    rule_logits, scene_labels = compute_event_factor_rule_logits(
                        raw_history,
                        self.feature_columns,
                        rated_power=self.rated_power,
                        factor_ids=self.scene_wiki_scene_ids,
                        **self.scene_wiki_rule_kwargs,
                    )
                    (
                        wiki_prompt,
                        scene_logits,
                        scene_probs,
                        event_activations,
                    ) = self.scene_wiki_router(channel_hidden, rule_logits=rule_logits)
                else:
                    rule_logits, scene_labels = compute_scene_wiki_rule_logits(
                        raw_history,
                        self.feature_columns,
                        rated_power=self.rated_power,
                        scene_ids=self.scene_wiki_scene_ids,
                        **self.scene_wiki_rule_kwargs,
                    )
                    wiki_prompt, scene_logits, scene_probs = self.scene_wiki_router(
                        channel_hidden,
                        rule_logits=rule_logits,
                    )

            if self.prompt_router in ("trend", "hybrid_wiki", "compositional_wiki"):
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
                if self.prompt_router == "hybrid_wiki":
                    # Causal Trend-Wiki Residual Prompting (CTWRP): the original
                    # trend prompt explains ordinary stable/up/down dynamics.
                    # Wiki semantics can only add a residual when the history
                    # supports an exception and retrieval is confident.
                    event_probability = 1.0 - scene_probs[:, 0]
                    if self.scene_wiki_router.top_k <= 1:
                        retrieval_confidence = torch.ones_like(event_probability)
                    else:
                        entropy = -(
                            scene_probs.clamp_min(1e-8)
                            * scene_probs.clamp_min(1e-8).log()
                        ).sum(dim=-1)
                        normalizer = torch.log(
                            torch.tensor(
                                float(self.scene_wiki_router.top_k),
                                device=entropy.device,
                                dtype=entropy.dtype,
                            )
                        )
                        retrieval_confidence = (1.0 - entropy / normalizer).clamp(0.0, 1.0)
                    self._last_wiki_intervention = (
                        event_probability
                        * retrieval_confidence
                        * torch.sigmoid(self.scene_wiki_router.prompt_gate_logit)
                    ).detach()
                    sample_prompt = sample_prompt + retrieval_confidence.unsqueeze(-1) * wiki_prompt
                elif self.prompt_router == "compositional_wiki":
                    # The router already applies physical support, confidence
                    # thresholding, Top-K sparsity and the learnable residual gate.
                    # If every factor abstains, wiki_prompt is exactly zero.
                    sample_prompt = sample_prompt + wiki_prompt
                    self._last_wiki_intervention = (
                        event_activations.amax(dim=-1)
                        * torch.sigmoid(self.scene_wiki_router.prompt_gate_logit)
                    ).detach()
                    self._last_wiki_factor_activations = event_activations.detach()
            else:
                regime_logits = scene_logits
                regime_probs = scene_probs
                sample_prompt = wiki_prompt

            # Channel-independent encoder rows belonging
            # to the same sample receive the same
            # power-regime prompt.
            soft_prompt = (
                sample_prompt.repeat_interleave(
                    num_features,
                    dim=0,
                )
            )

            # Pseudo-labels supervise the regime classifier during pretraining
            # only. Forecast fine-tuning consumes the predictor probabilities
            # directly, so forcing label construction here would make an
            # intentionally random-init ablation depend on pretrain thresholds.
            pseudo_labels = None
            if self.task_name == "pretrain":
                if self.prompt_router == "scene_wiki":
                    pseudo_labels = scene_labels
                else:
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
                            method=self.regime_label_method,
                            down_thresh=self.regime_down_thresh,
                            up_thresh=self.regime_up_thresh,
                        )
                    )

            return (
                soft_prompt,
                regime_logits,
                pseudo_labels,
                scene_logits
                if self.prompt_router in ("hybrid_wiki", "compositional_wiki")
                else None,
                scene_labels
                if self.prompt_router in ("hybrid_wiki", "compositional_wiki")
                else None,
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

        return prompt_emb, None, None, None, None

    def set_scene_scaler(self, mean, scale):
        """Install scaler statistics fitted on the training split only."""

        if self.prompt_router not in (
            "scene_wiki",
            "hybrid_wiki",
            "compositional_wiki",
        ):
            return
        mean_tensor = torch.as_tensor(
            mean, dtype=self.scene_scaler_mean.dtype, device=self.scene_scaler_mean.device
        ).reshape(-1)
        scale_tensor = torch.as_tensor(
            scale, dtype=self.scene_scaler_scale.dtype, device=self.scene_scaler_scale.device
        ).reshape(-1)
        if mean_tensor.numel() != self.enc_in or scale_tensor.numel() != self.enc_in:
            raise ValueError("Scene Wiki scaler width must equal enc_in")
        if not torch.isfinite(mean_tensor).all() or not torch.isfinite(scale_tensor).all():
            raise ValueError("Scene Wiki scaler statistics must be finite")
        if torch.any(scale_tensor <= 0):
            raise ValueError("Scene Wiki scaler scales must be positive")
        with torch.no_grad():
            self.scene_scaler_mean.copy_(mean_tensor)
            self.scene_scaler_scale.copy_(scale_tensor)

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
            scene_logits,
            scene_labels,
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
            result = {
                "pred": predict_x,
                "regime_logits": regime_logits,
                "pseudo_labels": pseudo_labels,
                "scene_logits": (
                    scene_logits if self.prompt_router == "hybrid_wiki" else None
                ),
                "scene_labels": (
                    scene_labels if self.prompt_router == "hybrid_wiki" else None
                ),
            }
            if self.prompt_router == "compositional_wiki":
                result["event_logits"] = scene_logits
                result["event_targets"] = scene_labels
            return result

        return predict_x

    def _operating_context(self, x):
        """Pre-norm last/mean wind and last power/yaw/pitch.  History only."""
        names = {name: i for i, name in enumerate(self.feature_columns)}
        pieces = []
        if "Wspd" in names:
            idx = names["Wspd"]
            pieces.append(x[:, -1, idx])
            pieces.append(x[:, :, idx].mean(dim=1))
        for name in ("power", "yaw_sin", "yaw_cos", "Pab_mean"):
            if name in names:
                pieces.append(x[:, -1, names[name]])
        if not pieces:
            return None
        return torch.stack(pieces, dim=-1)

    def forecast(self, x):
        batch_size, _, num_features = (
            x.size()
        )
        label_source = x

        last_observation = (
            x[:, -1:, :].detach()
        )
        context = (
            self._operating_context(x)
            if self.channel_mixer is not None and self.op_context_dim > 0
            else None
        )

        if self.use_norm:
            x_raw = x
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
            if self.revin_keep_indices:
                x = x.clone()
                x[:, :, self.revin_keep_indices] = x_raw[:, :, self.revin_keep_indices]

        x = self.channel_independence(x)
        x = self.patch(x)
        x = self.enc_embedding(x)
        x = self.positional_encoding(x)

        x = self.encoder(
            x,
            is_mask=False,
        )

        if self.use_soft_prompt:
            prompt_emb, _, _, _, _ = (
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

        if self.channel_mixer is not None:
            x = self.channel_mixer(x, context=context)

        # Forecast
        x = self.head(x)

        # Denormalise either as an absolute forecast
        # or as a correction to the persistence baseline.
        # In residual mode the mean must not be added:
        # the last observation is already on the
        # original input scale.
        power_slice = slice(-1, None) if self.mix_channels else slice(None)
        if self.residual_forecast:
            if self.use_norm:
                x = x * stdevs[:, :, power_slice]
            # Start close to the hard-to-beat persistence baseline.  Unlike a
            # zeroed head, this small learnable gate still lets gradients reach
            # the wind/context path from the first optimisation step.
            residual_gate = torch.sigmoid(self.residual_gate_logit)
            x = last_observation[:, :, power_slice] + residual_gate * x

        elif self.use_norm:
            x = x * stdevs[:, :, power_slice]
            x = x + means[:, :, power_slice]

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
            self.regime_label_method = getattr(
                args,
                "regime_label_method",
                "legacy_volatility",
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
                down = getattr(args, "regime_down_thresh", None)
                up = getattr(args, "regime_up_thresh", None)
                self.register_buffer(
                    "regime_down_thresh",
                    torch.tensor(float("nan") if down is None else float(down)),
                )
                self.register_buffer(
                    "regime_up_thresh",
                    torch.tensor(float("nan") if up is None else float(up)),
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

            pseudo_labels = None
            if self.task_name == "pretrain":
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
                        method=self.regime_label_method,
                        down_thresh=self.regime_down_thresh,
                        up_thresh=self.regime_up_thresh,
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
        prompt_enabled = not bool(getattr(args, "disable_regime_prompt", False))
        args.use_soft_prompt = prompt_enabled
        args.use_prompt_adaln = prompt_enabled
        super().__init__(args)

    def forward(self, batch_x):
        if self.task_name == "pretrain":
            result = self.pretrain(batch_x)

            if isinstance(result, dict):
                if self.prompt_router in ("hybrid_wiki", "compositional_wiki"):
                    return result
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
        prompt_enabled = not bool(getattr(args, "disable_regime_prompt", False))
        args.use_soft_prompt = prompt_enabled
        args.use_prompt_adaln = prompt_enabled
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
