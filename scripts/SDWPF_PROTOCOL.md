# SDWPF paper protocol

This protocol keeps the final 20% of timestamps sealed until all model and
hyper-parameter choices are frozen.

## Physically Composable Event-Wiki Residual Prompting (PCE-WRP)

The proposed PromptTimeDART route is `compositional_wiki`. It preserves the
three train-quantile-calibrated trend prompts (stable, ramp-up and ramp-down)
as the ordinary temporal prior. A separate event-factor Wiki in
`configs/wind_event_factor_wiki.json` contains four observable factors:
gust/turbulence, high-wind low-power, rated saturation and low-wind idle. These
are independent labels rather than one mutually exclusive scene, so, for
example, gust and rated saturation can be active in the same historical
window. High-wind low-power is deliberately not called curtailment because
SCADA history alone cannot identify its cause.

For each historical window, the final prompt is
`trend_prompt + sum(active_factor_residuals)`. A factor is active only when
both its history-only physical rule has positive support and its learned
semantic probability exceeds the frozen activation threshold. Top-K keeps the
composition sparse; confidence controls each residual's magnitude. If no
factor passes both gates, the event residual is mathematically exactly zero,
so the Wiki cannot perturb an ordinary window. Independent BCE supervision
trains the four semantic factors from train-only pseudo-labels, with bounded
train-only imbalance weights.

On the first run, the entry script uses the frozen model selected by
`WIKI_LLM_PATH` to encode the four versioned event-factor cards once. It writes
`outputs/wiki/wind_event_factor_wiki_qwen.npz`; no LLM is called inside
training or inference. The time-series encoder then retrieves and composes
those frozen semantic anchors using historical SCADA only. Train-split scaler
statistics, Wiki/config hashes, factor co-occurrence, null-window support,
activation statistics and prompt parameters are saved in the audit artifacts.

To build the semantic anchors explicitly (optional, because training scripts
do this automatically when the bundle is absent):

```bash
python scripts/build_wind_regime_wiki.py \
  --config configs/wind_event_factor_wiki.json \
  --output outputs/wiki/wind_event_factor_wiki_qwen.npz \
  --llm_path Qwen/Qwen2.5-0.5B
```

Required mechanism controls are separate complete runs (with matched
pretraining and fine-tuning checkpoints):

```bash
PROMPT_ROUTER=trend REGIME_LABEL_METHOD=trend_quantile \
bash scripts/train/SDWPF_paper_cv.sh
```

`PROMPT_ROUTER=hybrid_wiki` retains the old single-label five-card exception
Wiki, while `PROMPT_ROUTER=scene_wiki` retains the seven-card "Wiki replaces
trend prompts" route. They are ablations, not aliases for the proposed method.
Do not mix their checkpoints: every router requires fold/seed-matched
pretraining followed by matched fine-tuning.

Novelty boundary: Top-K semantic-anchor retrieval itself is prior art, and so
is generic hard/soft prompt fusion. The hypothesis tested here is narrower:
an ordinary trend prompt should remain authoritative for common dynamics,
while multiple observable physical factors contribute sparse,
confidence-gated semantic residuals and abstain exactly on unsupported
windows. This must be supported by trend vs legacy single-label vs
compositional routing, single-factor vs multi-factor composition, no-gate,
no-abstention and random-embedding controls. Report per-factor performance,
co-occurrence performance, null-window false intervention and forecast skill;
otherwise the mechanism is not established.

## Log layout

SDWPF entry scripts automatically allocate one searchable directory per user
invocation:

```text
outputs/logs/SDWPF/
├── latest.txt
└── YYYYMMDD/
    ├── index.tsv
    ├── latest.txt
    └── 001_<task>_<key-parameters>/
        ├── log_meta.json
        ├── status.env
        ├── *.env
        ├── *.log
        ├── *summary.txt
        ├── artifacts/        # final-eval metrics, arrays, PNG/PDF figures
        └── tb/
```

The three-digit prefix is the run number for that date. CV, final-training,
fold and ablation-suite scripts use one numbered parent directory and place
their fold/seed jobs below `runs/`. Existing callers may still override the
automatic layout with `SDWPF_LOG_DIR` or `SDWPF_LOG_FILE`.
Long visible parameter strings are shortened with a stable hash to stay below
common filesystem limits; `index.tsv`, `log_meta.json`, and the stage `.env`
files always retain the complete values.

Locate the newest run and its compact result with:

```bash
LATEST="$(cat outputs/logs/SDWPF/latest.txt)"
echo "outputs/logs/SDWPF/${LATEST}"
find "outputs/logs/SDWPF/${LATEST}" -maxdepth 3 -name '*summary.txt' -print
```

Browse every run created today with:

```bash
column -t -s $'\t' "outputs/logs/SDWPF/$(date +%Y%m%d)/index.tsv"
```

Before pushing or starting a server run, execute the fast regression suite:

```bash
python -m unittest discover -s tests -v
```

If the fixed CSV must be regenerated, keep the generated provenance JSON with
the dataset:

```bash
python scripts/fix_data.py \
  --input datasets/sdwpf_245days_v1.csv \
  --output datasets/sdwpf_fixed.csv
```

## 1. Validation-only cross-validation

```bash
NEW_MODULE_LEARNING_RATE=0.000005 \
bash scripts/train/SDWPF_paper_cv.sh
```

To confirm a selected setting only on the remaining predetermined seeds:

```bash
FOLDS='0 1 2' SEEDS='2025 2026' \
NEW_MODULE_LEARNING_RATE=0.000005 \
bash scripts/train/SDWPF_paper_cv.sh
```

`rolling_holdout` divides the 70%-80% interval into three disjoint expanding-
origin validation folds. It never evaluates the final 80%-100% holdout.
The three regime pseudo-label thresholds are calibrated from training-history
trend quantiles only, saved in the pretraining checkpoint, and reused unchanged
for validation. Logs report accuracy, macro-F1, per-class recall, support, and
the confusion matrix; accuracy alone must not be cited.

Run validation baselines with the same fold and horizon:

```bash
for fold in 0 1 2; do
  FOLD="$fold" SPLIT=rolling_holdout EVAL_SPLIT=val \
    bash scripts/eval/SDWPF_baselines.sh
done
```

Generate an interim prediction figure from one predeclared CV validation run
without opening the sealed test split:

```bash
FOLD=0 SEED=2024 SPLIT=rolling_holdout PRED_LEN=12 \
FINETUNE_CHECKPOINT='<matched_fold_seed_checkpoint.pth>' \
bash scripts/eval/SDWPF_logged_validation_plot.sh
```

The resulting run manifest records `stage=validation_report` and
`evaluation_split=val`.  Use the figure only as a validation illustration;
do not describe it as final test performance.

For a matched pretraining checkpoint, run the controlled ablations with:

```bash
FOLD=0 SEED=2024 SPLIT=rolling_holdout \
PRETRAIN_RUN_ID='<matched_pretrain_run_id>' \
bash scripts/finetune/SDWPF_ablation_suite.sh
```

## 2. Freeze the configuration and train final seeds

Do not change architecture, loss, feature, optimizer, or cleaning settings
after this point.

```bash
NEW_MODULE_LEARNING_RATE=0.000005 \
bash scripts/train/SDWPF_paper_final.sh
```

This trains three predetermined seeds with `time_ratio` (70% train, 10%
validation, 20% sealed test) and prints the checkpoint-specific final command.

## 3. Evaluate the final test once per predetermined seed

Run each command printed by stage 2. The required shape is:

```bash
CONFIRM_FINAL_EVAL=1 SPLIT=time_ratio N_FOLDS=1 PRED_LEN=12 \
FINETUNE_CHECKPOINT='<checkpoint.pth>' \
bash scripts/eval/SDWPF_logged_final_eval.sh
```

The numbered final-evaluation directory contains:

```text
artifacts/
├── forecast_trace.png
├── forecast_trace.pdf
├── forecast_trace.csv
├── forecast_trace_selection.json
├── predictions.npz
└── metrics_by_horizon.csv
```

The default figure selects the longest continuous test block, with deterministic
tie-breaking. To predeclare a specific example instead, set
`FORECAST_PLOT_TURBINE_ID` and optionally `FORECAST_PLOT_START` before running
the logged final-evaluation command.

Run the final baselines under the same split exactly once:

```bash
CONFIRM_FINAL_EVAL=1 SPLIT=time_ratio N_FOLDS=1 EVAL_SPLIT=test \
bash scripts/eval/SDWPF_baselines.sh
```
