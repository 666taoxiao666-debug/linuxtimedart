# Self-Evolving Event Wiki

This implementation adds an offline knowledge lifecycle to the compositional
wind-event Wiki. It is designed to keep model selection and final testing
leakage-safe.

## What is implemented

- **Knowledge merge:** candidates with the exact same executable physical rule
  are merged. Different physical rules are never merged merely because their
  text sounds similar.
- **Forgetting:** candidate reliability decays exponentially with the number of
  training steps since its last OOF evidence. Knowledge below the retirement
  threshold receives deployment weight `0` and cannot affect the forecast.
  An original seed factor is retained until it has actually been assessed, so
  a missing candidate record cannot silently erase the base Wiki.
- **Cross-turbine transfer:** every turbine is treated as one transfer unit.
  Reliability uses a lower confidence bound across turbine-level mean utility,
  not a pooled window average that lets one large turbine dominate. Overlapping
  forecast windows are not counted as independent confidence-bound samples.
- **Horizon-aware survival:** with `pred_len` evidence, merge, decay, transfer
  screening, and retirement operate on each forecast step. A short-lived ramp
  may survive on the steps where it helps even if its whole-horizon mean is
  negative; an unsupported step is assigned exactly zero reliability.
- **Selective residual:** the factorized event experts receive observable
  SCADA-rule support and a frozen horizon reliability mask. Their bounded
  residuals are routed by predicted step-wise utility; the abstention action
  exactly returns the original trend prediction.
- **Safe growth:** a new Wiki factor must provide a rule in a constrained JSON
  DSL. The DSL supports observable historical statistics only; it cannot run
  arbitrary LLM-generated code.
- **Frozen deployment:** the evolved JSON and embedding bundle are versioned and
  frozen before validation/test. Their hashes are checkpoint contracts.

## Evidence contract

The input JSON must set `source_split` to `train_oof`. Each candidate contains:

- `id`, `factor_id`, `prompt`;
- `created_step` and `last_evidence_step`;
- optional `rule` for a new factor (an existing factor inherits its rule);
- one `turbine_evidence` row per source turbine with `n_windows`,
  `mean_utility`, and `std_utility`;
- every evidence row must also be `source_split=train_oof`.

When launching a fold run, set top-level `sdwpf_fold` to that fold's integer
index; the launcher checks it against `FOLDS`. This declaration and the
`train_oof` flag are audit guards, not a substitute for preserving the actual
training cutoff and OOF prediction provenance.

For horizon-aware deployment also set top-level `pred_len` equal to the model
forecast horizon, and give every turbine row `horizon_mean_utility` and
`horizon_std_utility`, each a list of exactly `pred_len` finite values. These
are the per-step utility mean and uncertainty from the same train-only OOF
predictions. The evolved JSON and embedding bundle then store one reliability
weight per `(event, forecast step)`; a mismatched bundle or model `pred_len`
is rejected before training. Repeated evolution must keep the same horizon.
The self-supervised pretraining horizon may differ; the check applies when
factorized supervised forecasting actually consumes the per-step weights.

Utility is dimensionless and positive when the candidate helps, for example:

`mean_utility = mean(1 - candidate_window_MAE / reference_window_MAE)`

The candidate and reference predictions must be produced out of fold inside the
training interval. Do not calculate this value on the experiment validation
fold or the final test interval.

The lifecycle checks the declared split, turbine IDs, shapes, and statistics,
but cannot independently prove that an externally supplied evidence JSON was
actually generated out of fold. Preserve the OOF prediction manifest, source
checkpoint hashes, turbine split, and train-only timestamps with the evidence.

Exploratory event-level safety ablation: set `WIKI_MACRO_TRANSFER_GUARD=1`
when launching `SDWPF_evolved_wiki_cv.sh`. A rule is then withheld even if a
few horizons appear profitable unless its complete-forecast train-OOF gain has
both a positive cross-turbine lower bound and the configured positive-turbine
fraction. This is a training-evidence gate, not a validation-tuned threshold;
the default remains off to preserve horizon-only events. Report both variants
and the changed active-factor lists before interpreting validation MAE.

Exploratory temporal transfer guard: set `WIKI_TEMPORAL_STABILITY_GUARD=1`
and `TEMPORAL_EVIDENCE` to the matching fold's `train_oof_temporal_diagnostic_*.json`.
The guard requires each event to meet the existing source-turbine count, window
support, positive-turbine fraction, and cross-turbine lower-bound criteria in
*both* non-overlapping chronological OOF blocks. A failed block retires the
event; neither validation nor test labels can enter this decision. The report
must match the original OOF evidence's fold, horizon, checkpoint hash, data
hash, outer train cutoff, and window count. This one-cutpoint check is a
training-only stability filter, not proof of future transportability; evaluate
the frozen Wiki against the matching pure-trend model on validation afterward.

`configs/wind_event_wiki_evidence.example.json` is a schema starter only and is
marked `example_only`; the lifecycle command intentionally refuses to train
from it.

## Executable rule DSL

Each rule has `combine: all|any` and a non-empty list of conditions. A condition
uses a historical feature (or `power_ratio`), one statistic (`mean`, `std`,
`max_abs_step`, `last`, `trend_delta`), one direction (`above` or `below`), and
either a numeric `threshold` or a `threshold_key` from `rule_defaults`.

## Run

When `EVIDENCE` is omitted, the launcher now generates a real forward-OOF
candidate file. For each outer fold it trains a separate inner model on the
first 80% of that fold's original training time span, selects checkpoints on
the next 10%, and scores event candidates only on the remaining 10%. The
evidence targets end strictly before outer validation. The inner source fits
its scaler on inner training data only. Use a completed, matching trend CV:

```bash
conda activate timedart
TREND_CV_DIR=/path/to/matched_trend_cv \
FOLDS=0 SEEDS=2024 \
PRED_LEN=12 \
WIKI_LLM_PATH=outputs/model_cache/Qwen2.5-0.5B \
bash scripts/train/SDWPF_evolved_wiki_cv.sh
```

The producer can also run separately with `FOLD=0 SEED=2024
TREND_CV_DIR=... bash scripts/train/SDWPF_build_wiki_oof_evidence.sh`.
Its dated log directory contains `oof_plan.json`,
`train_oof_wiki_candidates.json`, `oof.env`, and the inner training logs.
Inspect it with `bash scripts/train/SDWPF_build_wiki_oof_evidence.sh --status`.
Pass a verified existing file as `EVIDENCE` to skip repeated inner training;
the fold and horizon are checked before use.

To audit time drift without fitting another model, rerun only the `collect`
subcommand with the same `oof_plan.json`, inner `OOF_CHECKPOINT`, and base Wiki
config, adding `--temporal-report /new/path/train_oof_temporal.json` and a
separate, new `--output` path. It partitions unseen training OOF targets at
the median target-start timestamp, drops forecasts crossing that boundary,
and reports paired event utility separately in the earlier and later blocks.
The report is diagnostic-only: it is not automatically consumed by the
lifecycle or fitted using outer validation/test labels.

The script creates a versioned Wiki directory and `lifecycle_audit.json`,
builds frozen LLM embeddings, then runs matched Wiki pretraining/static-Wiki
reference followed by factorized selective-residual fine-tuning. Both stages
use the same fold/seed grid; the safe default is one fold and one seed.
Run the command separately for folds 0, 1, and 2 with each fold's own
train-OOF evidence before assembling a 3-fold result. The script refuses a
single evidence JSON reused across multiple fold cutoffs.
`TREND_CV_DIR` must contain the paired trend checkpoint and metrics
for every requested fold/seed; the script refuses to guess this directory.
`source_cv_dir.txt` and `factorized_cv_dir.txt` in the versioned Wiki directory
point to the dated log directories. The factorized directory contains
`wiki_vs_trend.txt` and `wiki_vs_trend.csv`. Final test evaluation remains a
separate one-time command.

To inspect the current run without finding the directory manually:

```bash
bash scripts/train/SDWPF_evolved_wiki_cv.sh --status
```

## Train-only harm-risk veto (protocol v2)

The frozen v1 benchmark showed that a positive mean Wiki gain can coexist with
many harmful intervention windows.  Protocol
`configs/sdwpf_wiki_risk_veto_protocol_v2.json` therefore adds one independent
decision signal without changing the physical rules, residual experts, trend
checkpoint, expected-gain threshold, data split, or seed.  For every physically
available event and forecast step, a second head learns
`P(candidate absolute error > trend absolute error)` from the chronological
calibration-train partition.  At validation inference a candidate remains
eligible only when its physical evidence is present, expected gain passes the
existing threshold, and predicted harm probability is below the fixed 0.5
decision boundary.  No validation or test label enters this risk target.

Run the predeclared exploratory pilot with:

```bash
bash scripts/train/SDWPF_launch_risk_veto_wiki.sh
bash scripts/train/SDWPF_launch_risk_veto_wiki.sh --status
```

The factorized diagnostic reports physical candidate coverage, post-veto
coverage, veto rate, intervention rate, selected gain, harmful-intervention
rate, and exact abstention.  The pilot is a model-development result, not final
evidence; if it succeeds, freeze the configuration before expanding the
fold/seed grid.

## Claim boundary

The code implements the mechanism and its evidence guards. It does not itself
prove accuracy or novelty. A paper claim still requires an ablation against the
static compositional Wiki, merge-only, merge+forgetting, and full
merge+forgetting+cross-turbine transfer, plus genuinely held-out turbine tests.
Add a `horizon-aware lifecycle off/on` ablation with identical data, checkpoints,
and seeds, and report per-step MAE, event intervention rate, harmful-intervention
rate, and aggregate MAE. The current transfer rule tests stability across
source turbines; it is not a target-turbine-specific adaptation model. The
forward-OOF producer scores existing physical factors but does not invent new
LLM cards or rules. A single chronological evidence interval per outer fold
is not repeated inner cross-fitting; report it as forward OOF, not multi-fold
cross-fitting.
