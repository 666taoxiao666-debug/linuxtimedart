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
  not a pooled window average that lets one large turbine dominate.
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

Utility is dimensionless and positive when the candidate helps, for example:

`mean_utility = mean(1 - candidate_window_MAE / reference_window_MAE)`

The candidate and reference predictions must be produced out of fold inside the
training interval. Do not calculate this value on the experiment validation
fold or the final test interval.

`configs/wind_event_wiki_evidence.example.json` is a schema starter only and is
marked `example_only`; the lifecycle command intentionally refuses to train
from it.

## Executable rule DSL

Each rule has `combine: all|any` and a non-empty list of conditions. A condition
uses a historical feature (or `power_ratio`), one statistic (`mean`, `std`,
`max_abs_step`, `last`, `trend_delta`), one direction (`above` or `below`), and
either a numeric `threshold` or a `threshold_key` from `rule_defaults`.

## Run

First generate real train-OOF candidate evidence. Then run:

```bash
conda activate timedart
EVIDENCE=/path/to/train_oof_wiki_candidates.json \
bash scripts/train/SDWPF_evolved_wiki_cv.sh
```

The script creates a dated/versioned Wiki directory, writes
`lifecycle_audit.json`, builds frozen LLM embeddings, and launches the existing
3-fold x 3-seed validation protocol. Final test evaluation remains a separate
one-time command.

## Claim boundary

The code implements the mechanism and its evidence guards. It does not itself
prove accuracy or novelty. A paper claim still requires an ablation against the
static compositional Wiki, merge-only, merge+forgetting, and full
merge+forgetting+cross-turbine transfer, plus genuinely held-out turbine tests.
