# Kelmarsh hourly systematic experiment

The active experiment now uses a dedicated Kelmarsh entry point. Old SDWPF
entry points/data/checkpoints remain available for reproduction, not silently
rewritten into a different dataset. Protocol: `configs/kelmarsh_hourly_protocol_v1.json`.

## Fixed task and data path

- Six real 2050-kW turbines; provider 2016 SCADA explicitly aggregated to hourly
  records, with six valid observations per primary hourly channel.
- Use 2016-05-01 through 2017-01-01 exclusive: the first full calendar month after
  official commercial operations date 2016-04-15, not a label-selected easy period.
- Past 72 hours -> next **2 hours**, matching the physical forecast duration of
  the old 10-minute h12 task. Hour averaging, new history length and populations
  still differ, so old/new absolute metrics are not a controlled dataset-only comparison.
- Keep original negative/above-rated targets, no imputation, no assumed yaw error,
  no missing setpoint feature. Inputs are wind speed, temperature, geographical
  nacelle heading sin/cos, mean of three complete blade-pitch channels, rotor speed
  and observed power. Wind-direction/yaw-misalignment semantics are not inferred.
- Each input window must be complete. Only power is required in the future target
  window: rejecting windows because future weather is missing would be a future
  covariate selection rule and is deliberately avoided.

## Chronological protocol

Three outer folds have train endpoints at 60/70/80% of the fixed timeline and
validation windows of the next 10%. The last 10% is reserved and never forecasted
in this run. All neural models use the same eligible windows and labels.

Each outer training period is split into disjoint target windows:

1. First 60%: fit numerical model, pretraining and scaler.
2. Next 10%: select pretraining/forecast epoch, never choose on outer validation.
3. Next 10%: fit physical event transition memories.
4. Next 10%: first utility evidence block.
5. Last 10%: second utility evidence block.

Forecast targets crossing a boundary are excluded, while history before the
boundary is allowed. Scaler statistics remain frozen from the first fit block.
The selected numerical checkpoint is frozen before all knowledge fitting and
evidence collection. There is **no outer refit** that changes its identity.

## Full hourly adaptation of the existing innovation

The numerical base is the existing PromptTimeDART implementation, with actual
diffusion pretraining, learned stable/up/down prompts and channel-mixed forecasting.
The Wiki is the project's latest physical control-transition memory mechanism,
adapted to hourly data rather than migrating a minute-scale neural adapter.

- Historical wind/power evidence enables four event families; hourly wind
  variability is not called measured turbulence.
- Stored outcomes are capacity-normalized future power transitions from the
  last observed power. Retrieval uses physical history, measured pitch and actual
  nacelle heading, not an inner model's hidden representation.
- Nearby cases are combined using the existing turbine-balanced median; event
  corrections and additive multi-event composition are bounded separately.
- Utility is assessed for the **actual frozen candidate outputs** at each lead
  on both subsequent chronological training blocks. Evidence belongs to the
  exact checkpoint + scaler + rules + memory + readout + horizon identity.
- Retrieval granularity is event, event-power-bin or joint wind/power-bin. Use
  the finest supported cell. A supported harmful fine cell blocks an optimistic
  coarse fallback; missing/negative evidence means exact no intervention.
- Cells failing time/turbine benefit requirements are retired from execution.
  This is conditional retention/forgetting, not a claim of online autonomous LLM
  knowledge acquisition or an unseen theoretical transport guarantee.

Knowledge donors are turbines 1–5; turbine 6 supplies **no memory or utility-fit
labels**. Report its later prediction separately. The numerical backbone has
trained on all six: this is held-out **knowledge transfer**, not wholly unseen
neural-model domain generalization. The new six-turbine protocol explicitly uses
three supporting donor turbines, rather than an impossible old minimum of eight.

## Systematic comparisons and metrics

Three seeds (2024/2025/2026) for each of three folds, sequential execution:

- Persistence, previous-day persistence and fixed-alpha Ridge.
- The repository's DLinear and PatchTST implementations.
- PromptTimeDART trend base and full Wiki.
- Wiki event-only granularity, no-control memory and no-composition ablations.
- For each numerical model, a global Ridge residual control consumes the same
  donor memory/utility labels, measured physical coordinates and bound as Wiki,
  but has no event knowledge, case retrieval, granularity selection or composition.
  This controls for additional supervised labels and ordinary error calibration;
  comparing Wiki only with its earlier-fitted base would confound those benefits.

Neural hyperparameters, losses, budgets and learning rates are frozen before
training. Strong baselines are actual model implementations, not proxies; there
is no claim these fixed budgets exhaust their best possible configurations.

Report original-scale MAE/MSE/RMSE/R², MAE/RMSE Skill versus Persistence,
capacity-normalized errors and ±5/10% capacity tolerance hits, both lead times,
per-turbine errors and plots. Tolerance hits are not classification accuracy.

Report Wiki physical support, intervention window/point coverage, abstention,
selected point gain, point/window harm rates, events, granularity and turbine-6
transfer. Undefined harm/gain when nothing is selected is null, not a fabricated zero.
Compute issue-time-cluster block bootstrap and Newey-West error-difference tests
against the same numerical checkpoint. Seeds repeat the same validation timeline;
they are not nine independent datasets. Macro summaries are run means, not pooled R².

## Commands and artifacts

Activate the existing Python environment containing PyTorch, sklearn and pandas.

```bash
python -m unittest discover -s tests -p test_kelmarsh_hourly.py -v
python scripts/train_kelmarsh_hourly.py --audit-only
bash scripts/train/Kelmarsh_launch_hourly_system.sh
bash scripts/train/Kelmarsh_launch_hourly_system.sh --status
tail -f "$(cat outputs/logs/Kelmarsh/hourly_system_latest.txt)/launch.log"
```

Start once. `--resume` is only for a confirmed stopped/failed controller, not a
timed restart. Completed checkpoints are hash-validated and reused. Epoch-boundary
resume restores optimizer, schedule and RNG state; a plot/report failure does
not justify repeating training. No parallel GPU matrix is launched.

Results reside in `outputs/logs/Kelmarsh/DATE/NUMBER_hourly_system_h2_.../`:
`status.env`, `progress.json`, `protocol.json`, `data_audit_f*.json`,
`metrics_all_runs.csv`, `metrics_macro.csv` and `runs/fF_sSEED/` containing
checkpoints, histories, exact raw predictions, expert/policy identities, metrics,
interventions and prediction/accuracy figures. The live result pointer is
`outputs/logs/Kelmarsh/hourly_system_latest.txt`.
