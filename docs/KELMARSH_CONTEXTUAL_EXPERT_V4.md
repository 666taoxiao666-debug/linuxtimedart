# History-conditioned executable Wiki expert: v4

This is new CUDA gradient training, not another readout-only nearest-neighbor
probe. Three fold-matched frozen PromptTimeDART bases and their case banks are
reused. Two identically sized causal TCN residual adapters are trained per fold:
NumericContext receives numerical history; WikiContext additionally receives
20 typed case-summary features. These are physical case summaries, not LLM text.

## Frozen chronology and controls

Both arms see the same donor-turbine (1–5) labels, initial seed, parameter count,
optimizer and 20-epoch budget. Their fit set is the memory period plus the early
calibration period. Memory-period queries have **zero retrieval features**: the
complete immutable case bank was not available then. Early calibration queries
can retrieve only the earlier fixed memory. This makes additional numerical
label budget an explicit control rather than attributing it to Wiki knowledge.

The first half of the original late calibration period selects epochs using
MAE and RMSE jointly; the second half calibrates benefits after both checkpoints
are frozen. Windows whose targets cross the split are excluded. Neither part is
outer validation. Turbine 6 is excluded from adapter fitting, selection and
benefit calibration, but its numerical base previously saw it: this is knowledge
transfer, not an entirely unseen-turbine forecasting claim.

The certificate measures the **actual** bounded Wiki-minus-numerical correction,
binds both executing checkpoint hashes, and records label availability time.
Finest supported event/state/lead cells veto coarser fallback. Unsupported or
nonprofitable queries return the numerical control exactly. This certificate is
evidence from one later training period, not a guarantee of temporal transfer.
NumericCalibrated is a further non-Wiki control using the same final calibration
labels for a global per-lead keep/revert decision.

## Outputs and interpretation

Each fold reports Trend, NumericContext, NumericCalibrated, WikiRaw, WikiPhysical
and WikiSelective. WikiRaw/Physical expose candidate and gate failures; only
reporting the abstaining final policy could conceal an ineffective expert.
Reports include raw-scale MAE/MSE/RMSE/R², persistence skill, capacity-tolerance
hit rates and prediction plots. WikiSelective adds supported coverage, intervention,
harm and abstention. Time-clustered block bootstrap/Newey–West comparisons use
NumericContext as their reference. Compare against NumericCalibrated as well;
beating the old numerical base alone does not establish Wiki contribution.

The outer validation period was already inspected historically. Its single
predeclared evaluation per fold is development evidence, not independent
confirmation. No sealed test, power-label edits, outer epoch selection or
threshold search is allowed. Preserve unsuccessful outcomes.

## Server operation

Activate `timedart`, ensure no other GPU experiment is running, then:

```bash
bash scripts/train/Kelmarsh_launch_contextual_expert.sh
bash scripts/train/Kelmarsh_launch_contextual_expert.sh --status
```

Only after a genuine failed process has exited and the GPU is free:

```bash
bash scripts/train/Kelmarsh_launch_contextual_expert.sh --resume
```

Completed adapters are reused by immutable identity/hash; interrupted adapters
resume their optimizer, scheduler, epoch and Torch RNG states. Report-generation
failure must not retrain a completed adapter. The three folds run sequentially.
