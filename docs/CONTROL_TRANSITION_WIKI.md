# Control-conditioned physical transition knowledge (v15)

## Research question and code change

Does event knowledge transfer more usefully when it stores an observed
physical power transition, rather than the error of an earlier forecaster?
Can a case memory remain reusable while its base-specific utility evidence
is deliberately invalidated by a base replacement?

`utils/control_transition_wiki.py` implements an opt-in prediction plug-in.
It does not change TimeDART defaults, frozen v1/v12/v13 artifacts, labels,
neural normalization, or the completed v14 angle experiment.

For a source case i, store `d_i,h=(Y_i,h-P_i,last)/capacity`. History keys
contain power/wind working points and, in the declared primary arm, observed
mean/last/change of mean blade pitch and mean/last relative-yaw cosine.
Keys do not contain a hidden backbone state or its prediction. The source
scaler is inverted before constructing these physical coordinates. Wdir is
already relative to the nacelle; the legacy `Wdir-Ndir` yaw is not used.

For a query q, a fixed 32-neighbor, source-turbine-balanced median gives
`M_q,h=P_q,last+capacity*median(d_i,h)`. The actual event candidate is
`clip(B_q,h+clip(M_q,h-B_q,h, +/-20% capacity), 0, capacity)`.
Composition retains the earlier 5% capacity bound. This is observational
conditional forecasting, not identification of a causal event effect.

There are two distinct identities:

- **Memory identity**: exact cases, physical targets, scaling, event rules,
  capacity, horizon, implementation and neighbor/support definitions.
  Transition keys/targets do not depend on a base checkpoint. The residual
  ablation does depend on it and cannot reuse a changed base.
- **Execution evidence identity**: exact memory plus the current base
  checkpoint/manifest, actual bounded/clipped candidate outputs and utility
  rule. Changing the base invalidates this evidence even if the physical
  memory remains valid. Calibration is mandatory again; no old expert's
  observed gain is assigned to a newly initialized neural adapter.

The hypothesis is conditional stability of `Y_future-P_last` given observed
working point/control history. Input overlap alone does not establish that
stability; unobserved commands, faults, future wind and terrain can break it.

## Training-side motivation, inspected 2026-10-13

Read-only inspection used the existing fold1/seed2024 original-train OOF
source. No final check or outer validation labels were aggregated in this
inspection. In early calibration, the gust family contained 102 windows with
mean pitch <=5 degrees and 44 with mean pitch >5 degrees. Their average
future power changes were -46.0308 and +100.8156 kW respectively. In late
calibration these groups had 80/24 windows and +10.6535/+54.3055 kW.
Thus the event label mixes working/control regimes, and the temporal
change also warns against claiming conditional invariance already holds.
The five-degree split is a diagnostic only; the new model uses continuous
control coordinates, not a label-tuned threshold.

The previous v13 source had zero accepted policy cells and 100% abstention.
Its later calibration contained no low-wind-idle windows. This scarcity is
not fixed by adding features. The fixed utility requirements are retained;
zero coverage in v15 will be reported as zero useful transport, not success.

## Declared discriminating experiment

`configs/sdwpf_control_transition_protocol_v15.json` freezes a 2x2 design:
residual versus physical-transition values, with versus without control
coordinates. All four arms share history-key conventions, 32 neighbors,
support/bounds, time/turbine splits and utility rules. Excluding the
base-change key in all arms isolates the readout/control comparison; the
historical v13 output is not presented as an otherwise identical ablation.

First50% of the original-train OOF calendar fits memory; next25% supplies
two disjoint utility blocks; last25% is a development check. Turbine IDs
divisible by5 enter neither memory fitting nor utility calibration. The
frozen backbone may have seen these IDs, so this is **knowledge-only**
held-out transfer, not cold-start forecasting.

Utility keeps v13's existing 64 windows/8 turbines in each block, at least
2/3 positive turbines, and positive turbine-balanced MAE and MSE gains.
Routing uses the finest supported event/power/wind cell at each horizon;
a supported harmful fine cell cannot escape into a positive pooled average.
The worst temporal mean is descriptive, not a statistical risk guarantee.

Report all arms, without selecting a new winner, threshold, epoch or seed
from check results. Primary arm is `transition_control`. Report original
MAE/MSE/RMSE/R2, both Persistence Skills, capacity tolerance hits, candidate
support, intervention coverage, selected gain, harm and abstention. A
joint accuracy gain with nonzero coverage is required before discussing
external confirmation. Even a positive result here is development evidence:
the original-train period has been inspected in earlier experiments.

The primary paired-statistics estimand is now explicitly window-uniform.
Issue-time clusters are resampled with their gain sums AND window counts;
the CI concerns the same ratio as the main MAE gain. A Newey-West
ratio-mean influence test handles serial dependence approximately. Legacy
time-uniform output remains available and explicitly labeled; previously
saved v1 significance artifacts are not overwritten or silently reinterpreted.

## Closest work and honest novelty boundary

- [FreshCast, arXiv 2026 preprint](https://arxiv.org/abs/2610.07834): frozen
  forecaster, refreshed nonparametric memory and calibrated correction.
  Those ingredients alone are not our novelty. Re-anchoring a retrieved
  trajectory is also not claimed to be newly invented here.
- [RATL, arXiv 2026 preprint](https://arxiv.org/abs/2609.03937): explicitly
  retrieves base-specific residuals and routes their corrections. Our
  hypothesis tests a different stored quantity and separates memory from
  executor-specific gain evidence.
- [RACE, arXiv 2026 preprint](https://arxiv.org/abs/2610.00405): coherent
  neighbor residuals and a correction Gate, including cross-backbone
  transfer. A claim that transferable gates or residual correction are
  unexplored would be false.
- [Analog ensembles, UCAR](https://ral.ucar.edu/products/analog-ensemble-anen):
  physical analog forecasting is established. Adding pitch/yaw alone is
  an incremental domain adaptation, not a new forecasting paradigm.
- [Covariate shift, JMLR 2007](https://www.jmlr.org/papers/v8/sugiyama07a.html):
  distinguish changed input distributions from stable conditional outcomes.
  Our overlap and temporal checks are empirical evidence, not proof of
  unchanged conditional outcomes.

The defensible intended contribution is an **expert/base-specific
knowledge-transport problem and mechanism**: separate the physical response
being reused from evidence about its benefit for a particular executor;
test conditional transport across time and turbines. Hashes are enforcement
and reproducibility infrastructure, not the scientific insight themselves.
No new result, significance, innovation score or high-tier acceptance is
claimed by implementing this prototype. Independent-period/multi-farm
evidence and a closest-mechanism comparator remain necessary.

## Operation

In the server repository after activating the existing environment:

```bash
git pull --ff-only origin codex/self-evolving-wiki
conda activate timedart
python -m unittest discover -s tests -p 'test_control_transition*.py' -v
python -m unittest discover -s tests -p 'test_paired_gain_estimand.py' -v
python scripts/train_sdwpf_control_transition.py --preflight \
  --drift-dir "$(cat outputs/logs/SDWPF/residual_drift_audit_latest.txt)"
bash scripts/train/SDWPF_launch_control_transition.sh --start
bash scripts/train/SDWPF_launch_control_transition.sh --status
tail -f "$(cat outputs/logs/SDWPF/control_transition_latest.txt)/launch.log"
```

Use `--start` once. The launcher is CPU-only and defers while the GPU is busy.
`--resume` is for an exited real failure after repair, not a new trial; frozen
memory/policy and completed arms are verified and reused. Plot failures do
not cause retraining. No automatic outer refit, sealed test, additional
seed or periodic threshold search is part of this pilot. Primary plots are
under `transition_control/*_artifacts/forecast_accuracy_overview.png`.

## Targeted code-review result

Rating **C for scientific claim readiness**, not a certification of the whole
repository. No new P0 found in this scoped data flow. Top remaining checks:

1. [P1 addressed by contract/tests] Memory/base mismatch: utility cannot be
   reused after a base update; transition memory alone may be reused.
2. [P2 hypothesis, not solved by definition] Temporal/control confounding:
   continuous pitch/yaw keys cannot observe future commands/weather.
3. [P1 addressed for new statistics] Window/time weighting mismatch:
   explicit ratio bootstrap and primary-estimand labeling added; archived
   historical inference is preserved, not retroactively "fixed".
4. [P2 unresolved] Sparse event calibration: unchanged minimum support
   can still force zero intervention. More suitable independent evidence,
   not a threshold reduction, is needed if this happens.
5. [NEEDS-EVIDENCE] Text semantics and self-evolution: this module is a
   physical case-memory expert, not proof of an LLM semantic advantage or
   autonomous online knowledge evolution. Those claims need distinct tests.
