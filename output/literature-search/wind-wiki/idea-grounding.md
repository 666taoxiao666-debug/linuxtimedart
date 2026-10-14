# Fast decisions and transferable Wiki benefit: evidence packet

Reviewed 2026-10-14. Exploratory mechanism development, not a novelty verdict.
The numerical path, labels, existing protocols and historical results remain
frozen. No new prediction-quality result is claimed in this packet.

## Optimized task

Connect the current failure mechanisms to verified primary literature. Explain
what Jev publicly does, distinguish event relevance from expected correction
benefit, and propose one minimal falsifiable Wiki-only route without using
outer validation to fit or choose rules.

## Current project observations

The completed three-fold seed2024 quality policy has macro MAE 145.010081324 kW,
RMSE 209.345746876 kW and macro R² .743662445. It is only .140571047 kW better
than untreated WikiSelective and remains worse than OriginalTrend
(142.494880101/205.714966101/.753201239). Raw Wiki candidate MAE did not improve.
These are historically inspected development results, not independent tests.

The new training-only metadata audit reproduced every certificate cell's
window/turbine counts. Fold2's accepted cell contains 30 windows but just 10
distinct issue times and 2 UTC issue-day blocks. Date blocks are not guaranteed
independent weather episodes. See the full experiment report and the local
`output/kelmarsh_hourly/wiki_quality_review/evidence_support_audit.json`.

Current event action IDs all enable the same neural Wiki forecast: event
conditioning is not four independently executing neural experts. Exact
checkpoint/base bindings now exist, but identity alone cannot ensure transfer.

## Verified source cards

### 1. Jev / TypeSafe System One: public product documentation

- Sources: [official introduction](https://typesafe.ai/blog/introducing-system-one-models-and-jev?render=full),
  [System One](https://docs.typesafe.ai/concepts/system-one),
  [confidence](https://docs.typesafe.ai/confidence).
- Read: architecture/sampling comparison, workflow evaluation caveats,
  type-safety nuance, primitives and calibration/confidence definitions.
- Source observation: text/structured state and predefined questions produce
  typed probability outputs. The vendor describes a parallel sampler and
  Reinforcement Learning for Calibrated Decisions (RLCD). Confidence summarizes
  the returned distribution. Group calibration does not ensure each answer is
  correct. Schema conformity is not factual or numerical correctness.
- Unknown: the consulted sources do not specify enough architecture/training
  detail to reproduce the internal model or RLCD algorithm.
- Project inference: borrow finite-action interfaces and separate questions,
  not an API probability as supervision for wind-benefit estimation. Current
  routing is already computationally cheap; evidence quality, not latency,
  is the demonstrated bottleneck. No API or paid integration is needed now.

### 2. General Decision Models: Benchmarking and Insights Beyond Jev

- [Author preprint](https://arxiv.org/abs/2610.03935), October 2026; no accepted
  venue verified. Read abstract and benchmark/uncertainty discussion.
- Source observation: benchmarking reports weaknesses in specialized knowledge
  and uncertainty honesty; local speed does not automatically improve complete
  workflow success.
- Project inference: text-domain decision quality cannot be assumed to transfer
  to calibrated probabilities of numerical forecasting gains.

### 3. Regression with Multi-Expert Deferral — ICML 2024

- [Official proceedings](https://proceedings.mlr.press/v235/mao24d.html),
  [author full text](https://arxiv.org/html/2403.19494).
- Read: introduction, section 4 two-stage scenario and conditional cost setup.
- Source observation: a pre-trained predictor can be combined with a separately
  learned deferral function using expert prediction costs, under stated loss
  and learning assumptions. It also covers multiple experts and single-stage
  training. Its consistency results are not arbitrary time-shift guarantees.
- Project inference: the closest next model is a two-action conditional-cost
  selector over the actual frozen NumericContext and WikiContext, not a new
  event-text classifier. Learning-to-defer itself is prior art, not our novelty.

### 4. Learning-to-Defer in Non-Stationary Time Series via Switching State-Space Models

- [Author preprint, version 4](https://arxiv.org/abs/2601.22538v4), updated
  October 2026; no accepted venue verified. Read problem/abstract and feedback
  setup, not a reproduced implementation.
- Source observation: models time-varying expert performance with latent state
  and routing, distinguishing full historical expert information and partial
  deployment feedback; claims depend on model/identifiability conditions.
- Project inference: nonstationary expert routing is also existing research.
  Our cheap simulated candidates permit full feedback after labels mature;
  bandit exploration and a switching latent model are not automatically needed.

### 5. An Adaptive Approach for Probabilistic Wind Power Forecasting Based on Meta-Learning

- [IEEE TSTE 2024](https://doi.org/10.1109/TSTE.2024.3379835),
  [author version](https://arxiv.org/abs/2308.07980).
- Read cached full text: offline task construction and online loss/update
  equations 17–18, explicitly accounting for label delay and forgetting.
- Source observation: adaptation uses outcomes that have become available;
  meta-training targets adaptation across tasks, not a universal proof that
  event knowledge remains useful.
- Project inference: borrow maturity clocks and prequential evaluation of
  evidence expiry. Updating from outer labels would require a new declared
  online protocol; it is forbidden in the current frozen outer evaluation.

### 6. Self-RAG — ICLR 2024

- [Official paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/25f7be9694d7b32d5cc670927b8091e1-Paper-Conference.pdf),
  [full text](https://arxiv.org/html/2310.11511).
- Read sections 3.1–3.2: retrieval, relevance, support and utility reflection
  labels have distinct roles.
- Source observation: retrieval and judging generated content are decomposed
  rather than treating any retrieved text as helpful.
- Project inference: separate observed operating-state applicability from
  forecast benefit. An LLM's semantic utility judgment cannot replace actual
  paired forecast errors or justify the already unsuccessful binary veto.

### 7. A-Mem — NeurIPS 2025; A-RAG — 2026 preprint

- [A-Mem official proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/19909c36f51abc4856b4560aff3d36d6-Abstract-Conference.html),
  [A-RAG author preprint](https://arxiv.org/abs/2602.03442).
- Read A-Mem memory-note creation/linking/evolution in the cached full text;
  A-RAG hierarchical keyword/semantic/chunk interfaces in full HTML.
- Source observation: mutable memory organization and retrieval granularity
  are useful design axes. These papers study language-agent tasks, not
  numerical wind-error certificates or guaranteed forecast improvement.
- Project inference: keep evidence lineage and retrieve finer context only
  when justified. More fine cells are not automatically better: sparse
  correlated calibration data can make granularity selection less reliable.

### 8. Adaptive Conformal Predictions for Time Series — ICML 2022

- [Official proceedings](https://proceedings.mlr.press/v162/zaffran22a.html).
- Read abstract and method summary; not a full replication or selected model.
- Source observation: online aggregation of adaptive conformal methods targets
  prediction-interval calibration in dependent/changing time series.
- Project inference: interval coverage and selective correction benefit are
  different estimands. Conformal methods cannot manufacture independent
  calibration observations or automatically improve MAE/R².

## One prioritized route, not a module stack

First finish a training-only frozen-checkpoint replay. Preserve exact expert,
base, scaler, cohort and target clocks. Save each mature outcome's paired
action losses and event/context support; count issue-time and calendar blocks
instead of treating simultaneous turbines or adjacent windows as independent.
Estimate blockwise variability and report sparse cells, without changing any
current policy. An oracle comparison is retrospective diagnostic headroom,
never a deployable forecast or a way to select on outer labels.

If there is repeatable training-only headroom, declare a new two-action
experiment: action 0 leaves the actual frozen numeric output unchanged; action
1 executes the actual frozen bounded Wiki correction. Learn conditional
expected error difference from chronological cross-fitted training outcomes,
separately from the physical/context support mask. Compare one lightweight
cost selector to the current frozen rule using the same candidates and label
budget. Do not add new experts, seeds or validation threshold searches.

The target is signed error reduction, not only P(win). For illustration,
90% chances of saving 1 kW and 10% chances of losing 20 kW yield expected gain
-1.1 kW. This is a toy calculation, not an experimental result. A confident
event label or binary win prediction ignores this asymmetry.

Stop if the supported candidates have little oracle headroom or the training
blocks cannot support the conditional decision. Better retrieval/routing cannot
create predictive information absent from both candidates. With only two
date blocks in an accepted cell, a fine neural gate is not justified merely
because a GPU is available. Hierarchical shrinkage/partial pooling could be a
later competing hypothesis, not a simultaneously added confound.

## Novelty boundary and evidence needed

"Wiki + fast decision + deferral" is not enough for a new-method claim.
The research opportunity is narrower: define when benefit evidence transfers
across time/turbines while remaining tied to the actual executing correction,
including evidence maturity, compatibility and expiry. Hash checks alone are
reproducibility engineering; a scientific claim needs a precise transfer
assumption, failure cases and held-out evidence that tests it.

Minimum evidence: fixed-candidate oracle/realized benefit, same-label-budget
selector comparison, time-block and donor-turbine holdouts, expert-version
swap invalidation, expiry/forgetting ablation, paired MAE/RMSE/R² plus coverage,
selected gain, harm, abstention and block uncertainty. Do not describe the
existing development outer period as independent confirmation.
