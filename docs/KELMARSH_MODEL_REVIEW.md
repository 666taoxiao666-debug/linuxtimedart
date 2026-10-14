# Kelmarsh model review and forecast-relative Wiki pilot

Review date: 2026-10-14. Scope: hourly v1 data/model/knowledge/decision path and
its completed 3-fold x 3-seed experiment. This is a development review, not a
submission-ready claim or an independent assessment by multiple reviewers.

## 最终评级：C

The examined pipeline has explicit chronological boundaries, prefix-only
scaling, future-power-only target eligibility, raw negative/above-rated power
reporting, real baselines and expert-bound policy identities. No P0 leakage or
label alignment defect was identified in the paths examined. This is not a
repository-wide certification. The claimed accuracy benefit of Wiki is not
supported: mean MAE 141.973420 kW versus its actual base 141.668116 kW, with
only two active runs (both negative gain) and seven exact abstention runs.

## 必须立即修复 Top 5

1. **[P1] Forecast-relative utility is missing at inference.**
   `utils/kelmarsh_hourly_wiki.py -> candidates (lines 123-127), fit_policy
   (lines 159-177), apply (lines 204-214)`: a neighborhood is compressed to a
   median; the candidate is a bounded replacement, selected by an event/bin's
   historical average gain. Individual current forecasts and local outcome
   dispersion do not enter the cell score. Historical cell gain does not imply
   gain for this query. Evidence: f0/s2024 utility accepted five cells yet outer
   gain was -2.398346 kW; f0/s2025 gain -0.349388 kW. This establishes transfer
   failure, not its sole cause. Confidence: high for implementation, medium for
   causal attribution. **Implemented pilot:** retain local outcome distributions
   and make correction intensity a function of the current base forecast.

2. **[P1] Conditional future-level experts are not additive causal effects.**
   `utils/kelmarsh_hourly_wiki.py -> candidates (lines 129-132)`: summing
   `anchor + transition_k - base` across two events subtracts the same base twice.
   The events may also be correlated descriptions of the same condition.
   A cap limits size but does not repair the interpretation. Composition selected
   zero points in all nine runs, so it did NOT explain the measured v1 loss.
   Confidence: high. **Pilot:** disable additive composition and compare with
   the matched old no-composition control (identical to old full results here).
   A future composition experiment needs a joint-event outcome distribution or
   an identifiable non-overlapping effect definition, not arbitrary addition.

3. **[P2] Most advertised event experts never execute.**
   `utils/kelmarsh_hourly_wiki.py -> fit_experts (lines 78-80)`:
   the uniform 128-window minimum closes low-wind idle with 45/17/85 memory
   windows across the three folds. High-wind low-power has 0/5/0 and rated
   saturation 0/0/0. Only hourly wind variability is active. More neural epochs
   cannot create missing event memory. Confidence: high. Do not lower this
   threshold after seeing outer results. A separate study could test hierarchical
   pooling with an explicit rare-event uncertainty penalty or additional years.
   This pilot leaves support thresholds unchanged.

4. **[P2] Cross-turbine counts do not establish temporal independence.**
   `utils/local_transport_event_wiki.py -> _neighbors (lines 41-55)` and
   `utils/kelmarsh_hourly_wiki.py -> fit_policy (lines 163-175)`: nearby windows
   have overlapping 72-hour histories, and same-farm turbines share weather.
   The memory has no case timestamp in its exported expert, so local temporal
   diversity cannot be audited from it. Unit-balanced medians and a 2/3 positive
   turbine fraction are useful descriptive constraints, not a transport theorem
   or confidence bound. Confidence: high. Retain time-cluster bootstrap for
   evaluation; record case timestamps before a separate uncertainty-guarantee
   study. The new empirical rule makes no distribution-shift guarantee.

5. **[P2 / NEEDS-EVIDENCE] Numeric objective and desired joint improvement differ.**
   `scripts/train_kelmarsh_hourly.py -> train_stage (lines 214-220, 261-264)`:
   epoch selection uses MAE, while training blends absolute and squared error.
   Pretraining reconstructs historical inputs rather than directly forecasting
   future power (`models/TimeDART.py -> pretrain`, line 927 onward). f1 selects
   epoch zero for all three seeds; the numeric base improves macro MAE over
   Persistence by only 0.129982 kW. These are observed generalization/selection
   limits, not evidence that optimization is permanently blocked or the data are
   defective. Confidence: high for the code/results; needs training-side evidence
   for root attribution. A separate base study should test forecast-aware
   pretraining or train-only Pareto checkpoint selection, with matched budgets.
   The present pilot freezes the base to avoid conflating base and Wiki gains.

## Implemented mechanism: forecast-relative empirical Pareto readout

### Additional confirmed information omission

`data_provider/kelmarsh_hourly.py -> physical_context` and
`utils/kelmarsh_hourly_wiki.py -> keys` retain 12-hour wind mean/std and a
six-hour-average change, but not last wind, immediate wind ramp, or rotor speed.
For example, `[4]*10+[2,8]` and `[4]*10+[8,2]` have identical old wind coordinates
yet current wind 8 and 2 m/s. A regression test proves the coordinate alias.
This is a **[P2] confirmed representation limitation**, not proof that six extra
features improve a neural model or that it explains every loss. A separate
fixed-alpha Ridge probe fits only donor memory labels and predicts the two later
training blocks using old coordinates versus old plus six current-state signals.
It does not use outer labels or replace a full Wiki experiment. Its script is
`scripts/audit_kelmarsh_phase_coordinates.py`; support for any subsequent new
coordinate protocol must come from these train-side results, not another outer
threshold search.

For a query and lead, neighboring train cases provide normalized future power
transitions `z_i`. Each represented donor has equal total mass. Let `m` be their
weighted median, `mu` their weighted mean, and `b` the actual frozen base's
forecasted transition from the query's last observed power. Define:

```text
d = clip(m - b, -0.2, 0.2)
alpha = clip((mu - b) / d, 0, 1), or 0 when d = 0
candidate_transition = b + alpha * d
```

This is the squared-loss minimizer on the bounded segment from `b` towards the
median. It also lies on a segment along which weighted absolute risk cannot
increase. If mean and median lie on opposite sides of the base, `alpha=0`.
The non-increase statement concerns ONLY the empirical neighbor distribution,
not the unknown query outcome. Require positive empirical MAE and MSE gains and
joint gain in at least 2/3 represented donors; then re-estimate realized gain for
this exact readout on both subsequent train-utility blocks. The finest supported
harmful cell still vetoes optimistic coarse fallback. Selection is scored by the
minimum of two-block cell gain and the query-specific empirical absolute gain.

The stored memory, numeric checkpoint, scaler and old thresholds remain frozen.
New readout implementation/memory/protocol identity and full base contract bind
the new utility evidence; the old policy cannot authorize the new expert.
`MedianLocalWiki` uses full median replacement with the same local dual-loss
gate, isolating the contribution of closed-form intensity from local screening.
No alpha grid, risk-threshold search, external LLM judgment, or causal-effect
claim is introduced.

## Innovation boundary and verified nearby work

- [Nearest Neighbor Multivariate Time Series Forecasting](https://arxiv.org/abs/2505.11625)
  (2025 arXiv preprint; method): cached nearest-neighbor forecasting already
  exists. Retrieval alone is not a new contribution.
- [GNA](https://arxiv.org/abs/2609.36281)
  (2026-09-28 arXiv preprint; method): its inspected abstract explicitly describes
  two retrieval granularities and per-lead trust gating against Persistence and
  the backbone. Therefore granular retrieval plus a gate is not a sufficient
  novelty claim. Only the abstract was inspected; no claim of an exhaustive
  overlap check or reproduced performance is made.
- [Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html)
  (ICLR 2024; theory/method): calibrated expected-risk control is established.
  This pilot is NOT CRC, has no valid conformal exchangeability/shift proof, and
  must not borrow its guarantee terminology.

The proposed differentiating hypothesis is a joint-loss, forecast-relative
readout for physically constrained event transitions, with exact expert-evidence
binding and time/turbine transfer assessment. The mean/median mathematics is
standard. Novelty and actual utility require evidence; renaming it does not make
it original. No score or publication acceptance probability is assigned.

## 需重跑的实验与需补充的单元测试

Do NOT repeat the v1 backbone training or completed nine-run matrix. Refit only
the actual new policy on original donor-only train-utility blocks. Protocol:
`configs/kelmarsh_pareto_protocol_v2.json`, fixed folds 0/1/2, seed 2024.
First run training-only audits; one predeclared outer evaluation can then compare
Trend, old no-composition Wiki, matched-label NumericControl, ParetoWiki and
MedianLocalWiki. All share the exact same checkpoint and outer windows. Outer
validation was previously inspected: this is exploratory, not independent
confirmation. No sealed suffix prediction or future weather is used.

Tests added: closed-form joint-risk reduction including a full-median counterexample;
mean/median conflict abstention; correction bounds; donor duplication invariance;
random empirical-loss non-increase; inference without query truth/turbine labels;
exact no-event fallback; no additive composition; changed base/readout rejection;
chronological overlap and held-out donor-label rejection.

## Operations

```bash
conda activate timedart
python -m unittest discover -s tests -p test_kelmarsh_pareto_wiki.py -v
bash scripts/train/Kelmarsh_launch_pareto_wiki.sh          # start once: train only
bash scripts/train/Kelmarsh_launch_pareto_wiki.sh --status
bash scripts/train/Kelmarsh_launch_pareto_wiki.sh --evaluate # stopped controller only
tail -f "$(cat outputs/logs/Kelmarsh/pareto_wiki_latest.txt)/launch.log"
```

`--resume` reuses completed fold artifacts and never changes the frozen protocol.
Wait for real controller/GPU exit before either continuation. The outer
evaluation marker persists for safe interrupted-report continuation. No new
seed/threshold/candidate search is authorized by these commands.
