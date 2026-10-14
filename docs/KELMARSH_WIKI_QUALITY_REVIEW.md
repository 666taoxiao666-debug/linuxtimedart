# Wiki-only quality handling and v4 regression audit

## 【最终评级】：C

The current evidence does not support a precision-improvement claim. No new P0
leakage or time-alignment defect was identified in the paths inspected here.
This is a targeted audit, not certification of every historical experiment.

## 【必须立即修复 Top 5】

There are three supported concerns; do not manufacture two more findings.

1. Separate regression of the numerical adapter from incremental Wiki effects.
2. Restrict knowledge transfer using observed operating-state compatibility.
3. Do not call low/negative power, stopped rotation or unavailable capacity bad
   labels without independent measurement evidence. Keep the evaluation cohort.

## 【需重跑的实验与需补充的单元测试】

Do not rerun the frozen numerical arm. Run the declared Wiki-only quality arm
with the same initial seed, 20 epochs, labels, selection/calibration partitions
and neural architecture as v4. Reuse the untreated Wiki controls. Also replay
the old Wiki checkpoint with the new summaries: this isolates input handling
from retraining. Reassess benefit evidence for the new executing expert.

Tests cover sensor consensus, ambiguous-context abstention, chronological
metadata availability, matching states, expert identity, source-turbine exclusion
and byte-identical numerical inputs/labels/cohort after Wiki packing.

## === 详细问题清单 ===

### Numerical regression dominates the combined result

- 严重级别：P1 for an improvement claim, not a metric-calculation bug.
- 位置：`scripts/train_kelmarsh_contextual_expert.py` -> `run` -> numerical
  adapter inference and `route` fallback; `route` returns NumericContext when
  Wiki abstains, not the original PromptTimeDART forecast.
- 触发条件及证据：three-fold seed2024 mean MAE is 142.494880101 kW for the
  original Trend, 145.127339363 for NumericContext, and 145.150652372 for
  WikiSelective. Of the 2.655772271 kW deterioration, 2.632459262 kW already
  exists in the numerical adapter; the incremental Wiki difference is
  +0.023313009 kW. These are same-cohort descriptive differences, not a causal
  decomposition across independently randomized models.
- 对论文结论的影响：Wiki abstention is not protection against regression of
  the separately retrained numerical expert. Wiki-only treatment cannot in
  general recover errors on untouched numerical-path windows.
- 修复方案：keep both original Trend and the actual frozen numerical fallback
  visible; assess Wiki against its actual execution reference. Do not silently
  switch fallbacks to manufacture a quality-handling improvement.
- 验证修复的测试：freeze numeric checkpoint and prediction arrays; assert no
  numerical input, target, cohort or raw prediction changes in the new run.
- 置信度：高。

### Operationally different cases share the same retrieval bank

- 严重级别：P2, a plausible transfer limitation, not yet a proven root cause.
- 位置：`utils/kelmarsh_hourly_wiki.py` -> `keys` and `candidates`; legacy keys
  omit rotor-speed state and available capacity. `kelmarsh_episode_wiki.py`
  changes case weighting but does not enforce these operating-state matches.
- 触发条件及证据：donor-only memory has 20/18/43 observed low-rotation windows
  (RPM < 1) in folds 0/1/2. Fold1 memory includes seven zero-available-capacity
  windows. Neither condition means the power label is erroneous.
- 对论文结论的影响：similar wind/power averages can refer to different
  operating states, weakening transfer. Whether matching improves accuracy
  must be measured, not assumed.
- 修复方案：v5 changes only Wiki context/retrieval: same observed capacity
  category and low/normal rotation, plus contemporaneous blade consensus.
  Unknown/unsupported cases abstain without deleting evaluation rows. Original
  neighborhood size, minimum support, epochs and loss stay fixed.
- 验证修复的测试：reject cross-state retrieval, reject future metadata, preserve
  input/label arrays and reject stale expert identity.
- 置信度：高 for missing conditioning; 中 for its accuracy consequence.

### Rare-event and sensor evidence is not a blanket cleaning justification

- 严重级别：NEEDS-EVIDENCE for a claim that dirty labels caused the failure.
- 位置：`data_provider/kelmarsh_hourly.py` -> `KelmarshSeries` pitch mean;
  `utils/kelmarsh_hourly_wiki.py` -> `fit_experts` minimum support.
- 触发条件及证据：training donor data has 30 hourly blade-disagreement rows
  above 5 degrees, affecting 31 eligible history windows in the early base-fit
  segment. No such windows occur in the three Wiki memory, adapter-fit or
  calibration partitions. Negative training power is as low as -10.5117 kW;
  there is no basis here to replace these targets with zero. All three frozen
  memory banks activate only hourly_wind_variability: other event families lack
  the fixed 128-case support, with low_wind_idle counts 45/17/85.
- 对论文结论的影响：cleaning only Wiki-side pitch cannot repair a base trained
  earlier with those records; reporting four event names does not establish
  four functioning experts. Sparse unsupported events must remain explicit.
- 修复方案：leave the numerical dataset and power labels unchanged. Wiki may
  use a median only when two contemporaneous blade readings agree within 1
  degree and the third differs by more than 5; otherwise mark the context
  uncertain. Do not lower support thresholds based on outer validation.
- 验证修复的测试：consensus/nonconsensus cases, historical-only reads, no
  clipping of negative targets, no cohort removal.
- 置信度：高 for counts/scope; sensor-fault diagnosis remains unconfirmed.

## Reproducible v5 protocol

`configs/kelmarsh_wiki_quality_protocol_v5.json` is a new development protocol.
Numerical and untreated Wiki controls are reused from the completed v4 suite;
only the treated Wiki arm is trained. The first half of late calibration selects
epochs, the second calibrates the actual bounded correction, then one outer
development report is produced. Outer labels are not fitted or cleaned and the
sealed test is not inferred. This does not establish independent confirmation.

Launch: `bash scripts/train/Kelmarsh_launch_wiki_quality.sh`.
Status: append `--status`. After a genuine process failure, resume only missing
stages with `--resume`; completed neural checkpoints must be reused.

## Completed v5 result (2026-10-14)

All three declared fold/seed2024 runs completed successfully, 20 epochs per
Wiki arm, selected epochs 3/9/1 using only the internal training period. The
numerical prediction arrays were checked against the old cache and unchanged.
Results: `outputs/logs/Kelmarsh/20261014/001_wiki_quality_h2_f0-1-2_s2024_ep20`.

Three-fold unweighted means (macro R² is not pooled R²):

- Untreated WikiSelective: MAE 145.150652372 kW; RMSE 209.490632997 kW;
  MSE 44968.360081630 kW²; R² 0.743369590.
- Treated WikiQualitySelective: MAE 145.010081324 kW; RMSE 209.345746876 kW;
  MSE 44900.307835919 kW²; R² 0.743662445. Mean persistence MAE Skill
  -2.663315801%, RMSE Skill -0.605022947%; ±5%/±10% capacity hit rates
  52.759488505%/76.067323481% (not classification accuracy).
- Frozen numerical adapter: MAE 145.127339363 kW; RMSE 209.468342883 kW.
- Original Trend remains better: MAE 142.494880101 kW; RMSE 205.714966101 kW;
  R² 0.753201239.

The treatment improves final-policy macro MAE over the untreated policy by
0.140571047 kW (about 0.09685%), not a broad accuracy breakthrough. Per-fold
MAE gains versus untreated Wiki are +0.465555351, -0.011428704 and
-0.032413505 kW. Fold0's block-bootstrap interval is positive
[0.030178344, 1.240429931] kW but its Newey–West p=0.13297 is not below .05;
the other intervals cross zero. Do not select whichever test looks favorable.
Three folds sharing one seed and a historically inspected validation protocol
are development evidence, not independent confirmation.

Input-treatment counterfactuals explain the scope: same old weights with new
summaries give raw MAE 144.933780082 versus untreated raw 144.930634999;
retraining the treated raw expert gives 144.960731168. Thus raw candidate MAE
did not improve; the small final-policy benefit concerns routing/calibration,
not demonstrated sensor cleaning. All train and outer Wiki query audits have
zero repaired pitch hours. Only event0/state5 (full available capacity, normal
rotation) retains sufficient bank support, with 136/151/394 cases.

Effective intervention rates are 4.42177%, 10.65831% and 43.42404% by fold.
Folds1/2 still have negative selected gains and harm rates 54.41176%/56.65796%.
Fold2 benefit calibration has just 56 donor windows; correlated windows are
not 56 independent weather situations. Preserve these failures. Do not tune
new thresholds on these outer results, silently change the numerical fallback,
or claim that Wiki-only treatment fixed the dominant numerical regression.
