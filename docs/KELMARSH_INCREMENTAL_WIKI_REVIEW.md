# Kelmarsh Wiki-only incremental correction: implementation and experiment review

## 【最终评级】：C

The new implementation can be tested as a development hypothesis after its
critical-path tests and exact frozen replay pass. Accuracy and transportability
are not established. This is a targeted data-flow review, not certification of
all historical experiments. No new outer results exist at protocol creation.

## 【必须立即修复 Top 5】

Three substantiated concerns, rather than five invented findings:

1. Train Wiki as a bounded increment above its actual frozen numerical fallback,
   not as another independently learned full residual above OriginalTrend.
2. Preserve the distinction between candidate identity and time-transfer evidence.
   The original calibration counts do not establish independent weather support.
3. Do not present event attribution IDs as different neural expert outputs.

## 【需重跑的实验与需补充的单元测试】

Do not retrain v1-v5 or their numerical paths. The new v6 protocol declares one
fold1/seed2024 development pilot, chosen for the largest training-side calibration
time support. It fits two new Wiki-side incremental modules, 20 epochs each:
typed-memory inputs versus zero-memory inputs. Capacity, initialization, source
labels, support masks, optimizer and selection criterion are matched. An unchanged
NumericContext and the completed v5 Wiki results are reused controls.

Tests cover zero-init identity; exact zero increments outside support after
training; label-independent inputs/inference; case deltas rebased to the actual
numeric forecast; unsupported event masking; equal ablation initialization and
capacity; unchanged numeric arrays; completed-arm reuse; strict replay hash
rejection; training-only time blocks; and waiting rather than sharing a busy GPU.
Synthetic unit tests are not prediction-quality experiments.

## === 详细问题清单 ===

### Independently refitting a full residual obscures the Wiki increment

- 严重级别：[P1] for attribution of an accuracy improvement; not proof of leakage.
- 位置：`scripts/train_kelmarsh_contextual_expert.py:76` -> `tensor_data`,
  `(truth-reference)/capacity`; `train_arm` at line 98 independently initializes
  NumericContext and WikiContext; `infer` returns reference plus the fitted head.
- 触发条件及证据：the old numerical adapter is frozen for this task; Wiki's
  separately retrained full residual may depart from it even where useful case
  evidence is absent. Most v4 deterioration versus OriginalTrend already exists
  in the numerical adapter, so assigning all deterioration to Wiki is incorrect.
- 对论文结论的影响：overall gain versus a numerical baseline is not automatically
  a gain from retrieved knowledge; a Wiki-only change cannot repair every error
  of an unchanged fallback.
- 修复方案：`models/IncrementalWindWiki.py` has no numerical forecaster in its
  parameters. Its prediction is exactly frozen NumericContext plus a bounded,
  zero-initialized increment. The increment is exactly zero outside the same v5
  support mask. Train against `(truth-frozen_numeric)/capacity`, reuse the case
  bank, and compare to an identical zero-memory incremental module. No baseline
  checkpoint, raw input, scaler, label or cached outer numeric output is changed.
- 验证修复的测试：`tests/test_kelmarsh_incremental_wiki.py` identity, masking,
  feature rebasing, label mutation, matched capacity and completed-arm reuse.
- 置信度：high for execution/attribution issue; whether the fix improves metrics
  remains unknown until the declared experiment finishes.

### Sparse, correlated calibration is not a temporal certificate

- 严重级别：[P2].
- 位置：`scripts/train_kelmarsh_contextual_expert.py:187` -> `fit_certificate`,
  especially the window/turbine support and aggregate gain tests at lines 201-205.
- 触发条件及证据：the existing training-only audit reproduced every saved cell.
  Fold2's accepted cell has 30 windows but 10 issue times and only 2 UTC day blocks.
- 对论文结论的影响：passing this rule does not prove stable transfer over time.
  Multiple turbines in the same weather cannot be treated as independent trials.
- 修复方案：v6 first performs an exact old-checkpoint replay on fresh calibration,
  verifies both saved input/output SHA256 values, saves actual paired outputs,
  and reports each supported cell's daily gains, dispersion and sign. It does
  **not** add a validation-chosen day-count threshold or simultaneously claim a
  new temporal gate. Old utility thresholds remain fixed to isolate the
  incremental-executor hypothesis. Time-aware evidence calibration is deferred
  as a distinct mechanism, contingent on these training-only findings.
- 验证修复的测试：strict output perturbation rejection, train-boundary rejection,
  blockwise diagnostic and concentration audit tests.
- 置信度：high for missing temporal support; its causal contribution to outer
  harm remains unproven.

### Event attribution is not independent neural-expert selection

- 严重级别：[P2] for an unsupported architecture claim.
- 位置：`scripts/train_kelmarsh_contextual_expert.py:214` -> `route`, line 234
  chooses between one `candidate` and `numeric` despite returning an event ID.
- 触发条件及证据：the forecast candidate has no event axis in the executed output.
- 对论文结论的影响：four named events do not establish four executing neural experts.
- 修复方案：v6 explicitly uses a single Wiki increment with event-conditioned
  evidence and two actual actions (unchanged versus corrected). No multi-expert
  accuracy claim is made by renaming IDs.
- 验证修复的测试：new model and route shapes remain [window,horizon]; no fabricated
  event-expert axis is introduced.
- 置信度：high.

## Frozen experiment and success rule

Protocol: `configs/kelmarsh_incremental_wiki_protocol_v6.json`.
Only memory/early donor labels fit new parameters. The first chronological half
of late calibration selects the checkpoint jointly on MAE and RMSE relative to
the unchanged numerical forecast, with epoch0 fallback. The disjoint second
half establishes a fresh certificate bound to each new candidate checkpoint.
Both arms finish training and calibration before outer evaluation is called.

The method must beat both unchanged NumericContext and its matched zero-memory
selective ablation on **both** MAE and RMSE before claiming a Wiki accuracy
contribution. Also report MSE, R², persistence-relative Skill, capacity-tolerance
hit rates, per-lead metrics, coverage, selected gain, harm, abstention and paired
block-bootstrap/Newey-West results. Beating only the old Wiki is insufficient.
OriginalTrend remains an explicit stronger internal reference. Inspected outer
validation is development evidence, not an independent confirmation set.

The launcher has a single-run lock and waits at most 24 hours for an idle GPU.
It never kills or competes with another GPU process. A successful strict replay
with nonzero supported-candidate oracle headroom is required before fitting;
the oracle is only retrospective diagnostic, not an achievable performance
claim. Hash disagreement stops the job, rather than modifying old evidence.
Completed replay/training stages are reused on resume. Report-only fixes do
not change the training-source digest and must not cause completed-arm refits.

## Operations

Start: `bash scripts/train/Kelmarsh_launch_incremental_wiki.sh`.
Status: append `--status`. After a genuine stopped/failed process, append
`--resume`; never restart the old v5 suite. The new pointer is
`outputs/logs/Kelmarsh/incremental_wiki_latest.txt`. A queued job reports
`WAITING_FOR_GPU`; that is not a claim that model training has started.

No automatic seed/fold expansion, new Wiki thresholds, sealed-test inference,
numerical fallback changes or destructive result cleanup are authorized by this
protocol. A negative result is retained and ends this particular experiment.

## Deployment observation: 2026-10-14 14:40 Asia/Shanghai

Implementation commit `11cc4a1` was pushed and fast-forwarded on the server.
All 37 relevant tests passed locally and on the server; Python compilation,
Bash syntax, source-protocol and data hashes passed. Protocol SHA256:
`48f322088d9e444b7a082a5c66a64f65f34a970e52136eda498dfd585eb41be8`.

The launched job was observed waiting, not training: Bash PID 1836763,
Python PID 1836772, status `WAITING_FOR_GPU`. Another user's GPU job remained
untouched. Result directory:
`outputs/logs/Kelmarsh/20261014/001_incremental_wiki_h2_f1_s2024_ep20`.
Outer launch log:
`outputs/launch_logs/incremental_wiki_20261014_144016.log`.
Use the status command/pointer for live state rather than this dated note.

The queue is a single server experiment launcher, not a restored Codex
recurring automation. It starts the declared pipeline when the GPU becomes
idle, but does not provide periodic conversational notifications. No new
accuracy numbers are available at deployment; synthetic test metrics must not
be substituted for the real experiment.

## Completed-result numeric audit: 2026-10-15

Mode: quick numeric/claim audit of the completed fold1/seed2024 pilot, not a
new experiment or a full manuscript review. The original queued process
completed normally on 2026-10-14 around 18:02 local server time. No shared-GPU
mode was implemented or launched; the original idle-only strategy remained.
At the next inspection the GPU had no compute processes. Do not restart this
completed experiment or expand its folds/seeds.

Both WikiIncrement and ZeroMemoryIncrement completed all 20 epochs; their
training-only selection chose epoch3. The zero-memory arm retains the same
observed context and support mask, so it isolates the retrieved memory features,
not every possible use of knowledge. The original calibration input/output
replay was bit-identical. All 34 artifacts registered by the server's
`completed.json` passed SHA256 verification.

The final WikiIncrementSelective metrics are MAE **107.2067030973 kW**, MSE
**26567.2299595218 kW²**, RMSE **162.9945703375 kW**, R² **0.7617489661**,
Persistence-relative MAE Skill **−5.2192747587%**, and RMSE Skill
**−4.8391979658%**. Capacity tolerance hit rates are 64.8119122257% within
±5% of 2050 kW capacity, and 85.3448275862% within ±10%; these are not
classification accuracy, and both match the unchanged numerical control.

The unchanged numerical control has MAE 107.2369139300 and RMSE 163.0183172547.
Wiki therefore improves them by only **0.0302108327 kW** and
**0.0237469172 kW**. Its MAE-gain 95% temporal-block bootstrap interval is
[−0.0248542039, 0.1167059972] kW, and the Newey–West two-sided p-value is
0.3984361149. The primary estimand is window-uniform MAE, temporal clusters are
distinct forecast issue times, block length is 8, bootstrap replicates 5000,
seed2024. The saved Newey–West statistic is the ratio-mean influence test,
not an unweighted time-mean DM test.

ZeroMemoryIncrementSelective has MAE **107.2092946349** and RMSE
**162.9921781414**. Relative to it, retrieved memory improves MAE by just
**0.0025915375 kW** while worsening RMSE by **0.0023921961 kW**. The MAE-gain
interval [−0.0322661361, 0.0385076183] crosses zero, p=0.8840011735.
The prespecified joint-improvement criterion is therefore **false**. Do not
claim a demonstrated additional accuracy benefit from retrieved memory.

New Wiki's intervention-window coverage is **21.1598746082%**, abstention
**78.8401253918%**, selected-point mean gain **0.1554396068 kW**, and harmful
intervention-window rate **46.6666666667%**. The zero-memory selective arm has
coverage 10.6583072100%, abstention 89.3416927900%, selected-point gain
0.2591339750 kW, and harm rate 38.2352941176%. Coverage and harm are window
statistics, whereas selected gain is over actually intervened horizon points.

For context, old WikiQualitySelective has MAE 107.3182816597, RMSE
163.1230147249, R² 0.7613733209, coverage 10.6583072100%, and harm
54.4117647059%. New Wiki improves its MAE by 0.1115785623 kW, but this
comparison's interval also crosses zero, p=0.4347820943. OriginalTrend is still
better: MAE **101.8888443279**, RMSE **155.4710202472**, R² **0.7832358704**.
Its errors in this fold are numerically almost identical to Persistence; do
not count this as another independent strong-baseline win. The current
numerical adapter's deterioration cannot be assigned entirely to Wiki.

The evaluation contains 638 windows / 1276 points / 107 distinct forecast issue
times. These are one historically inspected development fold, not a three-fold
macro result or an independent held-out confirmation. The negative joint result
is retained. No new thresholds, seeds, epochs, labels, numeric inputs or model
weights were selected from these outer results. A new mechanism or changes to
the non-Wiki numerical path require a separate user decision.

Raw downloaded evidence and the reproducible numeric audit are under
`output/kelmarsh_hourly/incremental_v6_review/`. The compact result panel and
saved forecast figures present these same measurements; no synthetic test
numbers are substituted.

Local recomputation from seven aligned prediction arrays reproduced the saved
MAE/MSE/RMSE/R² and all new intervention statistics. The 27 registered artifacts
included in the downloaded review subset also passed local SHA256 checks; the
server verified all 34, including files intentionally not downloaded. The 126
exported report/control files are read-only copies. Target times run from
2016-10-19 12:00 through 2016-11-12 22:00 UTC. The overview and representative
forecast-example images were visually inspected; their displayed metrics match
the raw predictions. The result panel's live rendering was queued by the app,
not independently screenshot-verified.

Claim-evidence disposition: successful execution and exact replay are supported;
small point-estimate improvement over unchanged numeric and old Wiki is supported;
statistically significant gain is unsupported; extra retrieved-memory accuracy
contribution fails the declared joint criterion; cross-fold or independent-test
generalization is unverified. No citation or full manuscript review was attempted.

On completion, a native pause request for the existing `kelmarsh-gpu` monitor
returned that the automation no longer exists in the app. It was not recreated;
do not claim that a successful pause operation occurred. This is distinct from
the server experiment, whose completed state and retained results are verified.
