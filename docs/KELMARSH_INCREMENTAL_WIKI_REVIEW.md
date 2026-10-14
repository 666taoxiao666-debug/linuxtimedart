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
