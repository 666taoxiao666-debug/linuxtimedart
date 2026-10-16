# Kelmarsh model review and forecast-relative Wiki pilot

## 2026-10-16：外部分析报告的逐项核对与改动边界

以下核对针对用户提供的分析报告与当前24→4/v21—v22证据；下文2026-10-14审查是
历史Wiki v1范围，不能把其中“7/9弃权”推广为所有后续Wiki版本的统一统计。

【最终评级】：C。报告抓住了可用天气信息有限、早期过适应、强线性基线有竞争力、
跨通道交互值得检验等合理方向，但多处把假设或局部实验夸成已确认根因。
这不是创新评分，也不是对整个仓库无泄漏的认证。

【必须立即修复 Top 5】：

1. v22失败不能证明预训练无效：它没有随机初始化/不预训练的匹配对照。
2. Epoch0与Epoch1不能混同：只有零残差起点是Persistence，训练1轮不等于自动回退。
   不删除安全起点；应同时报告选择epoch、训练后最佳/最后结果和真实Skill。
3. 通道独立不等于完全不能拟合跨变量关系；已有后置MLP/context。原建议同时换通道
   结构和取消预训练，无法把结果归因于早期交互。
4. 不提供NWP限制可用信息，但未测量其“首要天花板”，不等于任务不适定；不能把
   未来实测/再分析天气直接当可用预报，不能删除停机样本抬高总体指标。
5. 校正运行身份和统计范围：v21实际796082参数；v22只做内部评估；Wiki“7/9弃权”
   是历史v1。不能把这些说成当前0.896M模型、所有版本、或正式封存test结论。

【需重跑的实验与需补充的单元测试】：不重跑v19—v22。新增v23两个等参数分支：
相同的同历史patch跨变量注意力，分别放在原时间编码器前/后，且都保留相同v20
预训练、v21功率恒等旁路、零残差起点、数据/Scaler/loss/LR/预算。只检验位置差异，
不声称一次实验能证明通道独立是根因。先完成两个分支×三fold的内部候选，固定门控
通过才一次评估全部六个外层候选。关键测试覆盖跨变量交互不跨样本/patch、真实放置
位置、身份初始化、等参数/等初始权重/RNG、真实训练梯度/LR、门控、历史分组不删样本，
以及启动环境快照与续跑身份；测试值不作为真实实验精度。

=== 详细问题清单 ===

- 严重级别：[P1，外部报告的因果过度解读]；位置：
  `scripts/train_kelmarsh_frozen_transfer.py -> decision/run`，
  `configs/kelmarsh_frozen_transfer_protocol_v22.json -> outer_gate`。
  证据：v22内部宏MAE/RMSE为158.691115/237.609277，v21为158.788086/235.683589；
  v22仍优于内部DLinear/Persistence，但未联合胜过v21。它不是“无法支持预测”的证明。
  影响：不能据此废除预训练或写完整因果结论。修复：降为冻结表示不足以解决泛化的
  局部证据，暂不改变预训练。验证：若以后研究预训练价值，需同结构、同更新策略的
  scratch/重构/预测导向对照，明确各自标签与计算预算；置信度高。

- 严重级别：[P1，外部报告建议的评价偏移]；位置：
  `models/TimeDART.py:1248 -> restore_scale`及
  `scripts/train_kelmarsh_hourly.py:239 -> train_stage`。
  证据：零头只定义epoch0为Persistence；v21选择epoch3/3/1，非三折epoch0。
  MAE与含MSE的混合目标存在取舍，但不是错误实现。影响：去掉零头会改变起点并可能
  使基线退化更难识别，不是“去伪优化”。修复：保留，并报告全部步长和训练后表现。
  验证：新旧初始化预测与Persistence严格一致的单测保留；置信度高。

- 严重级别：[NEEDS-EVIDENCE]；位置：`models/TimeDART.py:1142 -> forecast`、
  `layers/TimeDART_EncDec.py:24 -> ChannelMixer`。证据：编码器内按变量分批是真的，
  但后置MLP/context已有跨变量函数；v21并非逐折稳健胜过v20，仍不及DLinear。
  影响：不能宣称“完全丧失风速功率函数能力”或已证明它是首要瓶颈。修复：只做
  早/晚同模块匹配消融，参数增加的收益也必须对照晚置分支。验证：位置/参数/RNG/
  输入队列/预训练来源锁定与训练内部门控；对根因排序置信度低。

- 严重级别：[NEEDS-EVIDENCE]；位置：`data_provider/kelmarsh_hourly.py:80 ->`
  前缀Scaler与`utils/physics_normalization.py:7 -> normalize_history`。
  证据：Scaler60%是已记录的控制设计，v20拟合90%；当前模型已经窗口归一化并保留
  风速绝对信息。全局非退化仿射标准化本身不删除信息，也不推出功率正偏差的符号。
  影响：“夏季尺度必然造成全面高估”未经本数据量化。修复：本轮不换Scaler或重复加
  RevIN；以后需仅训练侧分布审计及匹配重预训练，不能混用旧归一化检查点。
  验证：原Scaler/窗口SHA与旧控制一致；对该因果解释置信度低。

- 严重级别：[P2，外部报告的身份/范围错误]；位置：v21各fold的
  `manifest.json -> parameters`、`docs/KELMARSH_MODEL_REVIEW.md`历史v1范围。
  证据：当前参数796082，无外部LLM权重载入；历史Qwen参数名不是权重使用证据。
  Wiki v1统计不能覆盖后续质量处理及神经Wiki。影响：误导论文方法身份和模块去留。
  修复：记录真实工厂/参数/初始化checkpoint，当前Wiki仍关闭但不删除历史代码或结果。
  验证：新runtime_identity记录实际工厂、LLM checkpoint=None，参数从实际模型计数；
  对当前运行身份置信度高，对未审查的所有历史实验不作统一断言。

### 接受、修正或暂不执行的建议

接受：强基线、分步长Skill、仅基于历史状态的分组诊断、有限的早期跨通道交互实验、
真实运行身份和复现日志。新分组按最后历史功率/额定容量<5%、5%—90%、≥90%划分；
这只是观察分组，不声称等同停机/额定工况，不删任何评估窗口、不作为路由/调参标签。

暂不执行：接入NWP（须先拿到含issue_time、valid_time、实际发布时间/延迟的历史
预报，验证issue_time/availability≤预测起点，不能用未来实测或再分析替代）、缩到
任意5—10万参数、改变loss、换Scaler或预测导向预训练。这些都需独立受控实验，不能
同一次改完后把收益归给任一机制。Ridge拟合不是“零训练成本只读诊断”；原全局Ridge
已经存在，局部归一化Ridge是新训练变体，不能直接把其效果等同Transformer无用。

基础文献核对而非替本项目担保：[PatchTST作者论文](https://arxiv.org/abs/2211.14730)
展示通道独立和掩码预训练在其基准可有效，反驳“结构/预训练必然无效”的普遍断言，
但不证明本风电数据也有效。[RevIN作者说明](https://seharanul17.github.io/RevIN/)
描述分布漂移归一化思路，不能证明本项目尚未采用。
[IEA Wind研究论文](https://iea-wind.org/wp-content/uploads/2021/04/IEATask36-Torque-2016_FullPaper.pdf)
讨论NWP和几小时以上预报的重要性，不能替本数据量化不可超越的误差下限。

新试验协议`configs/kelmarsh_channel_timing_protocol_v23.json`，SHA256
`43f17d453a2b52eda6c1d102a72d9d89f7daba9d78c49378a0afb2ae8e58b6a4`；
实现/部署/真实结果进度统一写入[数值底座记录](KELMARSH_NUMERIC_FOUNDATION.md)，
本审查不预填收益、不冒充已完成训练或新的论文创新。

---

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

The 2026-10-14 literature-grounded continuation is recorded in
`output/literature-search/wind-wiki/papers.md` and the working manuscript
`manuscript/wind_event_wiki.md`. Its timestamped episode-balanced readout is an
opt-in prototype in `utils/kelmarsh_episode_wiki.py`, not a replacement for the
frozen v1/v2 experiments. A three-fold training-only probe showed worse MAE/RMSE
in both fold0 utility periods, better in both fold1 periods, and no action in
fold2. The evidence does not justify a global accuracy-improvement claim.

- [Nearest Neighbor Multivariate Time Series Forecasting](https://ieeexplore.ieee.org/document/10752995/)
  (IEEE TNNLS 36(7), 2025, DOI 10.1109/TNNLS.2024.3490603; method;
  publication status reverified 2026-10-14): cached nearest-neighbor forecasting already
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

## Completed pilot evidence (2026-10-14)

All three declared folds, seed 2024, completed with the same frozen numeric
checkpoints, scaler, outer window timestamps and raw targets as v1. Reproduced
base predictions were checked against v1 saved arrays before new evaluation.
These are **three seed-2024 runs**, not the previous nine-run macro average.

- Trend: macro MAE 142.494880 kW, RMSE 205.714966 kW, R² 0.753201.
- Legacy Wiki: 143.294329 / 207.278705 / 0.749903.
- ParetoWiki (predeclared primary): 142.681657 / 206.606880 / 0.751329.
- MedianLocalWiki (predeclared mechanism control): 142.481954 / 206.548721 /
  0.751319. Its MAE advantage over Trend is only 0.012926 kW, while RMSE is worse;
  do not promote this ablation to primary or claim joint improvement.
- Matched-label NumericControl: 141.998796 / 207.024948 / 0.750443. It also trades
  a better MAE for a worse RMSE than Trend.

ParetoWiki reduces legacy damage but **does not beat the actual base**: macro
MAE is worse by 0.186776 kW and RMSE by 0.891914 kW. It intervenes only in fold 0.
In that fold, early/late train-utility gains were +2.357812/+2.277824 kW, whereas
outer gain was -0.560329 kW. This directly shows that empirical local dual-loss
safety plus two train blocks still does not establish transport to a later
period. The other folds have no accepted primary cells and exact abstention.
The primary mechanism has not established the desired positive contribution.

The additional current-state information probe also **does not support blindly
adding more inputs**. Old -> augmented MAE/RMSE in later training blocks:

- f0 early: 112.927149/144.280594 -> 111.758396/144.130896 kW (both improve).
- f0 late: 137.258014/202.615489 -> 140.476928/205.503757 (both worsen).
- f1 early: 193.499833/284.771846 -> 194.090848/284.874990 (both worsen).
- f1 late: 165.155931/238.147176 -> 166.062465/237.748306 (trade-off).
- f2 early: 139.832134/197.289560 -> 141.027981/198.472281 (both worsen).
- f2 late: 109.526795/162.756615 -> 109.866257/165.175169 (both worsen).

This is one fixed-alpha Ridge probe, not proof that these inputs cannot help
nonlinear retrieval or a neural base. The confirmed coordinate alias remains,
but it is not established as the dominant performance cause. No phase-augmented
outer search, new seed sweep, or rare-event threshold relaxation was launched.

### Research decision

Keep the new readout as a tested development branch, not a proven accuracy claim.
The immediate bottleneck is **predictive transport evidence**, not a missing
method name: only one event family has support; local distributions and benefit
signs shift; the base advantage over Persistence is already small. The next
substantive study should distinguish (a) forecasting-relevant rather than merely
reconstructive pretraining, and (b) event-duration/control-state-aware transition
knowledge with sufficient observed cases and an untouched later period. It must
not use a more elaborate gate as a substitute for that evidence. No subsequent
outer refit or tuning is included in this completed pilot.
