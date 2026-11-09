# STAGE_PLAN — 只写下一个阶段

作者：Claude（Codex 只提意见，不改本文件）。上限 150 行：要加新内容时，先删旧内容。
内容只从现有代码和结果中提取；拿不准的地方标「待确认」；事先设定的阈值标「事先设定」。
M2 的方案原文（CW1、T1 全部规则）见 `git show 0b3eaf9:docs/STAGE_PLAN.md`；CW1 结果和复核见 DECISIONS.md 的 M2 一节。

## 阶段名

M3：C 线第 2 轮（CW2，CPU），与正在运行的 T1（GPU）并行。
- T1：已在 M2 审定并于 2026-11-09 启动（PID 674567），规则不变，跑完后结果记在 M2 一节。
- L1：规则见 `git show a1ebfa0:docs/STAGE_PLAN.md` 的 L1 条，仍等模型权重；不与 T1 同卡同时运行。
- C 线一共最多做 3 轮，这是第 2 轮。

## 目标

1. **CW1 告诉我们的事（只作描述）**：情形信息试了 4 种加法，都没有超过"功率近邻"。
   - 4 种加法：近邻内加权、更强的密度比、分时效组合、按情形分组的 ACI。
   - C0b 里，M 在 h3–h4 比 B3 好。但到了新数据上反而更差：J1 的 h3、h4 Winkler 分别高 8.2、3.3 kW，J2（反向）高 60.8、60.4 kW。
   - 判断：用"加权"来做跨风场的协变量偏移修正，这个机制本身不对。这是换方向的理由。
2. **CW2 换一个机制**：不再用情形信息给校准样本加权，而是用它预测这次的误差会有多大（异方差）。
   - 依据：Kelmarsh 2024 的诊断显示，功率变化 ≥10% 容量的窗口贡献了 h4 MAE 的约 70%；上游的"变化类"特征（3 小时风速变化、气压变化、梯度变化）恰好能提前提示变化。
   - 检验：带上游变化特征的误差尺度模型，能否同时赢过只用预测功率的同类模型和现有最好的 B3-A。
3. **共用确认**：通过的线与底座合写一份 Hill of Towie 联合确认协议，封存年只用一次。

## 假设与判定标准

**CW2（事先设定）**
- **沿用的部分**：
  - 点预测器：正向用 P2 主折的 K-upstream，反向用 P2 主折的 P-upstream，都是 3 个 seed，只推理，各自用自己的 statistics。
  - A（ACI 截断 [0.005, 0.5]，各机组各步长，按真实延迟反馈）、规则 F、非有限值的处理，全部照旧。
- **方法：标准化 conformal，加一个学出来的误差尺度模型 σ̂(x)**：
  - 每个 h 一个 HistGradientBoostingRegressor：loss = absolute_error、max_iter 200、learning_rate 0.05、max_leaf_nodes 15、min_samples_leaf 200、l2 1.0、random_state 2024。
  - 拟合目标是源校准年的 |残差|。只用源校准年拟合，不用目标风场的任何数据。
  - σ̂ 设下限：max(σ̂, 0.1 × 训练集 |残差| 的中位数)。
  - 分数 s = |r| / σ̂(x)，用源校准集算分数。半径 = σ̂(x_test) × Q_{1−α_t}(s)，α_t 由 A 更新；分位点落在 +∞ 质量上时，按 F 取组内最大分数。
- **三个臂（只有特征不同，其余完全相同）**：
  - N-P（匹配对照）：只用该 h 的预测功率。
  - N-R（候选）：预测功率，加上起报时刻的上游特征：|upwind_dws3|、|local_dws3|、|regional_dws3|、|local_dp3|、|grad_magnitude_change3|、upwind_minus_local、regional_ws、upstream_missing，以及最近站风向的 8 个扇区（缺测单列一类）。
  - N-Rs（候选，只要变化类）：预测功率，加上 |upwind_dws3|、|local_dws3|、|regional_dws3|、|local_dp3|、|grad_magnitude_change3|。
  - 所有特征在起报时刻都已知，不含标签。
- **其他对照**：B3-A（现有最好）、B0、B1、B2-A，与 C0b、CW1 的定义相同。
- **挑选**：在 4 组已用过的数据上挑：Penmanshiel 2022、2023（正向），J1 = Penmanshiel 2024（正向），J2 = Kelmarsh 2024（反向）。校准设定都和之前相同。
  - 合格条件：4 组的覆盖率都在 [88%, 92%] 之内，并且 4 组都比 min(N-P, B3-A) 的 Winkler 低。
  - 合格的候选里，取相对 min(N-P, B3-A) 改善百分比（4 组等权平均）最大的；平局取 N-Rs。
  - 没有合格的候选，CW2 就停在挑选阶段，不判定，进入第 3 轮的设计。
  - 选完先写 selection_freeze.json，再计算判定数据。
- **判定数据**：J3（反向，事先预留）。源 Penmanshiel 2022，目标月 Kelmarsh 2022-12，评估 Kelmarsh 2023；校准截止 2022-12-31 23:00，并做因果断言。
  - P-upstream 只用了 Penmanshiel 2016–2019 拟合、2020 选择。
- **判定（J3，3 个 seed 平均，h1–4 合并，90% 名义覆盖）**：
  - C1：覆盖率在 [88%, 92%] 之内。
  - C2：Winkler ≤ 0.95 × min(B0、B1、B2-A 中覆盖率 ≥88% 的；都不到 88% 就取三者全部)。
  - C3：Winkler ≤ 0.97 × min(N-P, B3-A)，并且"较好的那个对照 − 候选"的 95% CI 下界 > 0。
    - CI 的做法同 CW1：按起报时刻做 block bootstrap（块长 8、5000 次、seed 2024），同一时刻的所有机组和 h 一起重采样，seed 先平均。
  - C1–C3 全过 → 进入 Hill of Towie 联合确认。
  - 没过 → 第 3 轮（最后一轮）。判定数据预留 Kelmarsh 2022（反向），方向等 CW2 结果出来再定。
- **只描述**：每个 h 的结果；σ̂ 的特征重要性；变化窗口和平稳窗口里分别的覆盖率和宽度（按真实变化分层，只用于描述）。
- **新意预估（如果通过）**：标准化 conformal 本身是已有方法（Papadopoulos 2008、Lei 2018）。我们的新意在于：用公开上游站点的变化指标做冷启动风场的异方差区间。约 4.5–5。

## 实验配置表

| 线 | 内容 | 规模 |
|---|---|---|
| CW2 | 3 个臂 + 对照；4 组挑选数据；J3 判定 | 不训练点预测器；CPU（2 个线程）约 1–2 小时（待确认） |
| T1 | 已在运行（M2 规则） | GPU，完成后交回 |
| L1 | 等权重 | — |

- CW2 只运行一个实例。代码和选择记录在计算 J3 之前冻结。

## 改动文件清单（= HANDOFF「允许改动」）

- CW2：`scripts/pilot_conformal_cw2.py`（新文件，导入冻结的 C0、C0b、CW1，只加尺度模型和 J3 设定）、`tests/test_pilot_conformal_cw2.py`、`output/conformal_cw2/**`。
- T1、L1：同 M2。

## 本阶段不做什么

- 不碰 Hill of Towie 封存年；不在 J3 上挑选或调参。
- 不改已冻结的 C0、C0b、CW1 文件和结果；不改 T1 的运行。
- 不在看到 J3 结果后改门槛、特征或超参数。

## 风险（≤5 条）

1. 误差尺度是在源风场上学的，迁移到目标风场后，尺度关系可能变化（CW1 的 J2 就显示，反向迁移时差异很大）。ACI 只能修正整体水平，修不了形状。
2. 上游变化特征里约 10% 缺测，缺测时 N-R 退化为接近 N-P；缺测子集要单独报告。
3. 只有一组判定数据（J3）。用"挑选数据 4 组全部方向一致"作为合格条件来补强。
4. 看过 CW1 结果后才提出的方向，有数据窥探的风险。用新的判定数据并在论文里披露来缓解。
5. CW2 的 CPU 任务和 T1 同机运行，CW2 限 2 个线程。

## 存档引用

- CW1：`output/conformal_cw1/RESULT_SUMMARY.md`、`r1/{selection_2022,selection_2023,J1,J2}/seed_*.json`。
- C0b：`output/conformal_c0b/RESULT_SUMMARY.md`；变化窗口诊断：DECISIONS.md 的 G0 一节。
- 冻结站点：Kelmarsh v32 的 36 站；Penmanshiel 的 11 站；Hill of Towie `output/hill_of_towie_e1/l0b_r2/qualified_stations.json`（5909e143…）。
