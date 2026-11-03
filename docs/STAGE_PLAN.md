# STAGE_PLAN — 只写下一个阶段

作者：Claude（Codex 只提意见，不改本文件）。上限 150 行：要加新内容时，先删旧内容。
内容只从现有代码和结果中提取；拿不准的地方标「待确认」；事先设定的阈值标「事先设定」。
长方案 `docs/UPSTREAM_INNOVATION_PROPOSAL.md` 已改为存档，只按章节号引用，不再整份送审。

## 阶段名

v35 P2：在 Penmanshiel 开发折上复现上游特征增益，并做 Kelmarsh → Penmanshiel 迁移（开发证据）。

背景：P1 数据准备通过（Q1–Q4 全过，起点 2016-08-01）；数据只在服务器上，元数据在 `output/penmanshiel_hourly/prep_v35/`。

## 目标

1. **复现**：在 Penmanshiel 本场训练时，加上游特征的 TSMixer 是否优于不加的（即 Kelmarsh v32 的 C1 在第二风场上能否复现）。
2. **零样本迁移**：只用 Kelmarsh 训练的模型直接预测 Penmanshiel，不用 Penmanshiel 的任何标签或统计量，能否胜过持续法；上游特征能否跨站网迁移。
3. **少样本**：Penmanshiel 只有 1 个月或 3 个月数据时，用 Kelmarsh 预训练再微调，是否优于从零训练。

## 假设与判定标准

- **评估**：Penmanshiel 2021 年（开发评估年）ALL 队列，h1–4 合并，单位 kW。
- **方法**：3 个 seed 的损失先平均，再做 v32 的统计——
  - MAE 用 Newey–West 检验，按起报时间聚类；
  - RMSE 用起报时间块自助法：块长 8，5000 次；
  - MAE 的 p 值在本节 5 个比较上做 Holm 校正。

| 编号 | 候选 vs 参照 | "成立"的条件（事先设定） | 成立 → / 不成立 → |
|---|---|---|---|
| R | P-upstream vs P-last10 | MAE 增益 CI95 下界 > 0 且 Holm p ≤ 0.05，且 RMSE 增益 CI95 下界 > 0 | 进入 P3，用 2022 做一次性确认 / 交用户决定是否仍做 P3 |
| Z1 | K-upstream（零样本）vs PersistenceHour | 同上 | 记为"零样本可用" / 如实记为负结果 |
| Z2 | K-upstream（零样本）vs K-last10（零样本） | 同上 | 上游特征可跨站网迁移 / 不能 |
| F1 | 1 个月：FT vs Scratch | 同上 | 预训练在 1 个月数据下有用 / 没用 |
| F3 | 3 个月：FT vs Scratch | 同上 | 同上，3 个月 |

- **只描述、不判定**：
  - K-upstream 零样本与 P-upstream 本场训练之间的差距；
  - normal 队列上的结果；
  - 分 h1–4 的结果。
- **定性**：这些都是开发证据，单场单年，不作为论文的确认结论。确认留给 P3，用 2022 年一次性完成。

## 实验配置表

公共设置：

- 模型族与配方沿用 v32：TSMixerBasicSmall（blocks 2、ff 32、dropout 0.1）。
- 训练：AdamW，lr 1e-3，wd 1e-4，batch 64；最多 20 个 epoch，patience 5，cosine 调度；损失 0.8·MAE + 0.2·MSE（标准化功率）。
- checkpoint 选择：按验证集原尺度 MAE 选，epoch 0 计入。
- 其他：seed 2024/2025/2026，输入 24、预测 4、步长 3，滞后 1 h。

| 臂 | 训练数据 | 通道 | 用于 | 拟合次数 |
|---|---|---|---|---|
| PersistenceHour、PersistenceLast10 | 不训练 | — | 参照 | 0 |
| P-last10 / P-upstream | Penmanshiel：拟合 2016-08–2019，验证 2020 | 9 / 21 | R | 2 × 3 = 6 |
| K-last10 / K-upstream | Kelmarsh：拟合 2016-05–2019，验证 2020（和 Penmanshiel 同日历年，以免训练期覆盖评估年 2021） | 9 / 21 | Z1、Z2，以及 FT 的起点 | 2 × 3 = 6 |
| FT-N | 从 K-upstream 同 seed 的权重出发，在 Penmanshiel 最后 N 个月（N=1：2019-12；N=3：2019-10 至 12）上微调，lr 1e-4 | 21 | F1、F3 | 2 × 3 = 6 |
| Scratch-N | 同样 N 个月，从零训练，配方同上 | 21 | F1、F3 | 2 × 3 = 6 |

- **零样本口径**：K-* 在 Penmanshiel 上推理时，scaler 和上游缺测的填补值都用 Kelmarsh 拟合年的值，不用 Penmanshiel 的任何统计量。两场额定功率都是 2050 kW，功率尺度一致。
- **少样本口径**：
  - FT 和 Scratch 都以 N 个月内时间上最后 20% 的数据作验证、做早停，不用 2020 年的标签。
  - FT 沿用 Kelmarsh 的 scaler 和填补值；Scratch 用这 N 个月自己的统计量。
- **预算**：24 次拟合加 2 个持续法，只跑 1 个真实实例。预计 ≤3 小时，具体待确认，参考 K1b 6 个臂共 37 分钟。

## 改动文件清单（= HANDOFF「允许改动」）

- `scripts/train_penmanshiel_p2_v35.py`：新文件，Codex 写。复用 v28 的 `build_model` / `fit_config`、`KelmarshUpstreamSeries`，以及 v32 的统计与判定函数；不改它们。
- `tests/test_train_penmanshiel_p2_v35.py`：Codex 写，用合成数据，测试要覆盖：
  - 零样本不读 Penmanshiel 统计量；
  - 少样本的月份切分和验证切分；
  - 判定分支。
- `output/penmanshiel_hourly/p2_v35/**`：结果，不拷权重和预测数组。
- 只读不改：P1 的准备代码与数据、Kelmarsh 的全部冻结代码和协议、v32 的结果。

## 本阶段不做什么

- 不碰 Penmanshiel 2022 年及以后的数据，也不碰 Kelmarsh 2021 年以后的数据作训练（K-* 只用 2016–2020）。
- 不追加臂，不调超参数，不换站，不改特征定义。
- 不做 P3；不作显著性"确认"的表述，P2 只算开发证据。
- 不重做 K1、K1b、D2-0 的任何路线。

## 风险（≤5 条）

1. Penmanshiel 最近的站在 63 km 外（Kelmarsh 是 27 km），R 可能不成立（P1 的 R1-S2）。这是要检验的问题，不在 P2 里补救。
2. 零样本时两场的输入分布不同（地形、站网距离、机型 MM92 和 MM82 的功率曲线不同），Z1 可能不过；这一项只记录，不调。
3. 少样本的验证集很小（1 个月的 20% 约 6 天），早停可能不稳定，seed 之间的差异会偏大。
4. Kelmarsh 的 K-* 只用到 2020 年，训练数据比 v32 少，零样本的能力会低于 v32 的模型。
5. 5 个比较、单场单年，开发阶段可能出现偶然显著；所以用 Holm 校正，并且只把 R 作为是否进入 P3 的依据。

## 存档引用

- 配方与统计：`configs/kelmarsh_upstream_protocol_v32.json`（training、statistics、rules）。
- P1 数据与审计：`output/penmanshiel_hourly/prep_v35/`（`series_audit.json`、`quality_by_year.csv`）。
- D2-0 站网与测试年登记：`output/kelmarsh_hourly/d2_feasibility_v34_r2/`。
- 下一阶段（暂缓）：P3 冻结 P2 的设计，在 Penmanshiel 2022 上做一次性确认。K-* 届时改用 Kelmarsh 2016–2021 训练。
