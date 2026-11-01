# STAGE_PLAN — 只写下一个阶段

作者：Claude（Codex 只提意见，不改本文件）。上限 150 行：要加新内容时，先删旧内容。
内容只从现有代码和结果中提取；拿不准的地方标「待确认」。
长方案 `docs/UPSTREAM_INNOVATION_PROPOSAL.md` 已改为存档，只按章节号引用，不再整份送审。

## 阶段名

v33 K1b：底座与站点分支联合训练（复测 K1）。

## 目标

1. 联合训练下，检验到达时间物理先验 A 是否同时胜过通用注意力 B2 和纯底座。底座为 T9 或 T21，在 2022 开发评估年上比较。
2. 修正 K1 暴露的设计缺陷。冻结底座后，残差分支会拟合样本内残差；包括不含任何新信息的 B1，也是从第 1 个 epoch 起验证集变差。
3. 顺带描述：站点编码器能否替代 11 个手工上游特征（只比较 T9+A、T9+B2 与 T21，不作判定）。

## 假设与判定标准

- 假设：底座和分支一起训练后，对至少一个底座 b，b+A 同时优于 b+B2 和 b。
- 指标：2022 年 ALL 队列，h1–4 合并的 MAE 和 RMSE（kW）。单 seed（2024），只看点估计，不做显著性检验。
- 记号：ΔRMSE% = 100·(RMSE_参照 − RMSE_臂)/RMSE_参照，ΔMAE = MAE_参照 − MAE_臂。两者都是正值表示更好。
- 判定（写在 `scripts/train_kelmarsh_arrival_v33_k1b.py` 文件头，互斥）：

| 分支 | 条件 | 继续 / 停止 |
|---|---|---|
| E0 无效 | 缺臂，或某个指标不是有限值 | 技术故障：修好后重跑，不下结论 |
| E1 先验成立 | 至少一个 b 同时满足两条：b+A 对 b+B2，ΔRMSE% ≥ 1 且 ΔMAE ≥ −1 kW；b+A 对 b，ΔRMSE% ≥ 1 且 ΔMAE ≥ 0 | **继续**：进入多 seed 确认（下一阶段，具体设计待确认） |
| E2 只有融合收益 | 不满足 E1，但某个 b+B2 或 b+A 对 b 满足 ΔRMSE% ≥ 1 且 ΔMAE ≥ 0 | **停止**物理先验这一主张；站点分支的收益另行记录，后续待确认 |
| E3 无收益 | 其余情形 | **停止**站点分支和物理先验这条线（C1）；下一步交用户决定 |

## 实验配置表

公共设置（来自 `configs/kelmarsh_arrival_v33_k1.json` 和 K1b 脚本，K1b 不改这份配置）：

- 数据与折：D1 折，2016–2020 拟合，2021 验证，2022 开发评估。seed 2024，滞后 1 h，站点文件只用 2016–2022。
- 底座：TSMixerBasicSmall。同一通道组的各臂共用相同的底座初始化（构建前先设 seed）。
- 损失：标准化功率上的 0.5·MAE + 0.5·MSE。
- 优化器：AdamW，lr 1e-3，wd 1e-4，batch 64；最多 20 个 epoch，patience 5；cosine 调度（T_max 20），梯度裁剪 1。
- checkpoint 选择：按验证集 J_p = MAE/MAE_持续 + RMSE/RMSE_持续 选，epoch 0 也参与，同分时取更早的 epoch。持续法 = 上一小时功率。
- 分支：沿用 K1 模块，不改。宽度 16，query 16，lead 16，score clamp 8。κ 的边界为 [0.5, 4]，初值 1；u_ref 用 50 km 半径。

| 臂 | 底座输入 | 站点分支 | 先验 | 训练方式 |
|---|---|---|---|---|
| T9 | 9 个通道（v28 的 `last10` 集合，索引 0–7 和 20） | 无 | — | 只训底座 |
| T21 | 21 个通道（9 个 + 11 个手工上游特征 + 缺测标志） | 无 | — | 只训底座 |
| T9+B2 | 同 T9 | K1 站点分支 | 通用注意力（q̃ = 1） | 联合训练 |
| T9+A | 同 T9 | K1 站点分支 | 到达时间先验，学习 κ | 联合训练 |
| T21+B2 | 同 T21 | K1 站点分支 | 通用注意力（q̃ = 1） | 联合训练 |
| T21+A | 同 T21 | K1 站点分支 | 到达时间先验，学习 κ | 联合训练 |

- 预算：6 次拟合，只跑一个真实实例。不追加臂，不改阈值。
- 用时：待确认。参考 K1：A 臂在冻结底座下约 21 秒/epoch，9 次拟合共 19 分钟；联合训练会更慢。
- 命令：
  - `python scripts/train_kelmarsh_arrival_v33_k1b.py --smoke --out /tmp/v33_k1b_smoke`
  - `python scripts/train_kelmarsh_arrival_v33_k1b.py --out outputs/logs/Kelmarsh/v33_k1b_<tag>`
- 产出：`k1b_decision.json`、`k1b_summary.md`、`data_manifest.json`，以及各臂的 `trained.json` 和 `history.json`。
  - 拷回 `output/kelmarsh_hourly/arrival_v33_k1b/`，不拷 `best.pt` 和预测数组。

## 改动文件清单（= HANDOFF「允许改动」）

- `scripts/train_kelmarsh_arrival_v33_k1b.py`：新文件，Claude 写。
- `tests/test_kelmarsh_arrival_v33.py`：新增 `K1bTests`，共 3 项，需要 torch。
- `output/kelmarsh_hourly/arrival_v33_k1b/**`：Codex 拷回的结果。
- 本轮另含协作协议 v2 的文件：`AGENTS.md`、`tools/collab_check.py`、`docs/review/DECISIONS.md`。这些只在本轮出现。
- 只读不改（K1 的审计文件 `audit/launch.json` 登记了它们的哈希，检查脚本会核对）：
  - `scripts/train_kelmarsh_arrival_v33.py`
  - `models/UpstreamArrivalResidual.py`
  - `utils/kelmarsh_station_history_v33.py`
  - `configs/kelmarsh_arrival_v33_k1.json`

## 本阶段不做什么

- 不改 K1 的代码、配置、阈值和结果；K1 照原样记为 D3（冻结残差设计）。
- 不追加臂，不跑多 seed，不调阈值或超参数。
- 不读 2023/2024 年的站点数据，不碰 Penmanshiel，不下载新数据。
- 不做 D2（Penmanshiel 迁移）、K2–K5、MM82 功率曲线等后续阶段。§14.18 的相关部分暂缓。
- 不作显著性或 SOTA 主张；K1b 只算开发证据。

## 风险（≤5 条）

1. 联合训练时底座本身也在变，b+A 相对 b 的差异混入了"多出的容量和正则化"。缓解：B2 是同等容量的对照，E1 要求 A 同时胜过 B2。
2. 单 seed 下，1% 的差距可能落在 seed 噪声以内（v26 的经验：残差改进与 seed 噪声相当）。E1 只作开发证据，确认需要多 seed。
3. T21 的输入已经含手工上游特征，T21+A 的提升空间可能很小。只在 T9 上成立也算 E1。
4. 2022 已在 K1 中作为开发评估年看过一次，K1b 再看同一年，不能当独立测试。
5. 联合训练的用时和显存都还没测过（待确认），先跑 smoke。

## 存档引用

- 方案：`docs/UPSTREAM_INNOVATION_PROPOSAL.md`。§14.16 是 D1+D2 正式方案；§14.17 修订了 D1 的先验公式、mask 和 κ 范围；§14.18 写了 raw-valid q、统计方法族和预算。
- K1 的结果与审查：`output/kelmarsh_hourly/arrival_v33_k1/`（`k1_decision.json`、`review.md`、`audit/launch.json`），提交 db9a833。
- K1b 的预登记：`scripts/train_kelmarsh_arrival_v33_k1b.py` 文件头的 docstring。
