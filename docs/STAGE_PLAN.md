# STAGE_PLAN — 只写下一个阶段

作者：Claude（Codex 只提意见，不改本文件）。上限 150 行：要加新内容时，先删旧内容。
内容只从现有代码和结果中提取；拿不准的地方标「待确认」；事先设定的阈值标「事先设定」。
存档：M5（L1 r3 修法、爬坡复现的 M1–M3 判定）见 `git show 2f74cc3:docs/STAGE_PLAN.md`；L1 原始规则见 `git show a1ebfa0:docs/STAGE_PLAN.md` 的 L1 条；M5 的结果复核见 DECISIONS.md 的 M5 一节。

## 阶段名

M6：L1 第 2 轮优化（L1-O2，GPU），最后一轮。
- 用户 2026-11-11 裁决：L1 r3 是 STOP_L1，但只跑了一轮优化，授权第 2 轮。预算翻倍、反馈加信息，门槛不变；不过就关闭 L1。
- M5 已结束：爬坡在 Penmanshiel 两年复现（RAMP_REPLICATED，开发数据上的复现）；不加缺测回退（NO_FALLBACK）。
- 合并 exp/M2 和 Hill of Towie 协议，等 M6 的结果出来再定。

## 目标

检验的问题和 r3 相同：大模型写的上游编码程序，能否在两个风场上同时赢过手工特征 H、专家程序 P 和随机语法程序 R*。
- r3 的差距（逐 seed 均值）：要过门槛，L* 在 Kelmarsh 要 ≤157.62 kW（实际 160.39），在 Penmanshiel 要 ≤181.64 kW（实际 184.36），两边都还差约 2.7–2.8 kW。
- r3 搜索阶段，同样的短训练下 H 的验证误差是 150.99 kW，比所有大模型程序（最好 152.15）和随机程序（最好 152.21）都低。
- 第 2 轮只改两件事：搜索更多（每臂 60 个），以及给大模型更多反馈：要超过的目标、误差分解、上一代失败的原因。检验的是"r3 失败是不是因为搜索不够、反馈太少"。

## 假设与判定标准

**搜索阶段（Kelmarsh，拟合 2016–2020、验证 2021，训练 seed 2024，短训练：最多 8 epoch、patience 3，同 r3）**
- 臂：
  - H、P：同 r3。
  - R：语法 R 用 seed 2024 抽 60 个。前 30 个与 r3 的 R 完全相同（我本地核对过），后 30 个是新的。
  - L：10 代 × 6 个 = 60 个；大模型种子 base_seed_L = 203400（r3 是 202400）；温度、top_p、max_new_tokens 1536、"不写注释"都同 r3。
  - 不再跑 L-blind：领域知识的作用在 r3 已经测过（逐 seed 均值上 L* 比 L-blind 好 5.5–7.9 kW），第 2 轮的问题和它无关。
- **新增反馈**（Claude 已写：`models/ProgramEncoder.py` 的 feedback_breakdown 和 render_feedback，措辞固定）：
  - 每个 L 提示词都附上 render_feedback 生成的一段文字，放在 PROGRAM_RULES_TEXT 之前。
  - 内容：同一次运行里 H 的短训练验证 MAE，作为要超过的目标；展示出来的前 3 个程序各自的分步长 MAE、平稳格子 MAE、变化格子 MAE（|功率变化| ≥10% 额定）、变化格子的误差占比；上一代失败程序的原因（最多 6 条，每条 ≤160 个字符）。
  - 第 1 代没有历史：展示专家程序 P，以及 P 的短训练分解和 H 的目标值。
  - 误差分解用的是验证年 2021 的标签。r3 已经把这一年的验证 MAE 当反馈，所以没有新增数据暴露。锚点是 split_pack 的 anchor（PersistenceHour）。
- 选择规则同 r3：每臂取验证 MAE 最好的 1 个；平局取编号更早的；失败照样占名额，不补生成。

**正式阶段（同 r3：完整配方，3 个 seed；Kelmarsh 评估 2022；Penmanshiel 程序结构冻结、参数重训，评估 2021）**
- 臂：H、P、R*、L*，共 4 臂 × 3 seed × 2 风场 = 24 次拟合。
- **判定（事先设定，不变）**：两个风场上 L* 的逐 seed 平均 MAE 比 H 好 ≥1.5%，并且比 R*、P 都好 ≥0.5% → CONTINUE；只赢 H、没赢 R* 或 P → SEARCH_ONLY_STOP_L1（只说"自动搜索"有用，不写大模型贡献）；其他 → STOP_L1。
- 判定之后 L1 都关闭，不再有第 3 轮。

**r3 审计缺陷的修正（确定缺陷，必须先修）**
- 先冻结，再评估：24 次拟合都只在验证集上做早停，保存最佳权重（checkpoint 写盘）。然后写 formal_selection_freeze.json 和 test_evaluation_started.json，最后才对评估年做推理。代码顺序要能直接看出"冻结在前、评估在后"。
- 保存正式预测：每个风场一个 cohort（truth、anchor、issue、times、turbines、window_starts），每个臂 × seed 一个预测 npz，SHA256 写进 summary。
- 统计口径同 P3（事先写的"同 P2"）：
  - 3 个 seed 的损失在相同窗口上平均，不做集成；
  - MAE 用 utils/kelmarsh_backbone_information.py 的 paired_loss_statistics（按起报时刻成簇，块长 8，5000 次，seed 2024，Newey–West 滞后 7）；
  - RMSE 用同文件的 rmse_difference_bootstrap。
  - 比较：L* 对 H、P、R*。CI 只作描述，判定只看逐 seed 均值。
- summary 的 stage 写 M6，protocol_id 改为 l1_llm_encoder_v4_o2，supersedes 写明 v3 的结果和审计限制。

**技术门槛（事先设定）**
- 预检：L 用种子 900005 真实生成 1 个第 1 代程序（带新反馈）。围栏闭合、并且 n_new_tokens < 1536 才启动；合法与否只记录。
- 搜索中 L 有 ≥6/60 碰到长度上限 → 技术失败：只跑完搜索、存档，不冻结、不进正式阶段，交回给我，不自动重跑。

## 实验配置表

| 线 | 内容 | 规模 |
|---|---|---|
| L1-O2 搜索 | H、P、R60、L60（Kelmarsh，短训练） | GPU，约 2.5 小时（r3 搜索阶段约 51 次短训练用了 79 分钟；O2 约 90 次，待确认） |
| L1-O2 正式 | H、P、R*、L* × 3 seed × 2 风场 = 24 次 | GPU，约 45 分钟 |

- 只运行一个实例。r3 的产物原样保留，O2 写到 output/l1_llm_encoder/o2/。

## 改动文件清单（= HANDOFF「允许改动」）

- Claude（已改）：`models/ProgramEncoder.py`（feedback_breakdown、render_feedback）、`tests/test_program_encoder.py`（新增 2 个测试）。
- Codex：
  - `scripts/llm_encoder_search_l1.py`（预算、10 代、新反馈接入、去掉 L-blind、先冻结后评估、保存预测、P3 口径的统计）；
  - `tests/test_llm_encoder_search_l1.py`（冻结标记先于评估、预测文件能写出、反馈文字进提示词）；
  - `configs/l1_llm_encoder_protocol.json`（v4_o2）；
  - `output/l1_llm_encoder/o2/**`。

## 本阶段不做什么

- 不改判定阈值、训练配方、数据年份、语法 R 的内容。
- 不换大模型（单张 3090 跑不了 14B 的 bf16），不下载新权重，不加新数据。
- 不碰 Hill of Towie 封存年。

## 风险（≤5 条）

1. 我估计第 2 轮通过的概率 ≤5%。r3 里 H、P、R*、L* 的正式 MAE 相差不到 0.3%（Kelmarsh）和 0.9%（Penmanshiel），和 seed 噪声同一量级；之前的固定探针也只测到 0.31% 和 0.53% 的余量。所以瓶颈更可能是信息量，不是搜索。
2. 后几代的合法率可能下降（r3 第 4 代 0/6）。新反馈里加了失败原因，预计有帮助，但不保证。
3. 把"目标值"告诉大模型后，它可能更激进地加参数、加通道，合法率反而下降。照实报告。
4. 60 个里挑最好的一个，会有"赢家诅咒"：搜索的验证 MAE 偏乐观。正式阶段换评估年、3 个 seed，可以纠正这一点。

## 存档引用

- L1：r3 `output/l1_llm_encoder/r3/`（RESULT_SUMMARY.md、l1_summary.json、top10_programs.json）；r1、r2 预检同目录上层。
- 爬坡复现：`output/ramp_diagnostics/m3_recovered/`。
- 冻结站点：Kelmarsh v32 的 36 站；Penmanshiel 的 11 站；Hill of Towie `output/hill_of_towie_e1/l0b_r2/qualified_stations.json`（5909e143…）。
