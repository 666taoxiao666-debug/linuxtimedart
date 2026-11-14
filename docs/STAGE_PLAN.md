# STAGE_PLAN — 只写下一个阶段

作者：Claude（Codex 只提意见，不改本文件）。上限 150 行：要加新内容时，先删旧内容。
内容只从现有代码和结果中提取；拿不准的地方标「待确认」；事先设定的阈值标「事先设定」。
存档：G1c 协议全文（十臂、判定、校准、时效规则）见 `git show 2dcfc1d:docs/STAGE_PLAN.md`；G1 结果复核见 DECISIONS「阶段 G1c」末段；G0 数据规则见 `git show cb090f1:docs/STAGE_PLAN.md`。

## 阶段名

G2a（2026-11-14）：G 线方法收尾后的两件事，并行，都不碰任何封存数据。
- A：2×2 拆解实验（上游观测 × 数值预报），Kelmarsh、Penmanshiel 开发年，12 次拟合，只描述。
- B：Hill of Towie 开发年（2016–2023）数据准备和只读审计；不训练、不打开 2024–2026。
- 用户 2026-11-14 裁决：G 线方法收尾（错位修正不成立，写负结果），做 Hill of Towie 确认；exp/G1 合回主线。

## 目标

1. **G1 已定的结论（不再改）**：
   - 公开 GFS 在上游观测底座上，两场 MAE 都降约 8.6%（Q1 通过），增益集中在 h3–h4 和爬坡。
   - 错位修正方法（M）不优于通用逐站编码、本场错位和只修幅度（Q3/Q4/Q5 不通过）。
   - 机制诊断：上游站点读不出时间错位（秩相关与换日负对照相同），GFS 10 m 误差以幅度为主。
2. **A 要回答的问题**：有了数值预报之后，上游观测还有没有用？B0（只有本场 SCADA）、B（+上游）、N0（+GFS）、N（+上游 +GFS）四格，决定论文主线写"互补"还是"数值预报覆盖了上游观测"，也决定 Hill of Towie 确认的检验族。
3. **B**：把 Hill of Towie 准备成和另外两场同口径的数据，供下一轮冻结确认协议后一次性使用。

## 假设与判定标准

**A. 2×2 拆解（事先设定，只描述，不作方法主张）**
- 数据：直接用 G1 正式运行 g1c_27312d7_r2 的 prepared 产物（路径 T，同窗口、同标签、同 GFS 验收名单），不重新准备。
- 新增两臂，配方、seed（2024–2026）、早停、先冻结后评估都与 G1 相同：
  - B0：去掉 H 的 11 个上游特征和上游缺测标志，只留本场 SCADA 和功率（通道名由 Codex 按准备脚本核对，并写进启动回执）。
  - N0：B0 + 与 N 完全相同的 GFS 通道（历史块 1、当前周期 13 个常数通道及掩码、时效、缺测标志）。
- B、N 不重训，直接用 G1 冻结的 checkpoint 和预测（哈希核对后复用）。
- 新协议文件 configs/g2a_protocol.json，单独锁 SHA；configs/g1_protocol.json 不动。
- 统计口径同 G1（3 个 seed 的逐窗损失先平均；paired_loss_statistics，块长 8，5000 次，seed 2024，NW 滞后 7）。报告 N 对 N0、B 对 B0、N0 对 B0、N 对 B 的增益、CI、p，以及分时距和爬坡指标。
- 解读规则（事先写下，只决定写法和下一轮检验族，不作门槛）：
  - N 对 N0 两场 CI 下界都 > 0 → "上游观测与数值预报互补"，Hill of Towie 的检验族加入 N 对 N0。
  - 两场 CI 都跨 0 → "在 1–4 h 上数值预报基本覆盖了上游观测"，论文如实写。
  - 两场方向不一致 → 两场分别报告，不合并下结论。

**B. Hill of Towie 开发年准备（只读，事先设定）**
- 范围：SCADA、MIDAS、GFS 都只用有效时刻 < 2024-01-01 的数据。2024、2025 两个封存年和 2026 保护年不读、不解码、不统计；GFS 封存原始字节不动。
- 风机：21 台 Siemens SWT-2.3-VS-82（额定 2300 kW）。按 Kelmarsh/Penmanshiel 的同一套逐小时口径生成 8 个 SCADA 输入、last10 和功率（Ndir 取机舱朝向 sin/cos，Pab 取三桨距角均值；聚合和截止时刻与已有两场相同，Codex 写明对应关系）。标签不修改、不插补。
- 上游站点：冻结的 12 站（qualified_stations.json，SHA 5909e143…），站 137 的 2016 年 2 个冲突小时按已定规则记缺测；11 个手工特征按 v31 定义生成。
- GFS：按 G1 规则把 B2、B1 的验收周期插值到 Hill of Towie 中心和 12 个站点（区域 49.5–60°N、−8…2.5°E，先核对都在格网内），时效审计 3–18 h，校准规则同 G1。
- AeroUp 改造（事先设定，各臂相同）：
  - 每台风机在其"第一次到最后一次施工日"之间的窗口，不进拟合和验证（不删数据，只是不用作样本）。
  - 所有臂加一个"该机已完成改造"的 0/1 输入通道（施工日期是事先已知的元数据，因果上可用）；封存测试年全部为 1。
  - 只描述：开发年里每台机改造前后的功率曲线变化。
- 切分（待确认，下一轮冻结）：拟合 2016-01 至 2022-12，验证 2023；Codex 报告各切分的窗口数、改造后样本占比和每年完整率，我据此冻结。
- 审计输出：风机和站点的逐年完整率、通道对应表、GFS 点位和时效审计、校准方法统计；全部写进 output/g2/hot_dev/，原始输入哈希登记。

**下一轮（G2b，预告，本轮不做）**
- 冻结 Hill of Towie 确认协议：在开发年上按冻结脚本重新拟合（I* 之类的选择只在验证年做），封存的 2024、2025 只评一次。
- 主检验族草案：N 对 B（数值预报增益），5 个 seed 的 N 集成对单 seed N；按 A 的结果决定是否加入 N 对 N0。两个测试年取 max-p 后做 Holm，规则同 G1。
- 次要：爬坡指标、分时距、上游缺测子集，只描述。G1 的方法臂（M 等）不进确认，只在论文里报告开发年的负结果。

## 实验配置表

| 线 | 内容 | 规模 |
|---|---|---|
| A 2×2 | B0、N0 × 3 seed × 2 风场 = 12 次；B、N 复用 G1 | GPU 约 1 小时（G1 平均约 5.3 分钟一次） |
| B HoT 准备 | 开发年 2016–2023 数据准备和审计 | CPU，不训练 |

## 改动文件清单（= HANDOFF「允许改动」）

- A（Codex）：`configs/g2a_protocol.json`、`scripts/train_g1.py`（只加 B0、N0 两臂和复用 B、N 的逻辑，不改已冻结臂的行为）、`tests/test_g1_pipeline.py`、`output/g2a/**`。
- B（Codex）：`scripts/prepare_hot_dev.py`（新文件；尽量复用 prepare_g1_batches.py 和已有 SCADA/MIDAS 解析，不复制大段代码）、`tests/test_prepare_hot_dev.py`、`output/g2/hot_dev/**`。

## 本阶段不做什么

- 不再做任何错位修正或站点修正的方法轮（G1 的优化轮触发条件没满足，Q3 点估计是 −0.60%、−0.96%）。
- 不读 Hill of Towie 的 2024–2026 数据，不解码 GFS 封存字节，不在 Hill of Towie 上训练。
- 不下载、不安装；不改 G1 的协议、checkpoint 和结果。

## 风险（≤5 条）

1. B0 的通道集合要和 Kelmarsh/Penmanshiel 历史上的 last10 对照一致，否则 B 对 B0 不能和以前的上游增益对比；Codex 先核对通道名再跑。
2. Hill of Towie 的 SCADA 字段名和 Kelmarsh 不同，对应关系出错会让底座变弱；审计里要列出每个字段的来源列。
3. AeroUp 改造让开发年和测试年的功率曲线不同，所有臂都受影响；改造标志通道和施工期剔除能缓解，不能完全消除。
4. Hill of Towie 在苏格兰北部，GFS 区域边界、站点覆盖（象限 1、2、3，东北是海）可能使上游信号弱于另两场。

## 存档引用

- G1 结果：output/g1/results_78/formal/results.json（SHA 34767c3b…）、completion_verified.json；准备产物 g1c_27312d7_r2。
- GFS 验收：output/gfs_g0/b2/acceptance.json、output/gfs_g0/b1/acceptance.json（SHA 1cf9022d…）。
- Hill of Towie 清点：output/hill_of_towie_e1/L0B_RESULT_SUMMARY.md、l0b_r2/qualified_stations.json；AeroUp 日期在 l0b_r1/readonly_inventory.json（21 台，2021-09-23 至 2023-05-26）。
