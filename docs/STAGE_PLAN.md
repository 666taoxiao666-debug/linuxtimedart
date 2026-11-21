# STAGE_PLAN — 只写下一个阶段

作者：Claude（Codex 只提意见，不改本文件）。上限 150 行：要加新内容时，先删旧内容。
内容只从现有代码和结果中提取；拿不准的地方标「待确认」；事先设定的阈值标「事先设定」。
存档：X4 计划见 `git show 1c0a125:docs/STAGE_PLAN.md`；X3 最终版见 `git show 0795f39:docs/STAGE_PLAN.md`。

## 阶段名

X5（2026-11-21）：信息源价值——上游观测在已有 GFS 时还值多少、值在什么时候、GFS 变旧时值多少。
- 用户 2026-11-21 定：先做 1–3 三块，做完再决定是否下载第二个数值预报（ECMWF IFS）。
- 起因：方法线已全部为负（G1、X2-ANI、X3 微调与集成、X4 A/B1/B2/D0；X4 结果见 DECISIONS「阶段 X5」开头）。论文主线改为信息源价值，只描述，不再追求"新方法降 MAE"。

## 目标

1. 块 1、2（已完成，零训练，本地）：把 2×2（有无上游 × 有无 GFS）写成两方 Shapley 分解；看上游剩余价值落在哪些可观测的时段。
2. 块 3（本阶段要跑）：GFS 发布延迟 d ∈ {6, 9, 12, 15, 18} h 时，上游观测的剩余价值曲线 U(d)，以及"混合延迟训练"的单一模型能否覆盖所有延迟。
3. 结果直接进论文第 4–5 节；IFS 是否下载等结果出来后由用户决定。

## 假设与判定标准

**块 1、2（已完成，结果 `output/x5/local_blocks12/`，脚本 `scripts/analyze_info_value_x5.py`，X2 part A 冻结预测，seed 2024）**
- 定义（在运行前写进脚本文档）：φ_U = ½[(L0−LU)+(LG−LUG)]，φ_G 同理；交互 I = (L0−LU) − (LG−LUG)。
- C1 分歧：D = |ŷ_LG − ŷ_LU|，每个风场、每个时距的 80% 分位只在 V1 上定；V2 为主读数，测试年只作探索。
- C2：GFS 时效 age = 6 + (起报小时 mod 6)。
- 结论（描述性，Claude 本地计算）：
  - φ_G 是 φ_U 的约 1.7–3.5 倍；交互 11–25 kW，即 GFS 吸收了上游价值的大部分（V2：Kelmarsh 无 GFS 时 17.7 kW → 有 GFS 时 2.3 kW）。
  - 上游剩余价值集中在两类预报分歧大的 20% 时段：这些时段在 V2 上占剩余价值的 54–85%（10 个集合合计 42–85%）；"分歧大减其余"的差值在 10 个集合中 9 个 CI > 0（HoT V2 跨 0）。
  - age 6–11 h 范围内没有单调趋势，范围太窄，所以需要块 3。

**块 3（事先设定，协议 configs/x5_protocol.json）**
- 延迟规则：起报 t0 用最新的满足 C + d ≤ t0 的周期，age ∈ [d, d+5]；时效超出 [3, 18] h 一律掩码，不用 f000，不外推。d = 18 时 t0+1…t0+4 全部没有 GFS，只剩 t0 时刻的历史值。
- 只重建树模型实际读到的 x50 末步 GFS 列 20–48，用冻结准备用过的同一批点位缓存和同一套 interpolate_leads / nwp_input 代码。
- 名义绑定（技术门槛）：d = 6 重建的列必须与冻结 x50 的每个窗口一致（atol 1e-5），训练、V1、V2、测试全部检查，不过就停。
- 窗口、标签、划分完全沿用 X2 part A；正式运行时核对 issue/turbine/truth 哈希与 `output/x2/x2_0f24875/a/A/<farm>_cohorts.json` 一致。
- 臂（LightGBM，x1 网格 2×2，V1 早停与选格，seed 2024，复用 run_x2.run_a_farm 不改）：
  - 匹配：每个 d 训练 LG_d、LUG_d（训练和评估都在延迟 d 下）；LU 只训练一次（与 GFS 无关，即"GFS 断供"端点）。
  - 失配：LG_d6、LUG_d6 直接用在其他延迟的输入上（意外延迟，只预测）。
  - 混合：LG_mix、LUG_mix，训练和 V1 的每个窗口随机分到一个延迟（default_rng(2024)），在每个 d 上评估。
- 读数：U(d) = MAE(LG_d) − MAE(LUG_d)；G(d) = MAE(LU) − MAE(LUG_d)；U_rise = U(18) − U(6)（同窗口配对）；U(d) 对 d 的 Spearman；失配与混合的 U 和 MAE；LUG_mix − LUG_d。
- 统计：窗口均匀的绝对误差，按起报成簇，块长 8，2000 次，seed 2024（paired_loss_statistics）。
- 描述性判定 H_latency（事先设定）：V2 上 U_rise > 0 且 CI 下界 > 0 的风场 ≥ 2/3 → SUPPORTED，否则 NOT_SUPPORTED。不作方法主张，测试年只作探索。
- 我的预期（估计，不是门槛）：U(18) 接近无 GFS 时的 16–27 kW，U(6) 为 2–4 kW；混合模型在各延迟上比匹配模型差不到 1%。

## 实验配置表

| 块 | 内容 | 规模 |
|---|---|---|
| 1、2 | 已完成（本地，零训练） | — |
| 3 冒烟 | 每场 4000 训练行、600 评估行，延迟 {6, 12}，1 个网格、40 棵树 | CPU 几分钟 |
| 3 正式 | 3 场 × 13 个拟合臂（5 个延迟 × 2 + LU + 2 个混合）× 4 网格 × 4 时距；另有失配与混合的跨延迟预测 | CPU 约 2–3 小时（估计，X2 part A 的 18 个臂用了 71 分钟），8 线程 |

## 改动文件清单（= HANDOFF「允许改动」）

- Claude 已写（本轮待审）：
  - `scripts/run_x5.py`（smoke / run 两个模式，名义绑定、延迟重建、三类臂、报告）
  - `configs/x5_protocol.json`、`tests/test_x5.py`（10 项，含假 LightGBM 的端到端测试和绑定失败测试）
  - `scripts/launch_x2.sh`（新增 x5smoke、x5run）、`scripts/status_x2.sh`（新增 X5 状态）
  - `scripts/analyze_info_value_x5.py`、`output/x5/local_blocks12/**`（块 1、2）
- Cursor 跑完后拷回：`output/x5/x5_<提交号>/**`（JSON、日志、SUMMARY；*.npy 不提交）。
- 文档：`docs/STAGE_PLAN.md`、`docs/review/DECISIONS.md`、`output/thesis_materials/**`；X4 结果目录 `output/x4/x4_1c0a125/**`（*.npy 不提交）。

## 本阶段不做什么

- 不训练神经网络，不改 X2 的冻结预测、窗口和标签，不读任何 2026 数据。
- 不重新打开新的封存数据：HoT 测试年只用 G2c/X1/X3 已经打开过的 sealed_prep 点位缓存。
- 不下载 IFS，不加新臂，不因结果改延迟集合或判定。
- 块 2 的条件分析不当方法门槛，不做"按条件切换"的上线规则（11-20 本地诊断：自适应权重只多 0.2%）。

## 风险（≤5 条）

1. HoT 开发点位缓存的文件名或内容与预期不同 → 名义绑定会直接停下，按回报修。
2. d ≥ 15 时大量时距被掩码，树模型可能把"掩码"本身当信号；这正是要测的真实情形，照实报告。
3. HoT 训练 x50 约 2 GB，原地替换 GFS 列不复制；若内存不够，Cursor 回报，改为分场运行。
4. 只有 seed 2024，U(d) 的差别若小于种子噪声（X2 中约 0.5 kW）就不解读。
5. 失配臂里 age 超出训练范围，属于外推，只描述。

## 存档引用

- X2 part A 冻结运行：`output/x2/x2_0f24875/a/A/`（dev_freeze、cohorts）。
- X4 正式结果：`output/x4/x4_1c0a125/report/dev_decisions.json`。
- 论文稿：`output/thesis_materials/paper_draft_v0.md`（GPT 2026-11-21 修改版，未提交）。
