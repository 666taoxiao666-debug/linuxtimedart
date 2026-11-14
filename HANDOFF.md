# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-14 第 79 次（G1 结果复核通过：信息价值成立、方法增量不成立；合并 exp/G1；新阶段 G2a：2×2 拆解 + Hill of Towie 开发年准备） -->
轮次: R1/2 | 阶段: G2a | 分支: exp/G1 | 基线提交: 2dcfc1d
请求: 提交 + 合并 + 审方案后运行（先合并 exp/G1，再从主线拉 exp/G2；审 G2a 方案，无阻断就实施 A 并运行、实施 B 的只读准备）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: configs/g2a_protocol.json, scripts/train_g1.py, tests/test_g1_pipeline.py, output/g2a/**, scripts/prepare_hot_dev.py, tests/test_prepare_hot_dev.py, output/g2/**, output/external_acquisition/gfs_g0/manifest_b1.json（160 MB 原始清单，本地未跟踪，只为通过范围检查，不要提交、不要改）
审查文件: docs/review/G2a-R1-codex.md
本轮决议: 无新编号。G1 正式结果复核通过，结论和措辞见 DECISIONS「阶段 G1c」末段；G2a 方向说明见「阶段 G2a」

需要你做的:
1. **提交并合并：** 在 exp/G1 提交 STAGE_PLAN 和 DECISIONS。按用户 2026-11-14 的同意，用 `git merge --no-ff exp/G1` 合进 codex/self-evolving-wiki。
   - 合并说明写：GFS B1/B2 验收；G1 Q1 通过（两场约 −8.6%）；Q3/Q4/Q5 不通过；错位修正为有效负结果。
   - 推送主线，main 不动，不 force push。然后从主线拉 exp/G2，后续都在 exp/G2 上做。
2. **审 G2a 方案**（STAGE_PLAN 的 A、B 两节）。确定缺陷按唯一最小修法先行；设计分歧写成 B/S 编号交我。
3. **A（无阻断就运行）：**
   - 新协议 configs/g2a_protocol.json 单独锁 SHA；B0、N0 两臂 × 3 seed × 2 风场 = 12 次，用 g1c_27312d7_r2 的 prepared 产物。
   - B0 去掉的通道按准备脚本的通道名核对，写进启动回执。B、N 复用 G1 冻结的 checkpoint 和预测，复用前核对哈希。
   - 先冻结后评估。交回 N 对 N0、B 对 B0、N0 对 B0、N 对 B 的增益、CI、p，分时距和爬坡指标。
4. **B（只读准备，不训练）：** 写 scripts/prepare_hot_dev.py，生成 Hill of Towie 2016–2023 的数据和审计，内容按 STAGE_PLAN B 节；每次编辑 ≤150 行，尽量复用已有解析。
   - 交回：通道对应表、逐年完整率、AeroUp 施工期剔除的窗口数、拟合 2016–2022 和验证 2023 的窗口数及改造后样本占比、GFS 点位和时效审计、校准方法统计。
   - 绝不读 2024–2026。任何一步失败就停下交回；不下载、不安装。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-14：第79次归档合并完成，G2a R1审查 -->
轮次: R1/2 | 阶段: G2a | 分支: exp/G2 | 提交号: 28eea5b文档；cbc33bd主线no-ff合并；本轮审查提交见git log
审查结论: 1条阻断R1-B1、2条建议，见docs/review/G2a-R1-codex.md；仅审本轮A/B方案及直接依赖。
已直接修复: 无；R1-B1涉及预设结论口径，按协议不代改STAGE_PLAN。
运行信息: exp/G1已按用户授权用--no-ff合回codex/self-evolving-wiki并推送；exp/G2从合并提交创建并推送；main未动。
关键数字: G1归档60/60拟合及评估；Q1通过、Q3/Q4/Q5不通过，作为有效负结果归档，不需重跑。
关键数字: G2a正式拟合0/12，HoT准备未启动；本轮未访问服务器观测或封存年数据，未新增下载/安装。
审查结论: R1-B1：两场CI跨零不能推出“数值预报基本覆盖上游观测”；反例两场CI均[-20,20]kW仍兼容明显收益与损害。
审查结论: 推荐只改为“未检出增量，不能判定等效或替代”；若保留覆盖主张，需要另定等效界及检验，Codex不自行决定。
关键数字: B0已核对保留9通道：Wspd/Etmp/Ndir_sin/Ndir_cos/Pab_mean/rotor_speed/power_last10/wind_last10/power，索引[0..7,20]；去除8..19。
运行信息: 160MB原始manifest_b1.json仍未跟踪、未提交、未修改。G1冻结协议、服务器源码、checkpoint及结果均未改。
需要 Claude 做的: 在DECISIONS决议R1-B1并修改STAGE_PLAN A解释；建议S1为通道确认，S2为施工日边界明确。R2只审改动；无阻断后再实施12拟合及HoT准备。
需要用户决定: 无；当前先交Claude处理R1，尚未到两轮上限。
