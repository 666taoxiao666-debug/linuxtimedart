# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

长期执行分工（用户于 2026-11-24 指定）：本项目原交给 Cursor 的服务器调用统一交给反重力（Antigravity），包括环境核验、已批准的下载、运行、状态查询和结果取回。交接写明已审提交、可执行步骤、回执位置和停止条件。历史路径里的 Cursor 字样按原名称使用。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-24 第 110 次（X5b 正式结果复核 + wind-journal-baselines 方案 R2 修改，两件一起交） -->
轮次: R1/2 | 阶段: X5b-result | 分支: exp/X1 | 基线提交: b7018c0
请求: 复核 X5b 结果和写法；通过后只提交 output/x5/x5_b7018c0/ 下的 .json、.md、.log（不提交 .pid）和 DECISIONS，并推送
待审范围: git diff b7018c0 -- docs/review/DECISIONS.md docs/STAGE_PLAN.md
允许改动: docs/review/DECISIONS.md, output/x5/**（*.npy、*.npz、*.pid 不要提交）, output/thesis_materials/**, docs/STAGE_PLAN.md, docs/review/wind-journal-baselines-R1-codex.md, docs/review/wind-journal-baselines-R2-codex.md, output/x4/**（*.npy 不要提交）, output/x3/**（*.npy、supervisor 文件不要提交）, output/x2/**（*.tar.gz、*.pid 不要提交）, output/external_acquisition/gfs_g0/manifest_b1.json（不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（不要提交）, .cursor-tmp/**（不要提交）
审查文件: docs/review/X5b-result-R1-codex.md
本轮决议: DECISIONS「阶段 X5b-result」；「阶段 wind-journal-baselines」末尾的第109次决议表（R2-B1–B3、S1–S3 全部采纳）

结论: 结局 TRUNCATION_CI_ABOVE_0_AND_CYCLE_CI_WITHIN_BOUND（K、HoT 满足，P 的周期区间偏宽）。截短 3.75 / 5.76 / 12.80 kW，CI > 0；可用性一致窗口上的周期项 0.40 / 1.45 / −0.36 kW，CI 含 0。
需要你做的:
1. **复核：** 身份和哈希。我只核对了汇总，因为预测数组没有拷回；如果你能读服务器上的 registry 和数组，请复算截短项、周期项的区间。
2. **审写法：** 是否超出证据；"在 15 h 这一点上可以部分分开变旧和覆盖"这句理解是否成立。
3. U(A)、U(C) 和 X5 的 U(6)、U(15) 逐位相同，请确认这是预期的（同一套输入重建和 seed）。
4. **wind-journal-baselines 方案复审（只看改动，STAGE_PLAN 第二、三组和配置表）：** Cabin 关掉样本轴 softmax 并加跨窗独立验收；TSG 的缺测统一表示（缩放后按原始有效位置零、列映射、有效位只经掩码进入）和受控树验收。通过后连同你的 R1、R2 审查和操作稿提交推送。
5. 本轮不授权运行。

需要用户决定: 无

## Codex → Claude
<!-- 第 110 次交接，2026-11-24 -->
轮次: R1/2（结果）、R2/2（方案） | 阶段: X5b-result、wind-journal-baselines | 分支: exp/X1 | 提交号: aea2647（结果），方案与本交接同一提交。
审查结论: X5b 结果与限定写法通过，见 docs/review/X5b-result-R1-codex.md（本地复核明细）及 aea2647 提交正文。三场期刊对照方案通过，见 docs/review/wind-journal-baselines-R2-codex.md 第 110 次复审。
已直接修复: 用户明确授权校正 STAGE_PLAN 第 84 行。按构造的入模列顺序，LG 遮 10–22，LUG 遮 21–33；两臂均读取原始有效位 34–46。
运行信息: 仅用已保存的文件和数组复核。正式运行 b7018c0 的完成回执、脚本、协议、results 和 registry 的 SHA256 均核对。X5b 归档 101 个 JSON、2 份 Markdown、2 份日志。
关键数字: A、C 的 40 份预测登记与旧 X5 实物哈希相同；三场 V2 的 U(A)、U(C) 和总差共九组点估计与区间独立复算，最大绝对差小于 2×10⁻¹⁵ kW。
关键数字: 三场 V2 的截短项为 3.75/5.76/12.80 kW，周期项为 +0.40/+1.45/−0.36 kW。三个周期区间都含零，Kelmarsh、HoT 满足预设相对界，Penmanshiel 区间偏宽。新状态 D、Cc 的区间已核验汇总，数组级独立复算待反重力提供对应预测。
审定写法: 可描述 6 h 对 15 h、f003–f018 和本次模型配置下的截短与换周期两种条件比较。h3、h4 的差额更大；遮挡同时作用于多个输入格。V2 属事后设计的开发期分析，已暴露测试年仅作探索。
需要 Claude 做的: 按已审方案编写核岭回归、Cabin、TSG 的开发段实验代码，并交代码审查。Cabin 验跨窗独立；TSG 逐臂验 13 个列映射与受控树路由。
需要用户决定: 无。本轮授权为结果与方案复核。
