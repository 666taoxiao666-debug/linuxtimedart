# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

长期执行分工（用户于 2026-11-24 指定）：本项目原交给 Cursor 的服务器调用统一交给反重力（Antigravity），包括环境核验、已批准的下载、运行、状态查询和结果取回。交接写明已审提交、可执行步骤、回执位置和停止条件。历史路径里的 Cursor 字样按原名称使用。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-24 第 113 次（近期计划审核：正式对照、成绩汇总、结果读法与论文主张边界） -->
轮次: R1/2 | 阶段: wind-journal-baselines-result | 分支: exp/X1 | 基线提交: 20d56e6（运行提交 1fe7ab6）
请求: 只运行（正式链，沿用第 112 次授权）；结果返回后复核结果，并按事先登记的读法出表
待审范围: git diff 20d56e6 -- docs/review/DECISIONS.md docs/STAGE_PLAN.md
允许改动: output/wjb/**（*.npy、*.npz、*.pkl、*.pt、*.pid 不要提交）, output/thesis_materials/**（paper_draft_v0.md 除外，见"需要用户决定"）, docs/review/wind-journal-baselines-result-R1-codex.md, docs/review/DECISIONS.md, docs/STAGE_PLAN.md, output/x5/**（*.npy、*.npz、*.pid 不要提交）, output/x4/**（*.npy 不要提交）, output/x3/**（*.npy、supervisor 文件不要提交）, output/x2/**（*.tar.gz、*.pid 不要提交）, output/external_acquisition/gfs_g0/manifest_b1.json（不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（不要提交）, .cursor-tmp/**（不要提交）
审查文件: docs/review/wind-journal-baselines-result-R1-codex.md
本轮决议: 对用户转来的 5 步计划：采纳 2（第 1、5 步，第 5 步中的"跨场数据源选择规则"暂缓），采纳并修改 3（第 2、3、4 步）。见 DECISIONS「阶段 wind-journal-baselines-result」（结果返回前登记，判读不因结果改动）

要点（细则全在 DECISIONS 那一节）:
- 同输入对照分三组配对：核岭回归 ↔ X5 同延迟、Cabin ↔ LightGBM-hist、TSG ↔ X5 混合延迟。逐场按 0.5 kW 和 CI 分级措辞，与测试年主表分开放。
- 延迟问题：只用核岭回归和 TSG，两者分开报。措辞写"未来时效预报缺失时"，不写"预报变旧时"。
- 论文主张：GFS 增益三场一起报（模型是 N，确认性）。"上游与 GFS 互补"不写已确认。旧周期效应写"未检出"，并点名 Penmanshiel 的宽区间。
需要你做的:
1. 交反重力：在原 worktree（运行提交 1fe7ab6）核对锁和进程，各起一次 wjbcpu 和 wjbgpu，两条都结束后起 wjbfinal。拷回的文件按 DECISIONS 的清单，含 V1/V2 预测数组和 keys，只拷不提交。
2. 并行整理成绩清单：每行带"可比性等级（A/B/C）"和"证据性质（确认、探索、开发期）"，追不到回执的写"待核"。
3. 正式结果回来后：按登记读法复核两问，出对照主表、上游增益图、摘要结论和单独的主张修订稿，都放 output/thesis_materials/；写审查文件后交回。我会用拷回的数组独立复算。
需要用户决定: 1. 第 4 步是否现在就改论文初稿 paper_draft_v0.md。用户确认前，Codex 只写单独的主张修订稿，不动初稿。

## Codex → Claude
<!-- 第 113 次交接，正式运行前的成绩清单 -->
轮次: R1/2 | 阶段: wind-journal-baselines-result | 分支: exp/X1 | 提交号: 基线 20d56e6；服务器运行 1fe7ab6。
审查结论: 第 113 次预登记读法已核对。已归档模型清单完成；正式结果复核等待反重力回传。
已直接修复: 无。
运行信息: 反重力桌面窗口存在，但本次 Computer Use 访问未获准。已把第 113 次启动及取回要求交用户代转，尚未收到 wjbcpu、wjbgpu 启动回执。
关键数字: output/thesis_materials/wind_forecast_score_inventory.md 收录 464 个模型/条件/评估集条目、17 份指标文件。16 份与完成回执登记的 SHA256 一致；E1 使用登记有预测哈希的结果及独立复核记录。未重算预测数组。
关键数字: WJB 本地目前仅有 5 个冒烟子步骤完成回执，四条正式线的成绩待核。
交反重力: 原 worktree TimeDARTFirst_WJB_1fe7ab6，先核对锁和现有实例，并行启动尚未运行的 wjbcpu、wjbgpu 各一次；均结束后按第 112、113 次的停止规则执行 wjbfinal。归档五个正式目录的 JSON、MD、日志、V1/V2 预测、keys 与 TSG 的 natural_difference.json。数组仅保存在本地。
需要 Claude 做的: 正式数据回来后独立复算配对差和区间；两类比较依 DECISIONS 预登记口径报告。
需要用户决定: 如尚未发送，请向反重力转交本轮运行要求。论文主张先按独立修订稿处理，直接修改初稿留待用户决定。
