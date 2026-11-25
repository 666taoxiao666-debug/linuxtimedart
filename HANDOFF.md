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
<!-- 第 113 次交接，正式结果复核完成 -->
轮次: R1/2 | 阶段: wind-journal-baselines-result | 分支: exp/X1 | 提交号: 运行 1fe7ab6，成绩清单前版 3cdf97a，本轮结果与本交接同一提交。
审查结论: 通过，0 阻断。见 docs/review/wind-journal-baselines-result-R1-codex.md。
已直接修复: 无。
运行信息: 反重力已完成 wjbcpu、wjbgpu、wjbfinal。Codex 在本地 output/wjb/wjb_1fe7ab6 从归档预测复算。五个完成回执与结果 SHA256 已核对，TSG 96/96 棵树。
关键数字: 本轮 102 个预测数组、24 个键包、30 份旧参照预测符合登记哈希。990 项指标、81 组配对统计与正式报告一致。
关键数字: 同延迟 LightGBM 在核岭回归的 12/12 格更准，混合延迟 LightGBM 在 TSG 的 12/12 格更准。同输入 LightGBM-hist 在 Cabin 的 5/6 格更准，Kelmarsh LU 未检出差异。
关键数字: U(18)−U(6) 按 K/P/H 排列，核岭回归 10.80/8.71/18.72 kW，TSG 12.46/6.81/23.55 kW，分别 3/3 场 CI > 0。HoT TSG 的 U(6)/U(18) 为 −18.13/+5.42 kW。
交付: output/thesis_materials/ 下 wjb_development_comparisons.md、wjb_upstream_gain.svg、wjb_abstract_conclusion.md、wjb_claim_revision.md 和更新的 wind_forecast_score_inventory.md（500 条）。
需要 Claude 做的: 用本地预测和键独立复算主表关键差值、区间与上游增益变化，按预登记边界复核摘要和主张修订稿。
需要用户决定: 是否把经 Claude 复核的独立修订稿写入 paper_draft_v0.md。
