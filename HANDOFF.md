# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

长期执行分工（用户于 2026-11-24 指定）：本项目原交给 Cursor 的服务器调用统一交给反重力（Antigravity），包括环境核验、已批准的下载、运行、状态查询和结果取回。交接写明已审提交、可执行步骤、回执位置和停止条件。历史路径里的 Cursor 字样按原名称使用。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-25 第 115 次（第 114 次改稿核对：通过） -->
轮次: R2/2 | 阶段: wind-journal-baselines-result | 分支: exp/X1 | 基线提交: 125ca78（运行提交 1fe7ab6）
请求: 复核结果（已通过；只剩反重力日志回执后的一句成因）
待审范围: git diff 125ca78 -- docs/review/DECISIONS.md HANDOFF.md
允许改动: docs/review/DECISIONS.md, docs/review/wind-journal-baselines-result-R2-codex.md, output/wjb/**（*.npy、*.npz、*.pkl、*.pt、*.pid 不要提交）, output/thesis_materials/**（paper_draft_v0.md 等其余文件保持现状，用户确认前不改不提交）, output/x5/**（*.npy、*.npz、*.pid 不要提交）, output/x4/**（*.npy 不要提交）, output/x3/**（*.npy、supervisor 文件不要提交）, output/x2/**（*.tar.gz、*.pid 不要提交）, output/external_acquisition/gfs_g0/manifest_b1.json（不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（不要提交）, .cursor-tmp/**（不要提交）
审查文件: docs/review/wind-journal-baselines-result-R2-codex.md
本轮决议: W1–W5 全部落实，0 条需修改；S1–S3 三处小改由 Claude 在写入初稿时处理。见 DECISIONS「阶段 wind-journal-baselines-result」末尾

补充一点: Penmanshiel LUG 的线性名义模型下界已经是 0.20–0.25（NN 0.008–0.028），所以"最早异常在稳健模型"对 P LUG 不完全成立；日志回执时一并核对。
需要你做的:
1. 提交本轮 DECISIONS 和 HANDOFF。
2. 反重力的 author_stdout 回执到后，在 R2 审查文件里补一句成因（含 P LUG 名义模型那点），只记录，不重跑。
需要用户决定: 1. 是否把主张修订稿写入论文初稿 paper_draft_v0.md（同意后由 Claude 写入，并处理 S1–S3）。2. wjb 结果阶段已结束，exp/X1（X1–X5b、wjb）是否合回主线。

## Codex → Claude
<!-- 第 115 次交接，改稿通过 -->
轮次: R2/2 | 阶段: wind-journal-baselines-result | 分支: exp/X1 | 提交号: 改稿 125ca78，本轮核对决议与交接同一提交。
审查结论: W1–W5 全部落实，Claude 核对通过，0 条需修改。S1–S3 已登记为写入初稿时的编辑事项。
已直接修复: 无。
运行信息: 只读核对已有冻结记录；本地尚未收到反重力的 author_stdout 回执。
关键数字: Penmanshiel LUG 的 8 棵线性候选树根节点名义损失为 0.204–0.249，对应 8 棵神经网络候选树为 0.008–0.028。
后续: 收到日志后，在 docs/review/wind-journal-baselines-result-R2-codex.md 补一句日志支持的成因，注明 Penmanshiel LUG 的名义损失在稳健训练前已偏高。
需要 Claude 做的: 用户同意写入论文初稿后，按 S1–S3 编辑。
需要用户决定: 是否将主张稿写入 paper_draft_v0.md，以及是否将 exp/X1 合回主线。
