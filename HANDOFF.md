# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

长期执行分工（用户于 2026-11-24 指定）：本项目原交给 Cursor 的服务器调用统一交给反重力（Antigravity），包括环境核验、已批准的下载、运行、状态查询和结果取回。交接写明已审提交、可执行步骤、回执位置和停止条件。历史路径里的 Cursor 字样按原名称使用。

长期写作分工（用户于 2026-11-25 指定）：只有用户明确要求写论文，Claude 才开始起草或续写论文。实验完成、结果复核和交接本身不授权写作，Codex 交接时遵守同一边界。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-25 第 120 次（登记 aei-submission R1-B1、R1-B2） -->
轮次: R1/2 | 阶段: aei-submission | 分支: codex/self-evolving-wiki | 基线提交: 3a5becc
请求: 只做记录（提交本轮 DECISIONS 和 HANDOFF）
待审范围: git diff 3a5becc -- docs/review/DECISIONS.md HANDOFF.md
允许改动: docs/review/DECISIONS.md, HANDOFF.md, docs/review/aei-submission-R1-codex.md, output/thesis_materials/**（本轮不改，已有未提交文件保持现状）, output/wjb/**（*.npy、*.npz、*.pkl、*.pt、*.pid 不要提交）, output/x5/**（*.npy、*.npz、*.pid 不要提交）, output/x4/**（*.npy 不要提交）, output/x3/**（*.npy、supervisor 文件不要提交）, output/x2/**（*.tar.gz、*.pid 不要提交）, output/external_acquisition/gfs_g0/manifest_b1.json（不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（不要提交）, .cursor-tmp/**（不要提交）
审查文件: docs/review/aei-submission-R1-codex.md
本轮决议: 采纳 2（R1-B1 G2a 训练入口和协议、R1-B2 X5 本地统计脚本，都由你在 3a5becc 修好，我复核通过）/ 反驳 0 / 暂缓 0 / 转实验 0；见 DECISIONS「阶段 aei-submission」

复核依据:
- R1-B1：train_g1.py 有 --g2a-parent 参数；configs/g2a_protocol.json 的 SHA-256 与 g2a results81 launch.json 里记录的一致。
- R1-B2：scripts/analyze_info_value_x5.py 的 SHA-256 与 local_blocks12/results.json 里的 script_sha256 一致。
需要你做的:
1. 提交本轮 DECISIONS 和 HANDOFF 并推送主线。
2. 不写论文、不打 tag：写论文等用户明确要求，aei-submission-v1 等用户确认最终版后再打。
需要用户决定: 1. 何时确认最终版并打 tag aei-submission-v1。2. 仓库许可证、默认分支、Zenodo 存档（见投稿前核对清单）。

## Codex → Claude
<!-- 第 120 次，登记两项复核决议 -->
轮次: R1/2 | 阶段: aei-submission | 分支: codex/self-evolving-wiki | 提交号: 与本轮 DECISIONS 同一提交，父提交 3a5becc。
审查结论: R1-B1、R1-B2 均已采纳、修正并通过 Claude 复核，见 docs/review/DECISIONS.md「阶段 aei-submission」。
已直接修复: 两处导读修正均在 3a5becc。
运行信息: 复核 train_g1.py 的 --g2a-parent 入口，并按 LF 字节复算 G2a 协议和 X5 本地统计脚本的 SHA-256，对照各自的运行回执。
关键数字: 两个哈希均一致（7280563d…、6ceb92da…），待处理阻断 0。
需要 Claude 做的: 本轮已结束，等待用户明确指定新任务。
需要用户决定: 本轮无。最终标签 aei-submission-v1 待用户确认最终版。
