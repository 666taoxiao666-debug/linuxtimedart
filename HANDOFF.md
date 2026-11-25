# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

长期执行分工（用户于 2026-11-24 指定）：本项目原交给 Cursor 的服务器调用统一交给反重力（Antigravity），包括环境核验、已批准的下载、运行、状态查询和结果取回。交接写明已审提交、可执行步骤、回执位置和停止条件。历史路径里的 Cursor 字样按原名称使用。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-25 第 117 次（用户同意：主张写入初稿、exp/X1 合回主线） -->
轮次: R1/2 | 阶段: wind-journal-baselines-paper | 分支: exp/X1 | 基线提交: 125ca78（运行提交 1fe7ab6）
请求: 复核结果（只看初稿改动是否超出 DECISIONS 已定的边界），通过后提交并合并
待审范围: git diff 125ca78 -- output/thesis_materials/paper_draft_v0.md docs/review/DECISIONS.md
允许改动: output/thesis_materials/paper_draft_v0.md, output/thesis_materials/任务线与目标_20261116.md, docs/review/DECISIONS.md, docs/review/wind-journal-baselines-result-R2-codex.md, docs/review/wind-journal-baselines-paper-R1-codex.md, output/wjb/**（*.npy、*.npz、*.pkl、*.pt、*.pid 不要提交）, output/thesis_materials/**（只提交本轮相关文件）, output/x5/**（*.npy、*.npz、*.pid 不要提交）, output/x4/**（*.npy 不要提交）, output/x3/**（*.npy、supervisor 文件不要提交）, output/x2/**（*.tar.gz、*.pid 不要提交）, output/external_acquisition/gfs_g0/manifest_b1.json（不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（不要提交）, .cursor-tmp/**（不要提交）
审查文件: docs/review/wind-journal-baselines-paper-R1-codex.md
本轮决议: 用户 11-25 同意我的两条推荐：主张写入初稿；exp/X1 用 --no-ff 合回主线。初稿改动清单见 DECISIONS「阶段 wind-journal-baselines-result」末尾

初稿改动要点:
- 摘要、5.1、5.2、5.6（新增表 4b）、新增 5.7（表 5、表 6）、局限、结论、参考文献。
- 表 4b、表 5、表 6 我已用脚本对照 results.json 逐格核对，29/1 的计数也复核过。
- 摘要 328 词，投稿前要压缩。
需要你做的:
1. 只读审初稿改动：每条主张是否在 DECISIONS 的边界内（证据性质、"未检出"与"无影响"的区别、不写"复现"或"覆盖主导"）。有越界就写成阻断；没有就提交。
2. 在 R2 审查文件补上第 116 次的 LR 日志成因（用"支持"），与本轮一起提交。
3. 提交后执行 `git checkout codex/self-evolving-wiki && git merge --no-ff exp/X1`。合并前确认没有 npy、npz、pkl、pt 或大文件进入历史；有冲突就停下交回，不要强推。合并后在 exp/X1 和主线各推送一次。
需要用户决定: 无

## Codex → Claude
<!-- 第 117 次交接，初稿边界核对 -->
轮次: R1/2 | 阶段: wind-journal-baselines-paper | 分支: exp/X1 | 提交号: 本次定稿审查与交接同一提交，正式运行 1fe7ab6。
审查结论: 初稿三处证据边界问题已按用户本轮授权直接修文，复核通过，遗留阻断 0；见 docs/review/wind-journal-baselines-paper-R1-codex.md。
已直接修复: 区分 HoT 的分组均值与逐机组 MAE；限定覆盖不足时的站网结论；注明线性候选失败限于 K/P，三场均由 V1 选择 NN。
运行信息: 本地用正式 JSON 核对表 4b、5、6 共 18 行，从保存的 V2 预测复算 HoT 的 21 台机组。R2 审查记录已补第 116 次日志支持的成因。
关键数字: 30 个开发期配对中 29 个 LightGBM 更准、1 个未检出。HoT 按起报年龄分组的平均预测差为 34.2/50.1 kW；21/21 指加站后 MAE 增大。
需要 Claude 做的: 本轮审稿结束。投稿准备时按目标期刊压缩摘要并完成引用校对。
需要用户决定: 无。
