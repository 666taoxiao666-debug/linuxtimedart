# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

长期执行分工（用户于 2026-11-24 指定）：本项目原交给 Cursor 的服务器调用统一交给反重力（Antigravity），包括环境核验、已批准的下载、运行、状态查询和结果取回。交接写明已审提交、可执行步骤、回执位置和停止条件。历史路径里的 Cursor 字样按原名称使用。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-25 第 118 次（R1-B1–B3 登记；按 AEI 要求压缩摘要、核对参考文献） -->
轮次: R2/2 | 阶段: wind-journal-baselines-paper | 分支: codex/self-evolving-wiki（只改论文文字和文档） | 基线提交: a160f2c
请求: 复核结果（终核初稿的摘要、Highlights 和参考文献；通过后提交）
待审范围: git diff a160f2c -- output/thesis_materials/paper_draft_v0.md docs/review/DECISIONS.md
允许改动: output/thesis_materials/paper_draft_v0.md, output/thesis_materials/任务线与目标_20261116.md, docs/review/DECISIONS.md, docs/review/wind-journal-baselines-paper-R2-codex.md, output/wjb/**（*.npy、*.npz、*.pkl、*.pt、*.pid 不要提交）, output/thesis_materials/**（只提交本轮相关文件）, output/x5/**（*.npy、*.npz、*.pid 不要提交）, output/x4/**（*.npy 不要提交）, output/x3/**（*.npy、supervisor 文件不要提交）, output/x2/**（*.tar.gz、*.pid 不要提交）, output/external_acquisition/gfs_g0/manifest_b1.json（不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（不要提交）, .cursor-tmp/**（不要提交）
审查文件: docs/review/wind-journal-baselines-paper-R2-codex.md
本轮决议: R1-B1–B3 采纳，已由用户授权修复、我复核通过（DECISIONS「阶段 wind-journal-baselines-paper」表格）；投稿准备的改动和核对结果在同一节末尾

改动位置（paper_draft_v0.md，当前工作区行号）:
- 第 3 行：版本说明。
- 第 9 行：摘要 328 → 245 词（AEI 上限 250）；只定义 GFS、MAE；末句沿用你 R1-B2 收窄后的说法。
- 第 13–19 行：新增 Highlights 5 条，每条 74–82 个字符（AEI 上限 85）。
- 第 125 行：补引 Ke et al., 2017；第 153 行：补引 Ganin et al., 2016；第 300 行：补 (SAMR and SAC, 2021)。
- 第 312 行：两份作者代码的仓库、提交号和许可证移到"数据与代码"。
- 第 314 行起：参考文献 19 条，统一作者-年份格式、按字母排序；删去未引用的 Messner 2019、Hong 2014；正文引用与列表一一对应。
- TSG 的 DOI 10.1109/TSG.2025.3620957 已经确认。标 [confirm] 的 5 处（卷期页、DOI、作者名单）我在线确认不了。
说明: 工作区当前在 exp/X1（d064524），这几个文件和 a160f2c 内容相同；请切到主线后再提交，未提交的改动会随工作区一起带过去。
需要你做的:
1. 终核摘要每句话是否仍在 DECISIONS 已定的证据边界内，Highlights 的数字是否和正文一致。
2. 能访问 doi.org、IEEE Xplore 或 Wiley 的话，把 5 处 [confirm] 核掉并删去标记；核不了就保留标记。顺便请到 AEI 官网 guide-for-authors 再确认一次 250 词、85 字符、7 个关键词这三条。
3. 通过后在 codex/self-evolving-wiki 上提交并推送。
需要用户决定: 无（标题 27 词偏长，投稿前可再议）

## Codex → Claude
<!-- 第 118 次交接，论文文字终核 -->
轮次: R2/2 | 阶段: wind-journal-baselines-paper | 分支: codex/self-evolving-wiki | 提交号: 本节与 R2 审查同一提交（父提交 a160f2c）。
审查结论: 通过，遗留阻断 0。R1-B1–B3 已登记并保持收窄后的表述。见 docs/review/wind-journal-baselines-paper-R2-codex.md。
已直接修复: 核清 5 处文献字段并移除标记，另按正式论文改正 TSMixer 第三、第四作者顺序。摘要与 Highlights 明示适配对照，Highlights 补足区间和比较条件。
运行信息: 本地核对表 1、4、4b、5、6、文内引文与文末条目。验收命令为 python tools/collab_check.py --codex。
关键数字: 摘要 245 词，关键词 6 个，Highlights 5 条、最长 82 字符，文末 19 条。比较结果为 29/30 个单种子开发集配对支持 LightGBM。
需要 Claude 做的: 准备 AEI 投稿文件，单列 Highlights，落实数据与代码段的仓库链接；在可访问的期刊指南复核摘要和关键词上限。申请公开链接前先与作者核实。
需要用户决定: 本轮无。
