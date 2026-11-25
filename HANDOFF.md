# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

长期执行分工（用户于 2026-11-24 指定）：本项目原交给 Cursor 的服务器调用统一交给反重力（Antigravity），包括环境核验、已批准的下载、运行、状态查询和结果取回。交接写明已审提交、可执行步骤、回执位置和停止条件。历史路径里的 Cursor 字样按原名称使用。

长期写作分工（用户于 2026-11-25 指定）：只有用户明确要求写论文，Claude 才开始起草或续写论文。实验完成、结果复核和交接本身不授权写作，Codex 交接时遵守同一边界。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-25 第 119 次（AEI 投稿材料：Highlights 文件、数据与代码段、审稿人导读） -->
轮次: R1/2 | 阶段: aei-submission | 分支: codex/self-evolving-wiki（只改论文材料和文档） | 基线提交: 10da29e
请求: 复核结果（核对投稿材料；通过后提交，并在最终提交上打 tag）
待审范围: git diff 10da29e -- output/thesis_materials/paper_draft_v0.md docs/review/DECISIONS.md；新文件 output/thesis_materials/aei_submission/（highlights.docx、README.md、投稿前核对清单.md）
允许改动: output/thesis_materials/paper_draft_v0.md, output/thesis_materials/aei_submission/**, output/thesis_materials/任务线与目标_20261116.md, docs/review/DECISIONS.md, docs/review/aei-submission-R1-codex.md, output/thesis_materials/**（只提交本轮相关文件）, output/wjb/**（*.npy、*.npz、*.pkl、*.pt、*.pid 不要提交）, output/x5/**（*.npy、*.npz、*.pid 不要提交）, output/x4/**（*.npy 不要提交）, output/x3/**（*.npy、supervisor 文件不要提交）, output/x2/**（*.tar.gz、*.pid 不要提交）, output/external_acquisition/gfs_g0/manifest_b1.json（不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（不要提交）, .cursor-tmp/**（不要提交）
审查文件: docs/review/aei-submission-R1-codex.md
本轮决议: 无待决编号；改动和核对结果见 DECISIONS「阶段 wind-journal-baselines-paper」末尾的"AEI 投稿材料整理"

改动位置:
- 初稿第 310–314 行：Data and code availability 重写，含五个数据 DOI、仓库链接（tag aei-submission-v1）、导读链接、作者代码出处。
- aei_submission/highlights.docx：由初稿第 13–19 行直接生成，内容完全一致。
- aei_submission/README.md：给审稿人的英文导读，逐表列出结果文件、脚本和协议（路径已在 HEAD 核过）。
- aei_submission/投稿前核对清单.md：给用户的清单。
已核: 仓库是公开的，默认分支 main（旧），没有 LICENSE；常见密钥格式扫描没有发现真实凭据；AEI 是单盲审稿（据检索）。
需要你做的:
1. 核对导读 README 里每一行的映射是否准确，尤其是 G1、G2a、G2b 用的是哪个脚本。核对 Penmanshiel 实际下载的 Zenodo 版本（可替换总 DOI 5946807），以及 GFS 各年份来自 GDEX 还是 AWS，有出入就直接改这两处。
2. 能访问 AEI 官网的话，复核摘要 250 词、关键词最多 7 个，结果写进审查文件。
3. 通过后在主线提交。等用户确认最终版后，在投稿那次提交上打 tag aei-submission-v1 并推送；打之前不要改动 tag 名，正文和导读的链接都依赖它。
需要用户决定: 1. 仓库是否加 LICENSE（仓库含最初 TimeDART 项目的代码，要保留原版权说明），或另建一个只含本研究文件的仓库。2. 是否在 GitHub 把默认分支改成主线（可选）。3. 是否为最终 tag 做 Zenodo 存档拿 DOI（可选）。

## Codex → Claude
<!-- 第 119 次，核对已有材料与实验回执 -->
轮次: R1/2 | 阶段: aei-submission | 分支: codex/self-evolving-wiki | 提交号: 本节与审查文件同一提交，父提交 10da29e。
审查结论: 现有材料核对通过，见 docs/review/aei-submission-R1-codex.md。最终版由用户确认。
已直接修复: 表 1 补 G2a 协议及启动模式，表 4 补本地统计脚本，按实际回执校正数据版本、来源年份和可用文件说明。
运行信息: 本地核对正式回执、结果哈希、映射和 DOCX 文本。AEI 官网已核实摘要最多 250 词、关键词 1–7 个。
关键数字: G1 完成 60 次拟合，G2a 完成 12 次新增拟合，HoT 完成冻结与封存评估。三场期刊对照各步及正式报告均已完成，30 个开发集配对中 29 个支持 LightGBM。
关键数字: 稿件 245 词、6 个关键词，5 条 Highlights 与 DOCX 逐条一致。X1 旧 B1 留有显存失败回执，后续 X3/X4 有独立完成记录。
需要 Claude 做的: 等待用户指定下一项研究任务。写论文须用户明确提出。
需要用户决定: 仅在用户确认最终版后创建并推送 aei-submission-v1。
