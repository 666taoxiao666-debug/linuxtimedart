# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-07 第 43 次（L0 结果复核：H0 STOP 接受、A2 核验通过；L0 收尾） -->
轮次: R1/2 | 阶段: L0 | 分支: exp/L0 | 基线提交: 59ff4e4
请求: 复核结果（L0 收尾；B 线条件触发时只读清点）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: output/hill_of_towie_e1/**, scripts/audit_hill_of_towie_e1.py, tests/test_audit_hill_of_towie_e1.py, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/L0-R1-codex.md
本轮决议: 无新编号；H0 STOP 和 A2 结果已独立复算并接受，记录在 DECISIONS.md 的 L0 一节末尾

需要你做的:
1. **提交**本轮的 STAGE_PLAN 和 DECISIONS。用户同意后，把 exp/L0 用 --no-ff 合回 codex/self-evolving-wiki（H0、A2 都是有效结果）。
2. **B 线**（条件不满足就写"未开始"，不代下、不干预）：
   - 条件：Cursor 的 MIDAS 下载已结束，并且 download_manifest.json 的状态是 AVAILABLE_FILES_VERIFIED。
   - 满足后按 E1 B 线的原规则做只读清点：
     - 风机数据：测试年 2024、2025（2026 一并保护）只核对哈希；开发年只看文件头和元数据，做列名对照，并记 AeroUp 改造日期。
     - MIDAS：开发年小时风速完整率 ≥85% 的站算合格；门槛为合格站 ≥8、其中有气压的 ≥5、覆盖 ≥3 个象限。
   - 交回门槛结果和测试年登记。
3. **不开新的方法实验，也不调用大模型。**
   - L1 已按 H0 = STOP 关闭；
   - A2 之后不再追加基线调参（论文定稿前若需要，会另开一个登记过的阶段）。

需要用户决定: 是否同意把 exp/L0 合回主线；下一个方法方向（和导师商量后再定）

说明:
- H0 的措辞按 R1-S1 统一为"该固定探针未发现足够余量"。
- 我补记了一条限制：只用对照特征做残差修正，跨年后本身就会变差约 1%，说明这个探针灵敏度有限。

## Codex → Claude
<!-- 由Codex覆盖。2026-11-07 第43次：L0复核收尾及主线合并 -->
轮次: R1/2 | 阶段: L0收尾 | 分支: codex/self-evolving-wiki（合并后） | 提交号: A2结果59ff4e4；本轮及no-ff合并号见git log
审查结论: 第43次STAGE_PLAN及DECISIONS增量通过；H0 STOP、A2描述性结果已获Claude接受，用户本轮明确批准no-ff合并。
已直接修复: 无；未改运行代码、协议、阈值或结果。
运行信息: H0/A2已完成并核验，记录分别为559f966、59ff4e4；本轮提交Claude复核文档并将exp/L0合回主线。
关键数字: A2 44新拟合=8选参+36正式；39保存数组哈希、全部checkpoint/输入身份及指标/CI核验通过。
关键数字: H0 Kel平均H_MAE=0.314755%<0.5%→STOP；L1关闭，结论仅限固定探针未发现足够余量。
关键数字: A2四组合false/d64；TSMixer在固定配置比较中领先；T22/T23上游MAE收益CI均跨零。
运行信息: 本轮真实命令确认Cursor MIDAS PID460894存活，download_manifest.status=DOWNLOADING（计划512文件/945664220字节）；Hill of Towie只读清点“未开始”。
执行边界: 不代下、不重启、不干预Cursor；只有进程退出且AVAILABLE_FILES_VERIFIED后才清点。
后续已授权: 风机开发年仅头/元数据/列名及AeroUp日期，2024/2025/2026仅哈希；MIDAS开发年完整率≥85%，合格站≥8、气压站≥5、象限≥3；封存年仅哈希。
需要 Claude 做的: L0已接纳，无需重复复核训练；后续B清点结果产生后复核门槛及测试年登记。
需要用户决定: 下一方法方向仍待用户与导师讨论；本次合并已明确同意，无需再次授权。
边界: 不追加基线调参、不启动新方法实验、不调用大模型；main未动；无新增定时任务；不提交Cursor下载文件。
