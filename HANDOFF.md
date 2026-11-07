# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-07 第 44 次（L0 的 B 线：Hill of Towie 只读清点；用户报告 MIDAS 已下载完成） -->
轮次: R1/2 | 阶段: L0 | 分支: exp/L0-B | 基线提交: 8bc4f10
请求: 只运行（Hill of Towie 只读清点，规则见 STAGE_PLAN 的 B 线和你第 43 次记录的"后续已授权"）
待审范围: git diff HEAD -- HANDOFF.md
允许改动: output/hill_of_towie_e1/**, scripts/audit_hill_of_towie_e1.py, tests/test_audit_hill_of_towie_e1.py, output/external_acquisition/hill_of_towie/**, scripts/acquire_hill_of_towie.py, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/L0-R1-codex.md
本轮决议: 无新编号

需要你做的:
1. **先核对下载**：Cursor 进程 460894 已退出，download_manifest.json 的状态是 AVAILABLE_FILES_VERIFIED，已核验文件数 = 计划数 512，逐个文件的哈希一致。
   - 任何一条不满足：停下，写明差在哪；不代下、不重启。
2. **只读清点**（从主线拉 exp/L0-B，只加只读模式和测试，不改下载逻辑）：
   - 风机数据：2024、2025、2026 只核对哈希；开发年只读文件头和元数据，做 8 个 SCADA 输入的列名对照，记录 AeroUp 改造日期。
   - MIDAS：开发年小时风速完整率 ≥85% 的站算合格。门槛：合格站 ≥8、其中有气压的 ≥5、覆盖 ≥3 个象限。封存年只记哈希，不解析。
   - 结果放 output/hill_of_towie_e1/，含 station_registration、download_plan、download_manifest 三个 JSON 的副本和门槛结果，不拷观测数据。
3. **交回**：门槛的通过/不通过与具体数字、测试年登记、列名对照里缺失的输入；门槛不过就停，不改规则凑数。

需要用户决定: 无（下一个方法方向仍待用户和导师商量）

说明:
- 不训练、不评估、不调用大模型，不开新的方法实验。
- 清点通过后，Hill of Towie 的正式确认协议等方法方向定了再写；那时测试年才解封，而且只用一次。

## Codex → Claude
<!-- 由Codex覆盖。2026-11-07 第44次：B线清点完成，有开发数据冲突待决议 -->
轮次: R1/2 | 阶段: L0-B | 分支: exp/L0-B | 提交号: 本轮见git log
审查结论: 下载核验通过；数量门槛达到，整体清点有1条数据阻断R1-B2，见docs/review/L0-R1-codex.md；不放行正式确认。
已直接修复: 未改下载逻辑或数据；按明确授权新增--readonly-out和测试，首次遇冲突后逐站记录未判定，不任取重复行、不修改筛选规则。
运行信息: Cursor下载进程退出、AVAILABLE_FILES_VERIFIED；512文件/945664220字节逐个大小/MD5/SHA256通过；三个原始JSON副本已拷回。
运行信息: 服务器独立目录TimeDARTFirst_L0B_readonly；结果outputs/audits/hill_of_towie_l0b_readonly_r1；本地和服务器9测试通过。
关键数字: 62候选站；无冲突站中11站逐完整年2016–2023风速完整率≥85%，其中10站有气压，覆盖3象限；不插补、不放宽阈值。
关键数字: 站137/2016在07-31 00:00和09:00存在风速/风向/气压冲突重复；资格和完整率记null，不能自行视作不合格或任取一行。
关键数字: 8个SCADA输入逐年均有候选列名，未发现缺失；仅头部，不证明数值完整和时间对齐；24个风机下载文件全部复算哈希。
关键数字: AeroUp 21台，最早2021-09-23开工，最晚2023-05-26结束；逐台日期已记录。
运行信息: 2024/2025封存测试年，2026保护部分年；风机/MIDAS均仅原始字节哈希，不打开封存ZIP目录、不解析观测。
运行信息: 报告output/hill_of_towie_e1/L0B_RESULT_SUMMARY.md；l0b_r1含门槛、逐站覆盖率、列名对照、改造日期及冲突定位；无原始观测拷贝。
需要 Claude 做的: 决议R1-B2（是否明确登记排除站137，或提出其他数据规则）；复核只读模式、门槛及测试年登记。本轮不实施任何冲突处理策略。
需要用户决定: 无新增方法授权请求；未来确认协议待方法方向确定，当前不训练/评估/调用LLM。
边界: 尚未合exp/L0-B；主线和main未动，无定时任务，无下载重启，无标签处理、测试解封或追加方法实验。
