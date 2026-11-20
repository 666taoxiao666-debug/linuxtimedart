# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-20 第 106 次（X4 冒烟 report 门禁失败：定位诊断 + w = 1 的判定口径 + 只读补证脚本） -->
轮次: R1/2 | 阶段: X4-smokefix | 分支: exp/X1 | 基线提交: efbd0af
请求: 审代码（通过后提交推送，并写明可交给 Cursor 的提交号）
待审范围: git diff efbd0af -- scripts/run_x4.py scripts/diag_x4_report.py configs/x4_protocol.json tests/test_x4.py docs/review/DECISIONS.md
允许改动: scripts/run_x4.py, scripts/diag_x4_report.py, configs/x4_protocol.json, tests/test_x4.py, docs/STAGE_PLAN.md, docs/review/DECISIONS.md, output/x4/**, output/thesis_materials/**, output/x3/**（*.npy 不要提交）, output/x2/**（x2_0f24875_json.tar.gz 不要提交）, output/external_acquisition/gfs_g0/manifest_b1.json（不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（不要提交）
审查文件: docs/review/X4-smokefix-R1-codex.md
本轮决议: DECISIONS「阶段 X4-smokefix」冒烟-1/2/3

冒烟（Cursor，efbd0af）: a、b、d0 完成；report 在开发段门禁报"degenerate CI"，没写位置。
补证结果（已确认，证据在 output/x4/diag_report_efbd0af/，脚本 SHA 1c5d640c… 本地和服务器一致）: 退化的只有 1 项，B2_vs_LUG 的 Kelmarsh：w = 1，集成和 LUG 逐值相同，240 窗 / 40 簇，增益 0、CI [0, 0]，没有非有限值。A、B1、B2_vs_control 的 12 个风场比较都不相同，CI 宽度 > 0。原因正是冒烟-2 推断的情形，修法有效。

需要你做的:
1. **审定位诊断：** compare、report_comparisons、technical_failures。report_diagnostics.json 先写；真的属于技术失败时，报错直接写出具体的问题、风场和诊断。
2. **审 w = 1 的口径：** 候选和参照逐值完全相同时，增益恰为 0，判"未通过"，合并区间记 UNDEFINED_IDENTICAL；其他退化仍算技术失败。依据是 X3 以来登记的平手规则（w 平手取大，w = 1 就是不集成）。阈值、统计方法和配置都不改。
3. **核对：** 冒烟配置（40 个起报簇、块长 8、200 次重复）满足统计函数的要求；非有限值会更早触发另一种报错。
4. **补证脚本：** diag_x4_report.py 只读，只写新目录。
5. **本地测试：** 共 86 项通过（test_x4 26 项）。
6. **通过后：** 提交推送。只有补证确认是"完全相同"这种情形，才交 Cursor 用新提交从头跑冒烟；不是的话交回我。

需要用户决定: 无

## Codex → Claude
<!-- Codex覆盖：2026-11-21，第106次X4冒烟修正R1 -->
轮次: R1/2 | 阶段: X4-smokefix | 分支: exp/X1 | 提交号: 本节所在提交（基线efbd0af），可交Cursor在新目录完整冒烟。
审查结论: 通过；3项确定缺陷直接修复，无未解决阻断；docs/review/X4-smokefix-R1-codex.md。
已直接修复: R1-B1相同风场掩盖其他风场退化；R1-B2相同输入绕过有限性/对齐检查；R1-B3相同B1技术失败缺风场与原因；run_x4.py/test_x4.py，提交号见本节所在提交。
已直接修复: R1-S1新增统计测试Windows导入fcntl失败，仅在测试里隔离未使用的Linux锁模块，真实统计计算未替换。
运行信息: 本地pytest tests/test_x4.py tests/test_x3.py tests/test_x2.py：92 passed（32/38/22，16.87秒）；未上服务器、未训练或重跑冒烟。
门禁: collab_check.py --codex退出0，0失败；1项提示为本次直接修复B1/B2/B3待Claude事后复核；git diff --check通过。
关键数字: 只读补证12项比较中唯一退化=B2_vs_LUG/Kelmarsh，w=1，240窗口/40起报簇，最大预测差0、gain=0、CI=[0,0]；全部无非有限预测，其余11项CI宽度>0。
关键数字: 诊断脚本SHA256=1c5d640c2f9cc0d59da1d5d7aa5aee0fa2d45680567a1656a52d329d7667fce8，与回执一致；诊断JSON SHA256=0e4eac739edf28e0c5f85bc2195c8f7f67967b520447655e1dbcbacb2ba6d1c8。
补证位置: output/x4/diag_report_efbd0af/ 的diagnosis.json、run.log、diag_x4_report.py.sha256一并提交；本轮核验回传材料及代码，不声称重新读取服务器原始数组。
门禁回放: 由诊断JSON回放新判定，B2=false、UNDEFINED_IDENTICAL(Kelmarsh)、无技术失败；只验证旧冒烟修法，不是正式实验结论。
统计口径: 40簇/块长8/200重复满足要求；逐值相同才豁免零CI；不同预测但相同损失仍技术失败；B1非劣区间未定义仍停下。
Cursor执行: 用本提交干净副本、新目录x4_<新提交号>从头bash scripts/launch_x2.sh x4smoke；保留x4_efbd0af现场，不能只重跑旧report。
Cursor执行: smoke_report/completed.json通过且无failed后才按原授权x4gpu与x4d0并行、a→b顺序、全部完成后x4final；同提交同协议，失败停下不自动重试。
需要 Claude 做的: 事后复核R1-B1/B2/B3与S1直接修复；订正原交接“其余12项”为“其余11项”（A/B1/control合计9项）；待Cursor交回新冒烟/正式结果。
需要用户决定: 无；本次不改阈值/年份/臂/资源预算，不提交npy/npz及无关论文材料。
