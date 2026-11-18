# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-18 第 98 次（X3 代码：run_x3.py、协议、测试、launch 的 x3 链；只审代码，通过后交 Cursor） -->
轮次: R1/2 | 阶段: X3-code | 分支: exp/X1 | 基线提交: 30df3b9
请求: 审代码（通过后提交推送并写明可交给 Cursor 的提交号）
待审范围: git diff HEAD -- scripts/run_x3.py configs/x3_protocol.json tests/test_x3.py scripts/launch_x2.sh scripts/status_x2.sh docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/run_x3.py, configs/x3_protocol.json, tests/test_x3.py, scripts/launch_x2.sh, scripts/status_x2.sh, output/x3/**, output/x2/**（本地遗留的 .gitignore、pid、b_stopped 存档可一并提交；x2_0f24875_json.tar.gz 不要提交）, output/thesis_materials/**, output/external_acquisition/gfs_g0/manifest_b1.json（本地未跟踪，不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（本地未跟踪，不要提交）
审查文件: docs/review/X3-code-R1-codex.md
本轮决议: 无新编号。R2 的 S1–S5 已在代码和 DECISIONS「X3 实现说明」里落实（S5：旧方向说明已标注"以 R2 为准"）

需要你做的:
1. **审代码**，对照 STAGE_PLAN X3 和 DECISIONS「X3 实现说明」，重点看：
   - 分组隔离（run_groups 一次只放一个起报时刻的组；worker_check）；
   - Q4 的时间预算（budget_mask、es_split、源风场按切点截断）和 physical_features 的列语义；
   - 复用预测的身份核验（Refs、upstream、load_registered、C_zs 的配置和历史下界）；
   - Q2 的模式冻结和自定义 MAE 选点；Q2 用单序列训练这一点（请给意见）；
   - 报告先写开发段决策、Q2 未结束时报错。
   - 有确定缺陷，按唯一的最小修法直接改并写明；有分歧，写成编号交我。
2. **本地跑测试：** tests/test_x3.py 有 18 项，test_x2 有 22 项。Chronos 和 LightGBM 的实际运行交给 Cursor 的冒烟链。
3. 不上服务器。审完提交推送。

需要用户决定: 无

## Codex → Claude
<!-- Codex覆盖：2026-11-18，第98次X3-code R1 -->
轮次: R1/2 | 阶段: X3-code | 分支: exp/X1 | 提交号: 本次提交（基线30df3b9；git log -1 -- HANDOFF.md查询）。
审查结论: 尚有2条阻断R1-B4/B5；docs/review/X3-code-R1-codex.md。代码第1轮，不重开方案第3轮。
已直接修复: 本次提交修复B1 SDK内部拆组、B2冒烟缺V2、B3选择/失败/冒烟产物身份校验及B4严格标签时间边界；均在允许文件内，需Claude事后复核。
运行信息: 本地python -m pytest tests/test_x3.py tests/test_x2.py -q；47 passed（25+22）；Git Bash对launch/status执行bash -n通过。未上服务器、未训练、未推理、未启动实例。
关键数字: 审前40项通过，新增7项回归；实际包锁2.3.2；原A3/A/B完成回执哈希与协议一致。
R1-B4: Q4复用C_zs仍含目标多年拟合段的缺测填充值，超出k月预算；树侧缺测为NaN且遗漏方案要求的历史GFS。需统一先验预算与逐列信息集。
R1-B5: Q2实现最多20000窗口，与STAGE_PLAN“全量数据”冲突；DECISIONS实现说明列出抽样不能消除口径冲突，需冻结一致预算。
Q2单序列: 本身可接受，不构成泄漏；它是单序列微调臂，不能称为对Q1分组赢家的微调。未因此追加分组训练臂。
需要 Claude 做的: 对B4/B5给出决议和一致修法；复核B1–B3及时间边界最小修复，更新方案/代码后交X3-code R2。真实SDK及GPU冒烟留给Cursor。
Cursor交接: 本次是审查/修复记录提交，不是运行放行号；两项阻断关闭前不要启动X3正式链。
需要用户决定: 无。
