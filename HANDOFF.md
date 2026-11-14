# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-14 第 81 次（B 的月边界修复批准，新目录重跑；A 继续；用户定：HoT 确认用 2024、2025 两年） -->
轮次: R2/2 | 阶段: G2a | 分支: exp/G2 | 基线提交: 9f60925
请求: 修复后运行（执行失败的恢复，不是新的方案审查轮）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/prepare_hot_dev.py, tests/test_prepare_hot_dev.py, output/g2/**, output/g2a/**, output/external_acquisition/gfs_g0/manifest_b1.json（160 MB 原始清单，本地未跟踪，只为通过范围检查，不要提交、不要改）
审查文件: docs/review/G2a-R2-codex.md（续写一节"第 81 次执行"）
本轮决议: 采纳 2（R2-E1、R2-P1）；R2-S1–S3 已在上一轮落实

需要你做的:
1. **修 B（R2-P1）：** read_month 和 prepare_scada 的规则：
   - 先按时间键拒读 ≥2024-01-01 00:00 的行，不转换数值；
   - 再把相邻月文件按（时间、机组）合并，完全相同的重复只留一份，取值冲突就报错停止；不删边界行，不平均冲突值。
   - 时间戳是区间起点还是终点，按官方说明写进审计，小时聚合口径与 Kelmarsh、Penmanshiel 相同。
   - 补合成测试：相邻月边界、跨年边界、2023-12 文件里的 2024-01-01 00:00 行被拒读、冲突重复报错。每次编辑 ≤150 行。
2. **新目录重跑 B**（只读，不训练，不读 2024–2026），失败现场 hot_dev_7b0e3d6_r1 保留。交回内容同第 80 次 B 节；若还有同类确定缺陷，修法唯一且不改数据、年份、规则的可按最小修法先行，最多 2 处，超过就停下交回。
3. **A 照常跑完**，交回 N 对 N0、B 对 B0、N0 对 B0、N 对 B 两场的增益、CI、p，分时距和爬坡；跨 0 按"未检出增量，不能判定等效或替代"写。
4. 不下载、不安装。两条线都交回后，我写 G2b 确认协议（用户已定：2024、2025 两年都用于这次确认）。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-14：A已完成核验；B的r4失败，取回证据后停止 -->
轮次: R2/2执行恢复 | 阶段: G2a | 分支: exp/G2 | 提交号: 1b6eb2b（月边界）；c12d799（空小时保留），均已推送
审查结论: 按第81次批准执行，不新增方案轮次；见docs/review/G2a-R2-codex.md“第81次执行”。
已直接修复: 1b6eb2b先按时间键隔离>=2024再转数值；跨月/年缓存末小时，完全相同重复折叠，冲突拒绝。
已直接修复: 官方Hill_of_Towie_data_dictionary.md明确TimeStamp是UTC区间终点；减10分钟后用既有聚合，官方字典加入输入MD5/SHA核验。
运行信息: B工作树/home/XT/nuist/pythoncode/TimeDARTFirst_G2_1b6eb2b；运行目录/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/G2/hot_dev_1b6eb2b_r2，子目录prepared。
关键数字: 月边界修复本地/服务器11/11测试通过；2026-11-14 02:54 UTC启动PID557324，回执及测试日志在output/g2/hot_dev/preflight81/。
运行信息: r2已退出，failed.json为“Power label altered during enrichment”；原r1/r2均保留，未访问封存年份、未进入MIDAS/GFS。
关键数字: R2-E2证据：2016-01机组2304526原744小时，inner合并后741；left合并恢复744且功率标签逐值相等。服务器r2/failure_diagnostic.json已保存，待取回。
已直接修复: c12d799仅inner→left保留空小时原始NaN标签；合成缺整小时测试先失败后通过，全部12项本地测试通过，额外唯一最小修法额度已用1/2。
运行信息: SSH已恢复；c12d799部署到/home/XT/nuist/pythoncode/TimeDARTFirst_G2_c12d799，服务器12/12测试通过，2026-11-14 03:10 UTC独立启动PID561997。
运行信息: B新目录/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/G2/hot_dev_c12d799_r3/prepared；完整回执及测试日志在output/g2/hot_dev/preflight81/r3/；无同类进程后才启动。
运行信息: A仍使用/home/XT/nuist/pythoncode/TimeDARTFirst_G2_e72e462及PID550508；run=/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/G2/g2a_e72e462_r1/formal。
关键数字: A已COMPLETED，12新增拟合+12复用；checkpoint/源码/冻结/全部预测与结果哈希通过，元数据及4组CI/p、分时距、爬坡在output/g2a/results81/。
关键数字: N对B的MAE收益Kelmarsh13.908kW、Penmanshiel15.803kW；N对N0为0.594[-0.144,1.618],p=.2146及1.747[.030,3.610],p=.0512。两场互补判据未满足，不判等效或替代。
运行信息: B的r3在2017-07-01日期简写解析处失败；bd78e5c只将严格YYYY-MM-DD补00:00:00，13项本地/服务器测试通过，额外最小修复额度2/2已用完。
运行信息: 07:00 UTC在新工作树TimeDARTFirst_G2_bd78e5c启动PID613651，目录/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/G2/hot_dev_bd78e5c_r4/prepared；回执在preflight81/r4，r1–r3保留。
运行信息: A协议SHA256=7280563d6404a45aefc3844ae61706746be74aaa8274fd7b541df0ec19ee199b，仍是原12拟合及12份B/N复用。
运行信息: 2026-11-14 07:25 UTC只读查询确认PID613651已退出；r4有failed.json，无completed.json，未启动替代实例。
失败证据: ValueError: Conflicting SCADA boundary observations；scripts/prepare_hot_dev.py:94的merge_boundary按批准规则拒绝冲突，日志最后完成年份为2020；日志未给出具体冲突时间键，未进一步读取原始数据。
证据取回: output/g2/hot_dev/preflight81/r4_failure/{failed.json,run.log}，本地与服务器SHA256逐文件一致；分别为2067c47767087bb55169791d7e2e26cfee7da93aac315813ac181547e89e4610、5f4451f16212941580a51dfb2d249b45b225c3d565698489a56a5980f28aade5。
需要 Claude 做的: 复核A结果及B失败证据，决定后续冲突调查方案；B未完成，不能据此冻结或启动G2b确认。未删除、平均或择取冲突值，未追加修复/重跑。
后续执行: r2失败标记和诊断已取回output/g2/hot_dev/preflight81/r2_failure/；A/B各自完成后核验哈希、交回4组CI/p及HoT开发年审计，不能把启动当完成。
需要用户决定: 本轮无；已按用户“失败取回后停下交回”停止，后续处理需另行批准。main、G1冻结产物及160MB原始清单未动。
