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
<!-- 由Codex覆盖。2026-11-14：第81次执行；月边界修复已运行，额外最小修复待网络恢复部署 -->
轮次: R2/2执行恢复 | 阶段: G2a | 分支: exp/G2 | 提交号: 1b6eb2b（月边界）；c12d799（空小时保留），均已推送
审查结论: 按第81次批准执行，不新增方案轮次；见docs/review/G2a-R2-codex.md“第81次执行”。
已直接修复: 1b6eb2b先按时间键隔离>=2024再转数值；跨月/年缓存末小时，完全相同重复折叠，冲突拒绝。
已直接修复: 官方Hill_of_Towie_data_dictionary.md明确TimeStamp是UTC区间终点；减10分钟后用既有聚合，官方字典加入输入MD5/SHA核验。
运行信息: B工作树/home/XT/nuist/pythoncode/TimeDARTFirst_G2_1b6eb2b；运行目录/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/G2/hot_dev_1b6eb2b_r2，子目录prepared。
关键数字: 月边界修复本地/服务器11/11测试通过；2026-11-14 02:54 UTC启动PID557324，回执及测试日志在output/g2/hot_dev/preflight81/。
运行信息: r2已退出，failed.json为“Power label altered during enrichment”；原r1/r2均保留，未访问封存年份、未进入MIDAS/GFS。
关键数字: R2-E2证据：2016-01机组2304526原744小时，inner合并后741；left合并恢复744且功率标签逐值相等。服务器r2/failure_diagnostic.json已保存，待取回。
已直接修复: c12d799仅inner→left保留空小时原始NaN标签；合成缺整小时测试先失败后通过，全部12项本地测试通过，额外唯一最小修法额度已用1/2。
运行信息: c12d799已推送，但SCP多次SSH banner超时、tailscale预热超时；未确认上传成功，尚未部署/启动r3，不把网络失败当运行失败。
运行信息: A仍使用/home/XT/nuist/pythoncode/TimeDARTFirst_G2_e72e462及PID550508；run=/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/G2/g2a_e72e462_r1/formal。
关键数字: 最后成功查询A进程存活、日志3/12拟合；无完成/失败标记。此后网络不可达，最新状态待查，不重启、不改其协议和源码。
运行信息: A协议SHA256=7280563d6404a45aefc3844ae61706746be74aaa8274fd7b541df0ec19ee199b，仍是原12拟合及12份B/N复用。
需要 Claude 做的: 事后复核R2-E2最小修复；无需重复批准已授权恢复。A结果及B开发年审计尚未齐，不冻结G2b或开始封存确认。
后续执行: 网络恢复后先查询真实进程/标记；c12d799新工作树跑12项测试后新目录恢复B，保留旧目录；取回r2失败证据。A完成后核验哈希并交回4组CI/p、分时距和爬坡。
需要用户决定: 无新方案裁决；当前外部阻塞是SSH连接，任务未完成。main、G1冻结产物及160MB原始清单未动。
