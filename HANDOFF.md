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
<!-- 由Codex覆盖。2026-11-14：第80次R2通过；A运行中，B月边界解析失败停止 -->
轮次: R2/2 | 阶段: G2a | 分支: exp/G2 | 提交号: e72e462（A实现）；7b0e3d6（B实现）；本轮执行记录见git log
审查结论: 通过，0阻断；只复审R1-B1/S1/S2，见docs/review/G2a-R2-codex.md。跨零不判等效或替代。
已直接修复: 7b0e3d6中将Linux专用上游依赖延迟导入，使Windows可跑纯逻辑测试；不改方案、阈值、臂或年份。
运行信息: A服务器工作树/home/XT/nuist/pythoncode/TimeDARTFirst_G2_e72e462；PID550508；2026-11-14 02:27 UTC独立启动。
运行信息: A目录/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/G2/g2a_e72e462_r1，子目录formal；完整命令见output/g2a/preflight80/launch_receipt.json。
运行信息: A协议SHA256=7280563d6404a45aefc3844ae61706746be74aaa8274fd7b541df0ec19ee199b；启动门禁已核对G1输入、60份checkpoint和12份B/N预测哈希。
关键数字: A仅B0/N0×3seed×2场=12拟合；B0=9通道、N0=38；同标签、同窗口，全部冻结后才评估，B/N不重训或重生成预测。
关键数字: 服务器tests.test_g1_pipeline 21/21通过、无跳过；本地21项通过（5项服务器依赖跳过）。A结果尚未完成，不作增益结论。
运行信息: B工作树/home/XT/nuist/pythoncode/TimeDARTFirst_G2_7b0e3d6；PID552838已退出；2026-11-14 02:36 UTC启动后在首月SCADA校验失败，未重启。
运行信息: B目录/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/G2/hot_dev_7b0e3d6_r1，子目录prepared；完整命令见output/g2/hot_dev/preflight80/launch_receipt.json。
关键数字: B本地/服务器tests.test_prepare_hot_dev均9/9通过；覆盖封存拒读、施工边界、缺失标签、重复时间拒绝、通道、共同窗口队列。
运行信息: B只打开2016–2023及既有元数据，核对原字节哈希；不下载、不安装、不训练；2024–2026不访问。
关键数字: B确定缺陷R2-E1：read_month过严地要求所有行同月；2016年1月三个表各含21条2016-02-01 00:00，重复键0、未知机组0。诊断见failure_diagnostic.json。
运行信息: B尚未生成准备产物、未进入MIDAS/GFS；原始数据未修改，未到封存年。失败源于本轮解析器月边界假设，不是数据损坏。
运行信息: 已保存两线启动回执及服务器测试日志；160MB原始manifest_b1.json仍未跟踪且未修改。G1冻结代码/协议/产物未改，main未动。
需要 Claude 做的: 复核A实现；A完成后核验4组比较CI/p、分时距和爬坡。B按第80次“任何一步失败停下交回”停止，审查文件R2-E1/P1说明月边界修法，待明确新目录重跑批准。
需要 Claude 做的: B修法需覆盖相邻月/年的边界重复、冲突拒绝、先按时间键隔离>=2024再解析观测值；不允许直接删除边界或平均重复功率。
需要用户决定: 暂无；A继续原12拟合，B不自动重跑；本轮尚无A最终结果。
