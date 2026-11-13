# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-13 第 78 次（R2-E1：导入最小修复已批准；预检后从头完整准备，门禁都过就跑 60 次拟合） -->
轮次: R2/2 | 阶段: G1c | 分支: exp/G1 | 基线提交: cf9c9db
请求: 修复后运行（执行失败的恢复，不是新的方案审查轮）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/prepare_g1_batches.py, tests/test_g1_pipeline.py, scripts/train_g1.py, output/g1/**, output/gfs_g0/**, output/external_acquisition/gfs_g0/**
审查文件: docs/review/G1c-R2-codex.md（续写一节"第 78 次执行"）
本轮决议: 采纳 1（R2-E1，最小导入修复）；R1-B1/B2 的处理复核通过。恢复范围见 DECISIONS G1c 末段

需要你做的:
1. **修复：** scripts/prepare_g1_batches.py:314 改为 `from scripts.screen_kelmarsh_upstream_observations import read_badc_csv, column`，加一个导入回归测试，提交。configs/g1_protocol.json 不动，PROTOCOL_SHA256 不变；新的准备脚本 SHA 写进启动回执。
2. **真实适配预检**（只读、不训练、不读 GFS）：跑 Penmanshiel 的 load_development 元数据部分，核对注册表哈希、11 个站坐标唯一、MIDAS 文件登记齐全、distance_km/bearing_deg 都在。
   - 再遇到同类导入或适配的确定缺陷：修法唯一、且不改数据/年份/阈值/臂/模型的，可以按最小修法先行直接修，最多 2 处，写进审查文件；超过 2 处或修法不唯一就停下交回。
3. **从头完整准备：** 新建运行目录，两个风场都重新准备，不复用失败那次的 Kelmarsh；失败现场 g1c_fd3ee91_r1 原样保留。
   - 新旧 Kelmarsh 产物逐数组（np.array_equal）和 JSON 比较，不比 npz 文件哈希（zip 含写入时间）；不一致就停下交回。
4. **门禁都过就直接跑 60 次拟合**（授权同第 76、77 次，正式拟合目前 0/60）：先冻结后评估，结果拷回 output/g1/。交回 Q1/Q3/Q4/Q5 两场的增益、CI、p、max-p、Holm 结论，I* 选了谁，两场的校准方法统计，机制诊断。任何一步失败就停下交回。不下载、不安装；manifest_b1.json 原文件不提交。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-13：第77次G1c R2，B1通过隔离验收，真实准备技术失败后停止 -->
轮次: R2/2 | 阶段: G1c | 分支: exp/G1 | 提交号: fd3ee91已推送；失败证据提交见git log
审查结论: R1-B1/B2处理通过；新增执行失败R2-E1，见docs/review/G1c-R2-codex.md；不进入第三轮方案审查。
已直接修复: 无超范围改动；按批准范围实现derive_b1_acceptance、实际读取时效门禁、准备产物哈希及评估报告补齐。
运行信息: B1验收output/gfs_g0/b1/acceptance.json；绑定原核验报告6f2dbba3…，原FAIL报告保留。
关键数字: complete_cycles=4371；excluded=9（7缺失、2字段不完整），统一cycle_ok=0、不回退、不删除标签；2021/2022/2023完整率99.52%/99.86%/100%。
关键数字: 成功唯一载荷222526985575字节；下载总量未核实，user_accepted_gap=2026-11-13。日志假设场景估算230137867780字节，不是可证明的上界；算法和源日志哈希见traffic_estimate.json。
关键数字: 服务器验收核验16/16（1.066秒）、最终流水线16/16（1.378秒），无跳过；合成评估报告可序列化，真实准备另记。
运行信息: 路径T；协议SHA256=92e93dd4d1b62b1654c02bbb95299019f507903f13ef6fe71d2f6687003e8ed4，train_g1.PROTOCOL_SHA256固定；服务器提交与哈希一致。
运行信息: worktree=/home/XT/nuist/pythoncode/TimeDARTFirst_G1_fd3ee91；run=/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/G1/g1c_fd3ee91_r1；准备PID385020已退出。
运行信息: scripts/prepare_g1_batches.py:314从错误模块导入read_badc_csv/column，Penmanshiel阶段ImportError；回溯和启动回执见output/g1/preparation_77/。
关键数字: Kelmarsh准备完成，train/val/test=75024/15680/16949；58465小时读取时效3–18h，37项校准均rma。Penmanshiel未完成。
运行信息: prepared/complete.json和formal_launch.json均不存在；真实拟合0/60，无模型冻结或最终评估；按第77次失败即停，不自动重跑。
运行信息: 不下载/安装/解码封存；原160MB manifest_b1.json未提交；原始失败现场保留。
需要 Claude 做的: 复核验收及执行失败；最小修复为改从scripts.screen_kelmarsh_upstream_observations导入两个函数，补实际适配预检，明确后续准备恢复范围。
需要用户决定: 本轮已按失败即停结束；下一步修复并重新准备需明确继续执行授权，60拟合预算尚未消耗。
