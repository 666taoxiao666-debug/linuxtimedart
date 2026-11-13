# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-13 第 77 次（G1c R2：B1 两个坏周期隔离；流量计量缺口用户接受；路径 T，条件满足就正式运行） -->
轮次: R2/2 | 阶段: G1c | 分支: exp/G1 | 基线提交: c1771e7
请求: 审改动后运行（只审 R1-B1/B2 的处理；写 B1 验收名单；脚本收尾；门禁都过就直接跑 60 次拟合）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/prepare_g1_batches.py, scripts/train_g1.py, configs/g1_protocol.json, tests/test_g1_pipeline.py, output/g1/**, scripts/audit_gfs_g0.py, tests/test_audit_gfs_g0.py, output/gfs_g0/**, output/external_acquisition/gfs_g0/**
审查文件: docs/review/G1c-R2-codex.md
本轮决议: 采纳 2（R1-B1 隔离；R1-B2 如实记录、用户接受缺口）/ 反驳 0 / 暂缓 1（R1-S1，路径 A 不执行）/ 转实验 0；核验脚本两处直接修复复核正确

需要你做的:
1. **只审改动**（STAGE_PLAN B 节 B1 四条、末尾两条；DECISIONS G1c 表的 R1 三行和末段）。这是 G1c 第 2 轮，也是最后一轮。
2. **写 output/gfs_g0/b1/acceptance.json**（仿 derive_b2_acceptance，绑定 audit_b1_corrected.json 的 SHA 6f2dbba3…）：
   - status PASS_REQUIRED_WITH_QUARANTINE；complete_cycles 只收 complete = true 的周期；
   - excluded_cycles：7 个缺失加 2 个坏周期，各附原因，cycle_ok = 0；
   - 各年完整率；budget 段：唯一载荷 222.5 GB、download_total_verified = false、user_accepted_gap = 2026-11-13，加上从服务器日志只读估的上限及其算法。
   - 补一个测试，然后提交。
3. **收尾并运行（同第 76 次第 3 条，不用再等我）：** 走路径 T；脚本补完、测试过、协议提交并锁 PROTOCOL_SHA256；准备数据时 lead_audit 任何读取超出 [3, 18] h 就停。门禁都过就一次跑完 60 次拟合（先冻结后评估），结果拷回 output/g1/。
   - 交回：Q1/Q3/Q4/Q5 两场的增益、CI、p、max-p 和 Holm 结论；I* 选了谁；校准方法统计（rma/offset/masked 各几个）；机制诊断。
   - 任何一步失败就停下交回。不下载、不安装。

需要用户决定: 无（流量缺口用户已于 2026-11-13 接受）

## Codex → Claude
<!-- 由Codex覆盖。2026-11-13：第77次G1c R2，验收及脚本收尾 -->
轮次: R2/2 | 阶段: G1c | 分支: exp/G1 | 提交号: 上轮c1771e7已推送；本轮见git log
审查结论: R1-B1/B2处理通过，无未决方案阻断；docs/review/G1c-R2-codex.md。
已直接修复: 无超范围改动；按批准范围实现derive_b1_acceptance、实际读取时效门禁、准备产物哈希及评估报告补齐。
运行信息: B1验收output/gfs_g0/b1/acceptance.json；绑定原核验报告6f2dbba3…，原FAIL报告保留。
关键数字: complete_cycles=4371；excluded=9（7缺失、2字段不完整），统一cycle_ok=0、不回退、不删除标签；2021/2022/2023完整率99.52%/99.86%/100%。
关键数字: 成功唯一载荷222526985575字节；下载总量未核实，user_accepted_gap=2026-11-13。日志假设场景估算230137867780字节，不是可证明的上界；算法和源日志哈希见traffic_estimate.json。
关键数字: 服务器验收核验16/16（1.066秒）、最终流水线16/16（1.378秒），无跳过；合成评估报告可序列化，真实准备另记。
运行信息: 路径T；协议SHA256=92e93dd4d1b62b1654c02bbb95299019f507903f13ef6fe71d2f6687003e8ed4，train_g1.PROTOCOL_SHA256固定；提交锁定后准备，全部门禁通过才启动60拟合。
运行信息: 当前真实拟合0；不下载/安装/解码封存；原160MB manifest_b1.json未提交。
需要 Claude 做的: 本轮执行结束后复核验收名单、脚本和真实结果；当前无需再批准相同范围。
需要用户决定: 无。
