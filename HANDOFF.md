# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-09 第 53 次（M1 R2：R1-B1 统一有限化规则 F + 非有限值判定；L1 规则提前写清） -->
轮次: R2/2 | 阶段: M1 | 分支: exp/M1 | 基线提交: d652a65
请求: 审方案后实现并运行（只审改动；无阻断就实现 C0b 并正式运行一次；L1 等权重）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/pilot_conformal_c0b.py, tests/test_pilot_conformal_c0b.py, output/conformal_c0b/**, scripts/llm_encoder_search_l1.py, models/ProgramEncoder.py, tests/test_llm_encoder_search_l1.py, configs/l1_llm_encoder_protocol.json, output/l1_llm_encoder/**, output/external_acquisition/llm_weights/**, output/external_acquisition/gfs_g0/**, output/hill_of_towie_e1/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/M1-R2-codex.md
本轮决议: 采纳 7（R1-B1、S1–S6）/ 反驳 0 / 暂缓 0 / 转实验 0；L1 的 S4–S6 现在就写进规则，L1 执行时不再另开一轮

需要你做的:
1. **提交**本轮的 STAGE_PLAN 和 DECISIONS。
2. **只审改动部分**，这是第 2 轮，也是最后一轮；仍有分歧就写明，交用户决定：
   - R1-B1：统一有限化规则 F（分位点落在 +∞ 质量时取组内最大残差，S 变体乘 κ），所有方法同处理；截顶占比要报告。
   - 非有限值：一律算技术失败；任何"≤"比较只要有一边非有限，就算不通过。
   - α 截断只用于限制 ACI 状态；两次校准截止分开实现；κ 异常报错；先冻结选择再评估。
   - L1：禁止跨 batch 运算 + batch 不变性测试；子进程隔离；失败程序占名额；各组设定一致。
3. **无阻断就实现 C0b 并正式运行一次**（新文件，导入冻结 C0，只加变体）：
   - 测试用合成数据覆盖：Codex 的 +∞ 反例在 F 下变为有限且等于组内最大残差；M 与 B3 同处理；非有限值触发技术失败；selection_freeze.json 在计算 2023 之前写入；两次校准截止都各自通过因果断言。
   - 交回：所选变体、2023 年各方法的覆盖率、宽度、Winkler、截顶占比，以及 C1–C3 判定。
   - L1 仍等 Cursor 交回权重 revision 和环境，到时按 STAGE_PLAN 的 L1 规则实现，跑完我再复核。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-09 第53次：M1 R2通过，C0b运行前冻结 -->
轮次: R2/2 | 阶段: M1 | 分支: exp/M1 | 提交号: 本轮见git log
审查结论: 通过，无遗留阻断；docs/review/M1-R2-codex.md；R1-B1及S1–S6已核对本轮改动。
已直接修复: 无额外设计变更；按用户及第53次批准实现scripts/pilot_conformal_c0b.py和tests/test_pilot_conformal_c0b.py；冻结C0未改。
运行信息: 本地7项合成测试通过；正式预算尚未使用，下一步服务器测试及--smoke通过后只启动一个正式实例。
运行信息: 2022选S/A/S+A，selection_freeze落盘并核对输入/源码后才计算2023；2023仅选中M、匹配B3和B0/B1/B2-A。
关键数字: F反例5000残差输出最大值5000且cap=true；M/B3共用κ和F；α界[.005,.5]，各机组/步长按真实区间延迟反馈。
关键数字: 两校准截止2021/2022年末23:00；非有限值技术失败；C1–C3阈值及年份不变，无点模型训练。
需要 Claude 做的: 待C0b正式结果回来复核实现/结果；L1待Cursor交权重revision与环境，当前未实现/运行L1。
需要用户决定: 无；exp/M1不合主线，main不动；正式失败不自动补跑。
