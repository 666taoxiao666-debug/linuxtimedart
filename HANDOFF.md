# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-05 第 29 次（N0 结题 + N1-PH 首轮） -->
轮次: R1/2 | 阶段: N1-PH | 分支: exp/N1-PH | 基线提交: 9f125d0
请求: 审方案（N1-PH STAGE_PLAN）+ 没有阻断就实现并运行
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: configs/n1ph_protocol.json, scripts/train_n1ph.py, tests/test_train_n1ph.py, output/n1ph/**
审查文件: docs/review/N1-PH-R1-codex.md
本轮决议: N0 一节没有新编号；N0 结果我已用 result.json 的 56 对原始记录独立复核，与你一致，判为有效负结果

需要你做的:
1. **N0 结题**：先在 exp/N0 上提交 DECISIONS.md 和 HANDOFF.md。用户同意的话，用 `--no-ff` 把 exp/N0 合回主线并推送，exp/N0 保留。STAGE_PLAN 先不提交，带到 exp/N1-PH。
2. **N1-PH**：从主线拉 `exp/N1-PH`，审 STAGE_PLAN。重点审三处：PH 通道有没有时间泄漏；零样本时是否只用 Kelmarsh 和 N0 的统计量；复用 P2、P3 数组时窗口是否一致。没有阻断，就按"执行顺序"实现并跑 1 个实例（≤1.5 h）：先冻结协议，再核对哈希，再完成 36 次拟合，最后只评估一次。确定缺陷可以按提速规则处理。
3. **交回**：「关键数字」写：
   - 主折上 ZP、ZQ、F1P、F3P 的判定（含效应、CI、Holm p）；
   - IP 的非劣结果；
   - 继续、部分、停止三者中的哪一个；
   - 描述折 T22、T23 的同方向数字；
   - 各臂 2021 年的 MAE/RMSE。

需要用户决定: 合并 N0（用户转话时一并说明）

说明:
- PH 的迁移优势来自 N0 的事后观察，所以 N1-PH 作为新假设在风场上重新登记。创新预期约 5–5.5，论文走 CCF-B 的应用赛道。
- 这是 N1-PH 的第 1 轮。方案如果有阻断，而且修法不唯一，就停在冻结协议之前交回。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-11-05，N1-PH正式运行中 -->
轮次: R1/2 | 阶段: N1-PH | 分支: exp/N1-PH | 提交号: 实现2c8201c；修复/运行41a5b74
审查结论: 方案通过，R1-B1唯一最小修法已实施，8项服务器合成测试（含训练smoke）通过；见docs/review/N1-PH-R1-codex.md。
已直接修复: R1-B1（2c8201c），14通道、5872参数；41a5b74修正缺失时间行映射，按N0均值填补H11再推断；未改标签/臂/阈值/年份。
运行信息: 新增configs/n1ph_protocol.json、scripts/train_n1ph.py、tests/test_train_n1ph.py；冻结历史源码不改。
运行信息: 本地py_compile、服务器8项测试及只读preflight通过；证据output/n1ph/preflight/，三折starts/truth/times/normal_mask逐元素一致。
运行信息: 正式PID371498；worktree=/home/XT/nuist/pythoncode/TimeDARTFirst_N1PH_2c8201c（实际HEAD41a5b74）。
运行信息: run=/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Penmanshiel/n1ph_41a5b74_r1；python -u scripts/train_n1ph.py --data-root /home/XT/nuist/pythoncode/TimeDARTFirst --out <run> --device cuda:0。
运行信息: 最新只读检查13/36拟合完成，进程命令匹配且正常；无failed/completed，最终评估尚未开始；无重复启动、无新增定时任务。
关键数字: PH checkpoint SHA256=584fa1e929eae27576ed49679b695c3e5ce7696c3baab5c53ff88f819e50b7e0，服务器复算一致；N0归一化两向量均11维有限。
关键数字: 3折×4臂×3seed=36，1实例≤5400秒；ZP/ZQ/F1P/F3P为主折开发族，IP单独非劣，T22/T23只描述。
关键数字: 协议SHA256(LF)=6b0aabe483c7d1d44af7f2f6e98fad85ac7c40c3c1515dca3b3cea1762818e48；主折FT1窗口2443/686、FT3窗口7698/1989，与P2一致。
需要 Claude 做的: 结果交回后复核R1-B1最小修法并修正STAGE_PLAN通道计数；在DECISIONS登记决议；本轮结果不作为独立确认。
需要用户决定: 无。
状态: 等待唯一正式实例结束，禁止修改部署源码或新增实例；结果核验后另次提交交Claude复核，未经批准不合主线。
