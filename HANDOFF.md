# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-05 第 30 次（N1-PH 的 R1 决议，正式实例运行中） -->
轮次: R2/2 | 阶段: N1-PH | 分支: exp/N1-PH | 基线提交: 8be1279
请求: 只运行（正式实例跑完后交结果）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: output/n1ph/**
审查文件: docs/review/N1-PH-R2-codex.md
本轮决议: 采纳 3 / 反驳 0 / 暂缓 0 / 转实验 0（R1-B1、R1-S1、R1-S2）；41a5b74 的缺失时间行修复事后复核通过（见 DECISIONS.md 的 N1-PH 一节）

需要你做的:
1. **不动正式实例**（PID 371498）：不改部署代码，不重启，不补跑，等它跑完。
2. **跑完以后**：核验结果，拷到 `output/n1ph/`，核对 SHA256。按 v2 覆盖你那一节，「关键数字」写：
   - 主折上 ZP、ZQ、F1P、F3P 的判定（效应、CI、Holm p）；
   - IP 的非劣结果；
   - 继续、部分、停止三者中的哪一个；
   - T22、T23 的同方向数字；
   - 各臂 2021 年的 MAE/RMSE。
3. 本轮 STAGE_PLAN 和 DECISIONS 只改了文字（通道数、"开发比较族"的叫法、缺失小时、归因），协议 JSON 不动。顺带在 exp/N1-PH 上提交即可。

需要用户决定: 无（N1-PH 结果复核后再请用户同意合并）

说明:
- 即使 ZP 获得支持，也只叫"开发支持"，确认要靠第三个风场。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-11-05，Claude第30次决议复核通过 -->
轮次: R2/2 | 阶段: N1-PH | 分支: exp/N1-PH | 提交号: 本轮文档登记（父提交8be1279）；正式代码41a5b74
审查结论: 通过，无新阻断；R1-B1/S1/S2及缺失时间行修复均已闭环，见docs/review/N1-PH-R2-codex.md；原8项服务器测试通过，本轮未重复测试。
已直接修复: R1-B1（2c8201c），14通道、5872参数；41a5b74修正缺失时间行映射，按N0均值填补H11再推断；未改标签/臂/阈值/年份。
运行信息: 新增configs/n1ph_protocol.json、scripts/train_n1ph.py、tests/test_train_n1ph.py；冻结历史源码不改。
运行信息: 本地py_compile、服务器8项测试及只读preflight通过；证据output/n1ph/preflight/，三折starts/truth/times/normal_mask逐元素一致。
运行信息: 正式PID371498；worktree=/home/XT/nuist/pythoncode/TimeDARTFirst_N1PH_2c8201c（实际HEAD41a5b74）。
运行信息: run=/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Penmanshiel/n1ph_41a5b74_r1；python -u scripts/train_n1ph.py --data-root /home/XT/nuist/pythoncode/TimeDARTFirst --out <run> --device cuda:0。
运行信息: 最新只读检查24/36拟合完成，进程命令匹配且正常；无failed/completed，最终评估尚未开始；无重复启动、无新增定时任务。
关键数字: PH checkpoint SHA256=584fa1e929eae27576ed49679b695c3e5ce7696c3baab5c53ff88f819e50b7e0，服务器复算一致；N0归一化两向量均11维有限。
关键数字: 3折×4臂×3seed=36，1实例≤5400秒；ZP/ZQ/F1P/F3P为主折开发族，IP单独非劣，T22/T23只描述。
关键数字: 协议SHA256(LF)=6b0aabe483c7d1d44af7f2f6e98fad85ac7c40c3c1515dca3b3cea1762818e48；主折FT1窗口2443/686、FT3窗口7698/1989，与P2一致。
需要 Claude 做的: 本轮决议已登记，无需重复审方案；等待结果交回后复核效应/CI/Holm/IP和继续/部分/停止结论，暂不修改已部署代码或合并。
需要用户决定: 无。
状态: 等待唯一正式实例结束，禁止修改部署源码或新增实例；结果核验后另次提交交Claude复核，未经批准不合主线。
