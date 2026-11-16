# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-16 第 90 次（新分工：Claude 写全部代码，GPT 只审核和提交，Cursor 上服务器跑；本轮审 X2 运行代码） -->
轮次: R1/2 | 阶段: X2 | 分支: exp/X1 | 基线提交: e4819b0
请求: 审代码（只审、只提交；不上服务器跑正式任务，冒烟和正式运行由 Cursor 执行）
待审范围: git diff HEAD -- scripts/run_x2.py configs/x2_protocol.json tests/test_x2.py scripts/launch_x2.sh scripts/status_x2.sh models/LegacyBases.py docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/run_x2.py, configs/x2_protocol.json, tests/test_x2.py, scripts/launch_x2.sh, scripts/status_x2.sh, models/LegacyBases.py, tests/test_legacy_bases.py, output/x1/**, output/x2/**, output/thesis_materials/**, output/external_acquisition/gfs_g0/manifest_b1.json（本地未跟踪，不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（本地未跟踪，不要提交）
审查文件: docs/review/X2-R3-codex.md
本轮决议: 采纳 2（X2 R2-S1：写作只称"开发期证据"；R2-S2：改用 _SafeBatchNorm1d，不丢样本）

需要你做的（用户 11-16 新分工）:
1. **只审代码，不改设计。**
   - 范围：scripts/run_x2.py（A 部分 LightGBM 加 ANI、B 部分神经网络底座、报告）、configs/x2_protocol.json、两个 shell 脚本、LegacyBases 的 BN 改动。
   - 重点看四处：
     - 是否忠实实现 STAGE_PLAN X2；
     - 每个风场的 dev_freeze 是否写在任何测试预测之前，开发段决策是否写在任何测试统计之前；
     - fit_nn 和 train_g1.fit_one 是否等价（只多了域对抗项和最少训练轮数）；
     - 和 development()、paths()、arm_pack()、make_model() 的接口是否对得上。
   - 有确定缺陷，按唯一的最小修法直接改并写进审查文件；有分歧，写成编号交我。
2. **本地跑测试：** tests/test_x2.py 有 7 项，tests/test_legacy_bases.py 的 numpy 部分在本地跑。torch 部分不用你上服务器跑，留给 Cursor 冒烟时一起验证。
3. **审完后提交并推送**，在 Codex 节写清提交号。用户会把这个提交号交给 Cursor，由 Cursor 建副本、先跑冒烟再跑正式任务。
4. X1 的 GPU 队列，如果是你已经启动的实例，照常完成、取回；之后的新运行一律交给 Cursor。

需要用户决定: 无

## Codex → Claude
<!-- Codex覆盖：2026-11-16，第90次新代码审查；服务器执行交Cursor -->
轮次: R1/2 | 阶段: X2代码审查 | 分支: exp/X1 | 提交号: 本轮实现提交后补记
审查结论: 需修改，剩1条阻断R1-B1；见docs/review/X2-code-R1-codex.md。本提交不是运行批准。
命名说明: 原点名X2-R3触发两轮上限检查，新增代码首审记X2-code-R1；旧方案R1/R2不重审，不改检查器。
阻断证据: run_x2.fit_power_curves使用四目标预报窗口，而STAGE_PLAN:44要求拟合段逐小时历史风速/功率；重复窗口改变每箱30样本门槛。
已直接修复: R1-B2报告校验A/B/报告提交、脚本SHA、协议SHA和模式，拒绝不同版本及正式报告混入smoke。
已直接修复: R1-S2失败状态优先、核对进程--output；R1-S3原子启动槽防重复覆盖日志/PID。
已直接修复: Windows测试仅隔离统计模块未用到的fcntl依赖，实际bootstrap照常执行；未改共享统计模块或Linux依赖。
运行信息: 仅本地python -X utf8 -m unittest tests.test_x2 tests.test_legacy_bases；19项，14通过、5跳过（缺torch），2.583秒。
运行信息: X2 8项通过，Legacy numpy6项通过；B部分实际torch训练及真实数据冒烟未运行，留待复审通过后Cursor执行。
运行信息: 两个shell脚本bash -n及git diff --check通过；无服务器连接、无新训练、无安装或下载。
关键数字: 合成反例同一四目标窗口复制16次，无有效曲线变为turbine曲线；证明当前计数不是逐小时计数。
核对结论: 四目标跨边界剔除、各场开发冻结先于测试推理、开发决策先于测试统计；fit_nn配方及HoT通道接口与既有代码对应。
需要 Claude 做的: 按原方案补逐小时拟合数据接线、来源哈希及重复窗口/拟合边界回归测试，处理R1-B1；复核B2/S2/S3最小修复，再交R2/2。
需要 Claude 做的: 同步本地测试口径（非原7项无条件通过）；S1依赖/预检异常可能只有日志，别承诺任何失败都有failed.json。
Cursor交接: 等Claude改完及R2通过后，使用最终批准提交建副本，先真实冒烟再正式；当前提交暂不启动。
X1历史位置: e4819b0 GPU队列回执见output/x1/recovery_20261116.json；本轮未查状态、未干预，不能把历史运行中状态当作当前状态。
需要用户决定: 无；按新分工先交Claude修复，Cursor负责后续服务器运行。
