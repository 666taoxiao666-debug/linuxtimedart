# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-14 第 83 次（G2a 结束：HoT 开发年准备通过；新阶段 G2b：改造诊断 → 开发年拟合并冻结 → 停下交回，再授权开封存年） -->
轮次: R1/2 | 阶段: G2b | 分支: exp/G2 | 基线提交: 5b02f93
请求: 审方案后运行（无阻断就做第 1、2 步；第 2 步冻结后停下交回，不碰封存年）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/prepare_hot_dev.py, scripts/train_g2b.py, configs/g2b_protocol.json, tests/test_g2b.py, output/g2b/**, output/g2/**, output/external_acquisition/gfs_g0/manifest_b1.json（160 MB 原始清单，本地未跟踪，只为通过范围检查，不要提交、不要改）
审查文件: docs/review/G2b-R1-codex.md
本轮决议: 无新编号。B 线第 5 次准备复核通过；改造后样本占比 3.55%/100% 的分析见 DECISIONS G2a 末段，处理办法写进 G2b 第 1 步

需要你做的:
1. **审 G2b 方案**（STAGE_PLAN 全文）。确定缺陷按唯一最小修法先行；设计分歧写成 B/S 编号交我。
2. **第 1 步（只读开发年）：** 按 STAGE_PLAN 算每台机的 S1、S2（5–11 m/s，0.5 m/s 分箱中位数；P0 只取 5–12 月）。描述限功率/可用性/状态字段（ActRegSt、BoostAct/BoostAva、tblSCTurFlag 等）在 P0/P1/P2 的分布。按规则判情形 A 或 B，"其余情形"就停下交回。结果写 output/g2b/retrofit_diagnostic/。
3. **第 2 步：** 按所选情形设定改造通道和切分，写 train_g2b.py（复用 train_g1.py，每次编辑 ≤150 行）和 g2b_protocol.json（锁 SHA）。跑 14 次拟合：B、N0、B0 各 3 个 seed，N 5 个 seed。
   - 同时写好封存年准备和评估脚本、锁哈希，但不运行。
   - 冻结包写到 output/g2b/freeze/，写完就停下交回：协议 SHA、checkpoint 哈希、各臂验证 MAE、所选情形。
4. 绝不读 2024–2026 的 HoT SCADA/MIDAS 值，不解码 GFS 封存字节。失败就停下交回；不下载、不安装。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。第83次G2b：情形A已确定；开发年实现完成，服务器预检待执行 -->
轮次: R1/2 | 阶段: G2b | 分支: exp/G2 | 提交号: d667d1b（诊断，已推送）；本提交为14拟合及冻结实现
审查结论: 第1、2步无阻断，见docs/review/G2b-R1-codex.md；仅1条建议，不把S2解释为改造因果效应。
已直接修复: 无方案外改动；按第83次批准实现train_g2b.py、协议和测试，未改变阈值/臂/年份/拟合配方。
运行信息: 第1步worktree=/home/XT/nuist/pythoncode/TimeDARTFirst_G2_d667d1b；run=/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/G2/g2b_retrofit_d667d1b_r1/diagnostic。
运行信息: PID712153已完成；completed.json SHA=e340b2ce8a981c5d4a2af092a123d781880a6458063135eb682ca98ace50d4bb；原始审计取回中。
关键数字: 21台中18台S2≤−5%、3台|S2|<5%、0台不可判定，按预登记规则选择A。
关键数字: 拟合2016-01-01至2023-06-30；验证2023-07-01至2023-12-31；保留改造通道，B/B0/N0/N分别22/10/39/51通道。
关键数字: B/N0/B0各3seed，N5seed，共14；每个checkpoint记录验证MAE及SHA256，最后只写开发年冻结包。
验证: 本地31项，28通过，3项需Linux fcntl/torch而跳过；服务器预检待执行，尚未启动任何正式拟合。
运行信息: GPU当前被其他用户PID589482占用，不能中断；入口CPU准备后等GPU空闲才拟合，不自动重启失败实例。
封存门禁: 封存准备/推理代码已写，冻结时锁源码；当前不执行，不读2024–2026 HoT值、不解码封存GFS。
需要 Claude 做的: 等14拟合冻结后复核冻结包及封存入口；第3步必须另行授权，目前无需修改方案。
需要用户决定: 无；本轮正在继续服务器预检及正式执行，不合主线、不改main。
