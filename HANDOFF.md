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
<!-- 由Codex覆盖。第83次G2b：14拟合完成、冻结核验通过，停下交回 -->
轮次: R1/2 | 阶段: G2b | 分支: exp/G2 | 提交号: 4471499（限额运行代码）、a065430（启动记录），本提交登记冻结验收
审查结论: 第1、2步完成；见docs/review/G2b-R1-codex.md；仅S1建议，不把诊断当改造因果效应。
已直接修复: 按用户2026-11-15明确授权改共享资源门禁；无实验阈值、配方、臂或年份变动。
诊断结果: 21台18台S2≤−5%、3台|S2|<5%，选择A；完整原始状态分布压缩包及SHA在output/g2b/retrofit_diagnostic/。
运行信息: worktree=/home/XT/nuist/pythoncode/TimeDARTFirst_G2_4471499；run=/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/G2/g2b_4471499_shared_r1/formal。
运行信息: PID739134已退出；北京时间2026-11-15 00:33:19启动、03:13:56完成，总耗时约2小时41分；14/14，FROZEN_DEVELOPMENT_ONLY，无failed.json。
运行信息: 原PID719679仅等待且0拟合时取消留档；未增加拟合预算，其他用户进程未动。
资源限制: PyTorch allocator1024MiB（CUDA上下文另计）；CPU2核/2线程、nice10；共享算力无硬配额。
验证: 服务器32/32测试通过；14份checkpoint、trained.json、最优验证MAE/epoch、源码/协议/准备产物及追加开发输入SHA全部核验通过。
验收文件: output/g2b/freeze/；23份原始JSON/日志/PID逐文件SHA一致，含14份trained.json；未拷权重、原始数据或预测数组。
冻结SHA256: 981f3ae27e6792f5da71c177af1c24a0e91c4dfcd880322773fb4f9fe34607f6。
协议SHA256: da809d691e93b7bb284fc7ce282f1c16a59ecd0fe5342f68ceab0e8530966a53。
脚本SHA256: 411095d75a8eb8bcaa6ad3cfd5641f5e3a0a3fd909e5316f5ee2cfeec4985823。
关键数字: 情形A拟合420475窗/验证30056窗，改造后8.5287%/100%；拟合截至2023-06-30，验证2023-07至12月。
关键数字: 验证MAE种子均值kW：B0(3)=221.9221，B(3)=202.7767，N0(3)=184.9365，N(5)=182.2926。
关键数字: N前3seed=182.2244，对B降低20.5524kW，对N0降低2.7122kW；仅验证描述，无独立检验或显著性结论。
关键数字: N五种子均值是单模型MAE平均，不是集成预测MAE，不能据此判断Qe。
封存门禁: 未产生test_evaluation_started.json；未访问2024–2026观测值或解码封存GFS，冻结后已停止。
需要 Claude 做的: 复核freeze/SUMMARY.md、verification.json、formal/selection_freeze.json和封存准备/评估入口；通过后提出第3步的明确授权范围及已批准GFS输入清单。
需要用户决定: 当前无；完成本轮后停下，封存年仍需另行授权，不合主线、不重训或补跑。
