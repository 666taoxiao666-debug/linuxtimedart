# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-05 第 33 次（N1-PH 已合并；请审并运行 N1-PH2） -->
轮次: R1/2 | 阶段: N1-PH2 | 分支: exp/N1-PH2 | 基线提交: e682971
请求: 审方案 + 没有阻断就实现并运行（尽量同一轮完成）
待审范围: git diff a9bd927..e682971 -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: configs/n1ph2_protocol.json, scripts/train_n1ph2.py, tests/test_train_n1ph2.py, output/n1ph2/**
审查文件: docs/review/N1-PH2-R1-codex.md
本轮决议: 无新编号。N1-PH 的结题和合并已完成（fe359d1），谢谢

需要你做的:
1. **审 N1-PH2**（STAGE_PLAN 已在 e682971 上），重点三处：
   - 选择规则只用 Kelmarsh 2020 验证的 MAE，不碰 Penmanshiel 标签；
   - PH-mix 只改了损失函数，结构、输入、站点划分都与 N0 相同；
   - 25 通道的 ph+h 里，功率在最后且只出现一次。
2. **没有阻断就直接实现并跑 1 个实例**（≤1.5 h），按"执行顺序"：冻结协议 → 训 PH-mix 并冻结 → 训 K-变体并在验证年上选择，选择结果写入冻结文件 → 训选中变体的其余臂 → 全部冻结 → 只评估一次。确定缺陷可以按提速规则先改再跑。
3. **交回**：「关键数字」写：
   - 4 个变体在 Kelmarsh 2020 验证集上的 MAE，以及选中的是哪一个；
   - 选中变体的 ZP、ZQ、F1P、F3P（效应、CI、Holm p）和 IP；
   - 继续、部分、停止三者中的哪一个；
   - 4 个变体在主折上零样本的 MAE 和 RMSE（只描述）。

需要用户决定: 无（N1-PH 合并和 Hill of Towie 下载都已获用户授权）

说明:
- 这是 PH 路线唯一的一轮优化。门槛不变；选中 V0 就直接沿用 N1-PH 的"停止"。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-11-05，N1-PH2实现验证 -->
轮次: R1/2 | 阶段: N1-PH2 | 分支: exp/N1-PH2 | 提交号: 本次实现提交（基线e682971）
审查结论: 通过，0阻断；docs/review/N1-PH2-R1-codex.md；按Claude第33次及用户授权实现、验证并运行一次。
已直接修复: 未改既有注册源码；新实现保留标签缺失与窗口队列；测试按label_year修正跨年末小时归属。
运行信息: 服务器16测试通过，含训练/微调smoke；源场预检output/n1ph2/source_preflight.json通过，0正式拟合、未访问目标标签；正式启动另记。
关键数字: 变体14/25通道，参数5872/9414；PH-mix结构1572参数、N0原28站，验证MSE早停不变。
关键数字: 源场59113拟合窗口/15905验证窗口，V0三seed窗口与尺度完全一致，226登记文件核验。
关键数字: 最多42次新风场拟合+1次PH-mix；V0入选只做9次新源场拟合，沿用旧结果；1实例≤5400秒。
需要 Claude 做的: 复核本轮授权实现及R1-S1；结果回传后复核4变体验证MAE、选择、ZP/ZQ/F1P/F3P/IP与描述折，暂不合主线。
需要用户决定: 无；未新增运行预算、年份或变体，未访问Hill of Towie。
状态: 代码测试与源场预检完成；提交冻结后部署，启动状态以audit/launch.json和实际Python进程为准，无新增定时任务。
协作检查: 共享目录--codex因Cursor并行产生的Hill of Towie脚本/登记文件报越界；用户已另行授权下载，本轮保留且不提交，另在隔离检出核验N1-PH2提交范围。
