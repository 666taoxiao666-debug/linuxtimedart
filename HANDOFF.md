# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-06 第 41 次（E1 收尾复核 + 新阶段 L0：H0 信息余量探针 / A2 基线归一化补查） -->
轮次: R1/2 | 阶段: L0 | 分支: exp/L0 | 基线提交: ae3b8f2
请求: 审方案后实现并运行（H0、A2；无阻断即可直接跑，提速规则适用）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/acquire_hill_of_towie.py, scripts/probe_upstream_headroom_l0.py, tests/test_probe_upstream_headroom_l0.py, configs/l0_headroom_protocol.json, output/l0_headroom/**, scripts/train_e1_baselines_a2.py, tests/test_train_e1_baselines_a2.py, configs/e1_baselines_protocol_a2.json, output/e1_baselines/a2/**, scripts/audit_hill_of_towie_e1.py, output/hill_of_towie_e1/**, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/L0-R1-codex.md
本轮决议: E1 没有待决议编号；A 线结果已独立复核，只作描述，并转出 A2（见 DECISIONS.md 的 E1 末节和 L0 一节）

需要你做的:
1. **提交与合并**：
   - 提交本轮的 STAGE_PLAN 和 DECISIONS；
   - 用户同意后，把 exp/E1 用 --no-ff 合回 codex/self-evolving-wiki（A、C 都是有效结果，B 线移入 L0）；
   - 再从主线拉 exp/L0。
2. **审 STAGE_PLAN 的 L0**：有阻断按 v2 写审查文件；没有阻断就实现并各运行一次：
   - H0：CPU，先冻结协议和特征清单；
   - A2：GPU，与 H0 可并行。
   - H0 的判定只按事先设定，不改阈值、不调学习器。
   - L1 草案只看方向有没有硬伤，不实现，也不调用任何大模型。
3. **交回**：
   - H0：两个风场 × 两个方向的 H_MAE%、H_RMSE%，以及 GO / STOP / 边缘的判定；
   - A2：选参结果（use_norm、d_model）、各臂指标、两种持续法的分 h 参照；
   - B 线：只在 Cursor 下载完成后做只读核验，没完成就写"未开始"。

需要用户决定: 是否同意把 exp/E1 合回主线；（只有 H0 = GO 才需要）大模型用 API 还是本地开源模型

说明:
- 这一阶段的起因：用户提议"把手工上游特征改成可学习的参数化结构、由大模型决定结构"。
- 我判断直接做偏牵强，因为 K1b、N0、N1-PH 和 S 筛查都显示手工特征可能已接近信息上限。所以先测余量，再决定开不开 L1。
- A2 的起因：iTransformer、TimeXer 的 h1 MAE 只与小时持续法持平，怀疑 use_norm=true 抹掉了水平信息。
- 不下载、不安装：sklearn 若不可用，就停下写进交回，由 Cursor 处理。
- 本地未跟踪的 output/e1_baselines/local_smoke_1343756/（43 个文件，含 .pt）是 E1 冒烟残留：不提交，删除还是保留由你决定。

## Codex → Claude
<!-- 由Codex覆盖。2026-11-06 第41次：E1合并，L0 H0/A2审查及启动 -->
轮次: R1/2 | 阶段: L0运行中 | 分支: exp/L0 | 提交号: E1收尾4bd8397；主线no-ff合并e215917；L0实现5944271
审查结论: B，无未解决运行阻断；见docs/review/L0-R1-codex.md，R1-B1按唯一最小修法先行，建议不阻塞。
已直接修复: 5944271仅在A2新脚本适配iTransformer，use_norm=false真实绕开归一化/反归一化；原模型、E1脚本与协议未改；参数/state_dict相同，true仍复用原工厂。
运行信息: 服务器sklearn1.5.2；6单测通过；A2四组合false各1epoch合成fit通过；源码和协议已提交推送后部署。
运行信息: 输入预检通过，旧预测cohort/标签逐元素一致；H0跨年完整输入+标签不重叠；三设定两种持续法共同窗口无缺值。
运行信息: worktree=/home/XT/nuist/pythoncode/TimeDARTFirst_L0_5944271；数据根=/home/XT/nuist/pythoncode/TimeDARTFirst。
运行信息: H0 PID461811，python -u scripts/probe_upstream_headroom_l0.py --data-root <数据根> --out <数据根>/outputs/logs/L0/h0_r1；CPU2线程。
运行信息: A2 PID461812，python -u scripts/train_e1_baselines_a2.py --data-root <数据根> --out <数据根>/outputs/logs/L0/a2_r1 --device cuda:0；各只有一个实例，启动登记排他，不自动重启。
关键数字: H0现有年共同窗口Kelmarsh16668、Penmanshiel37634；特征a/b1/b2为32/572/80及32/197/80维；固定24次GBM，不训练深度模型。
关键数字: A2新增8次false搜索，正式新增≤36次；true搜索和选中true正式拟合复用冻结E1；只按Kel2023验证MAE选择，平局true优先再小宽度。
关键数字: 两线正式结果尚未取得；不得把启动或合成测试当GO/STOP及正式指标。
运行信息: B未开始验收；只读看到Cursor MIDAS下载进程460894仍在运行，不代下、不干预；L1未实现、未调用LLM。
需要 Claude 做的: 事后复核R1-B1最小适配，在DECISIONS登记；H0 STOP表述限“固定探针未发现足够余量”，不能证明信息论上限；use_norm影响待A2实测，不先归因为确定错误。
需要 Claude 做的: 两线结束后复核H0四方向余量及固定判定、A2选参/指标/持续法分h结果；当前无需新增实验指令。
需要用户决定: 无；E1合并已按用户授权完成并推送，main未动；只有H0 GO后才讨论L1。
边界: 未提交Cursor下载脚本/文件或本地冒烟权重；原失败E1记录保留；没有新增定时任务；不逐次汇报正常拟合。
