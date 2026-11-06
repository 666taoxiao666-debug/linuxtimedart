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
<!-- 由Codex覆盖。2026-11-06 第40次：A线仅评估恢复已获用户具体授权 -->
轮次: R2/2 | 阶段: E1只恢复评估 | 分支: exp/E1 | 提交号: 批准记录8a2eaa5；恢复实现6204100
审查结论: 通过，评级B；R2-P1已获Claude采纳和用户明确例外授权，见docs/review/E1-R2-codex.md末节。
已直接修复: 6204100按批准范围实现新配置/恢复入口；v32参照manifest已纳入Git；不依赖漏部署的seed JSON，改用v32 completed登记的预测哈希。
运行信息: 服务器7项测试通过（含缺参照文件不写评估标记、篡改冻结记录拒绝、恢复路径不调用fit/choose）。
运行信息: 原64拟合及全部登记产物哈希、原源码和未改共享代码身份检查通过；v32 manifest哈希a8cc5708…通过。
运行信息: 新worktree=/home/XT/nuist/pythoncode/TimeDARTFirst_E1_eval_6204100；PID446391，真实完整命令已确认。
运行信息: python -u scripts/train_e1_baselines.py --data-root /home/XT/nuist/pythoncode/TimeDARTFirst --out /home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/E1/e1_b5be60a_eval1 --device cuda:0 --recover-evaluation
运行信息: 输出目录固定唯一且拒绝复用；父目录launch_once排他登记已写入，原r1保持只读；已于北京时间13:44:26完成，无failed标记；本次仅一次评估。
关键数字: 复用64拟合（10搜索+54正式），新增训练=0、重新选参=0；三设定/六臂/三seed，评估仅一次。
关键数字: C0维持stop，M覆盖81.2799%未达88–92%；B2/M+无限区间照实保留；Claude已复核，不再优化或补跑。
需要 Claude 做的: 复核output/e1_baselines/RESULT_SUMMARY.md及eval1/result.json、independent_verification.json；各臂指标、增益CI、相对TSMixer差距均已列出，全部仅描述。
关键数字: TSMixer在Kel/T22/T23的MAE=166.2789/213.6308/188.8388（精确值见result），三设定全部新增基线均更差。
关键数字: TimeXer上游MAE增益4.0119/4.6883/4.1757kW，95%CI均>0；iTransformer为4.0338/1.0364/0.3902kW，后两组CI跨0。
运行信息: 57数组哈希、64拟合原产物重新核验；保存数组复算全部指标和24比较，结果见独立核验JSON；无新增定时任务。
需要用户决定: 无；结果待Claude复核，未合主线。
边界: 未改原冻结协议/旧worktree、未重新训练/选参、未合主线、main未动、无新增定时任务、未提交Cursor下载文件。
