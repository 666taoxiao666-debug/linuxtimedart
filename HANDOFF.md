# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-03 第 21 次（P2 第 2 轮） -->
轮次: R2/2 | 阶段: P2 | 分支: exp/P2 | 基线提交: 3ae8655
请求: 审方案（只看 R1 修改）+ 实现并运行训练（24 次拟合）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md AGENTS.md
允许改动: AGENTS.md, scripts/train_penmanshiel_p2_v35.py, tests/test_train_penmanshiel_p2_v35.py, output/penmanshiel_hourly/p2_v35/**
审查文件: docs/review/P2-R2-codex.md
本轮决议: 采纳 3 / 反驳 0 / 暂缓 0 / 转实验 0（R1-B1、R1-S1、R1-S2，见 DECISIONS.md 的 P2 一节）

需要你做的:
1. **复审**：只看 STAGE_PLAN 中改动的地方，包括判定表措辞、"表述"和"归因"两条、Scratch-N 一行，以及少样本口径的切分、窗口、统计量、测试四条；再加 DECISIONS.md 的 P2 一节。
2. **实现与运行**：没有阻断，就实现 `train_penmanshiel_p2_v35.py` 和测试，测试要包含 R1-B1 要求的反例：改动验证段之后，Scratch 的统计量和训练目标不变，跨分界点的窗口被排除。之后在服务器上跑测试和 smoke，再跑 1 个真实实例。实现中遇到确定缺陷，可在允许改动的文件里直接修，最多 2 个真实实例。
3. **结果**：拷到 `output/penmanshiel_hourly/p2_v35/`，不拷权重和预测数组；核对 SHA256；launch.json 写 `source_code`。跑 `--codex` 之后，按 v2 覆盖你那一节，写明：R、Z1、Z2、F1、F3 各自的判定；各臂 2021 年的 MAE/RMSE（3 个 seed 的平均）；增益的 CI 和 Holm p。

需要用户决定: 无

说明:
- 这是 P2 的第 2 轮，也是最后一轮。复审如果还有阻断，就交用户裁决，不进入第 3 轮。
- **新规则（用户 2026-11-03 同意，已写进 AGENTS「先审后改」）**：方案中的确定缺陷如果修法唯一、又不涉及判定口径，可以按最小修法先实现、先运行，标【已按最小修法先行】，由我事后复核。本轮起生效，请顺带审一下 AGENTS.md 这几行。
- 只有 R 决定是否提请进入 P3；没过门槛的一律写成"本次开发实验未获得支持"。
- 边界：不碰 Penmanshiel 2022 年及以后的数据；K-* 只用 Kelmarsh 2016–2020 年训练；不追加臂，不调超参数，不改冻结模块。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-11-03，P2复审通过并启动第1真实实例 -->
轮次: R2/2 | 阶段: P2 | 分支: exp/P2 | 提交号: 实现e3ae7a6；启动记录见本次提交
审查结论: 通过，评级B，0条剩余阻断；docs/review/P2-R2-codex.md，R1-B1/S1/S2全部解除；AGENTS新增提速条款通过。
已直接修复: 无额外设计变更；按Claude批准范围实现两个新文件，未改冻结模块或STAGE_PLAN。
运行信息: 服务器timedart环境10/10合成测试通过；smoke八个臂各1seed/1epoch全部完成，11.029秒，保存数组复算一致；smoke不是正式实验指标。
运行信息: 本地语法检查通过；默认Python缺torch、timedart加载超时，不记作功能测试通过；服务器首次旧HEAD发现0测试也未计入通过证据。
运行信息: 第1真实实例PID243138，脱离SSH运行；服务器worktree /home/XT/nuist/pythoncode/TimeDARTFirst_P2_e3ae7a6，HEAD e3ae7a6。
运行信息: python -u scripts/train_penmanshiel_p2_v35.py --device cuda:0 --data-root /home/XT/nuist/pythoncode/TimeDARTFirst --out /home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Penmanshiel/p2_v35_e3ae7a6_r1。
运行信息: 启动前核对smoke完成、源码SHA、同类Python和锁；launch.json登记source_code，process.json登记完整命令；保留日志，不重复启动存活进程。
关键数字: 8个训练臂×3seed=24拟合；参数last10=4262/upstream=8126；未加臂或调整超参数；FT在epoch0前加载同seed的K-upstream最佳权重。
关键数字: Scratch仅用各N月前80%拟合统计量；零样本/FT直接使用Kelmarsh统计量；10项测试覆盖扰动不变性、目标边界、5比较Holm、模型计数和微调初始状态。
关键数字: 尚未完成24拟合，无正式2021开发MAE/RMSE/R²或五比较判定；全部checkpoint选定后才评估，不运行P3。
需要 Claude 做的: 等真实实例完成并回传结果后，复核实现与结果；本阶段未合主线，P1主线仍882f30c，main未动。
需要用户决定: 无。
