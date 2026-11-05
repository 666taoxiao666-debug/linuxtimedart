# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-05 第 32 次（N1-PH 结题 + N1-PH2 一轮优化） -->
轮次: R1/2 | 阶段: N1-PH2 | 分支: exp/N1-PH2 | 基线提交: a9bd927
请求: 审方案 + 没有阻断就实现并运行（尽量同一轮完成）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: configs/n1ph2_protocol.json, scripts/train_n1ph2.py, tests/test_train_n1ph2.py, output/n1ph2/**
审查文件: docs/review/N1-PH2-R1-codex.md
本轮决议: N1-PH 一节没有新编号；N1-PH 的结果我已独立复核，stop 判定成立

需要你做的:
1. **N1-PH 结题**：在 exp/N1-PH 上提交 DECISIONS.md 和 HANDOFF.md。用户同意的话，用 `--no-ff` 把 exp/N1-PH 合回主线并推送，exp/N1-PH 保留。STAGE_PLAN 先不提交，带到 exp/N1-PH2。
2. **N1-PH2**：从主线拉 `exp/N1-PH2`，审 STAGE_PLAN。重点审三处：
   - 选择规则确实不碰 Penmanshiel 的标签；
   - PH-mix 只改了损失函数；
   - 25 个通道里功率只出现一次。
   
   没有阻断就按"执行顺序"实现并跑 1 个实例（≤1.5 h）。确定缺陷可以按提速规则处理。
3. **交回**：「关键数字」写：
   - 4 个变体在 Kelmarsh 2020 验证集上的 MAE，以及选中的是哪一个；
   - 选中变体的 ZP、ZQ、F1P、F3P（效应、CI、Holm p）和 IP；
   - 继续、部分、停止三者中的哪一个；
   - 4 个变体在主折上零样本的 MAE 和 RMSE（只描述）。

需要用户决定: 合并 N1-PH；是否现在就下载 Hill of Towie（封存，只记哈希），作为最终确认用的第三个风场

说明:
- 这是 PH 路线唯一的一轮优化。门槛不变；选中 V0 就直接沿用 N1-PH 的"停止"。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-11-05，N1-PH结题合并 -->
轮次: R2/2 | 阶段: N1-PH结题 / N1-PH2待审 | 分支: exp/N1-PH | 提交号: 结果a9bd927；本次结题提交
审查结论: Claude第32次已独立复核N1-PH有效负结果，stop成立；用户已明确批准合回主线。
已直接修复: 无；本轮只登记复核和执行已批准合并。
运行信息: 原36拟合及评估已完成、核验通过；无重训补跑，无新增定时任务。
关键数字: ZP MAE增益−4.121 kW；四项主比较均不支持，IP非劣失败；原负结果完整保留。
需要 Claude 做的: N1-PH无需重复复核；N1-PH2方案带到独立分支后待正式审查，不视为通过或已运行。
需要用户决定: 无；合并N1-PH及由Cursor下载Hill of Towie已获授权。
状态: 本提交后以--no-ff合入codex/self-evolving-wiki并推送，保留exp/N1-PH；main不动。
下载交接: 用户自行转交Cursor；仅Hill of Towie原始字节封存/大小和哈希登记，不解压解析、不训练评估；尚无下载完成证据。
