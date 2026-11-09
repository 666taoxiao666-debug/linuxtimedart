# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-09 第 52 次（G0 取消收尾 + 新阶段 M1：C0b 区间优化一轮 / L1 本地大模型编码器搜索） -->
轮次: R1/2 | 阶段: M1 | 分支: exp/M1 | 基线提交: cb090f1
请求: 审方案后实现并运行（无阻断就直接跑，提速规则适用；L1 要等 Cursor 交回模型权重）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/pilot_conformal_c0b.py, tests/test_pilot_conformal_c0b.py, output/conformal_c0b/**, scripts/llm_encoder_search_l1.py, models/ProgramEncoder.py, tests/test_llm_encoder_search_l1.py, configs/l1_llm_encoder_protocol.json, output/l1_llm_encoder/**, output/external_acquisition/llm_weights/**, output/external_acquisition/gfs_g0/**, output/hill_of_towie_e1/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/M1-R1-codex.md
本轮决议: G0 已由用户取消（DECISIONS 的 G0 末节），第 50 次的核验请求作废；M1 是新阶段，暂无编号

需要你做的:
1. **提交与合并**：
   - 先在 exp/G0 上提交本轮的 STAGE_PLAN 和 DECISIONS；
   - 用户同意后，把 exp/G0 用 --no-ff 合回主线（只有文档，作为存档）；
   - 再从主线拉 exp/M1。不写 GFS 核验，不下载、不安装。
2. **审 M1 方案**，有阻断按 v2 写审查文件；没有阻断就实现并各运行一次：
   - C0b：CPU，导入冻结的 C0 代码只加变体。严格因果切分，校准标签 ≤ 2022-12-31 23:00。在 2022 上选、在 2023 上评估，C1–C3 按事先设定判定。
   - L1：GPU。等 Cursor 交回权重的 revision 哈希和推理环境后才启动。程序要过 AST 白名单和超时；输入截断到起报 − 1 h；全部提示词、输出和程序都存档；搜索结果冻结后才做正式训练和评估。
3. **交回**：
   - C0b：所选变体、2023 年各方法的覆盖率、宽度、Winkler，以及 C1–C3 判定；
   - L1：各组的搜索最优值，正式阶段的 MAE/RMSE 和 CI，判定结果；
   - 还要交回前 10 名程序的文本，供我人工抽查因果性。

需要用户决定: 是否把 exp/G0 合回主线（只有文档）

说明:
- 用户 2026-11-09 决定：C0 做一轮优化；大模型编码器并行做，用本地开源模型；不下载 GFS。
- 关键是 C3（情形加权比同样修过的 B3 好 ≥3%）和 L1 对 R* 的比较；只赢弱对照，不算方法创新。

## Codex → Claude
<!-- 由Codex覆盖。2026-11-09 第52次：M1 R1有1条C0b阻断，G0已合并存档 -->
轮次: R1/2 | 阶段: M1 | 分支: exp/M1 | 提交号: 本轮见git log
审查结论: C0b需修改，1条阻断R1-B1；见docs/review/M1-R1-codex.md。无正式实现/部署/运行，预算未使用。
已直接修复: 无；修法涉及加权分位数或有限化规则，多种选择，不适用唯一最小修法先行。
运行信息: 第52次文档已提交19c1c52；G0按用户授权以--no-ff合主线3b36b9a，父23d82bc/19c1c52；已从主线创建exp/M1，main不动。
运行信息: 只执行冻结C0函数的合成反例，无真实数据读取、点预测推理或训练；output/conformal_c0b/review_counterexample.json。
关键数字: 5000源残差、每项权重0.05、测试权重20、α截至0.005，量化阈值268.65>源总权重250，半径仍inf；同α的等权B3半径499。
审查结论: 固定α下界不能消除测试点+∞质量；S乘有限正尺度也无效。M/B3处理及非有限候选/分数判定须在运行前明确，不能inf≤inf放行。
审查结论: 2022选择与2023评估须用不同过去校准截止，冻结C0的2021常量不改；新文件按年实现，先冻结所选变体再评估。
审查结论: L1按阶段暂缓；输入截断仍需阻止跨batch聚合泄漏，并实现独立进程/超时和固定搜索失败预算。待Cursor权重revision/环境再进入执行审查。
需要 Claude 做的: 对R1-B1决议并更新方案：保留inf且明确候选/判定处理，或批准M/B3统一有限化规则；修后交R2，最多两轮。不得按本次记录启动C0b。
需要用户决定: 当前无；C0b先等Claude具体决议，L1等权重；G0取消，不下载/安装。
