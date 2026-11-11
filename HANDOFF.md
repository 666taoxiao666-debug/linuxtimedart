# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-11 第 64 次（M5 复核通过；用户授权 L1 第 2 轮优化 O2，最后一轮） -->
轮次: R1/2 | 阶段: M6 | 分支: exp/M2 | 基线提交: 2f74cc3
请求: 审代码后运行（审 Claude 的反馈模块；实施 O2 和 r3 审计修正；预检后运行一次）
待审范围: git diff HEAD -- models/ProgramEncoder.py tests/test_program_encoder.py docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: models/ProgramEncoder.py, tests/test_program_encoder.py, scripts/llm_encoder_search_l1.py, tests/test_llm_encoder_search_l1.py, configs/l1_llm_encoder_protocol.json, output/l1_llm_encoder/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**（后三项是本地历史未跟踪文件，只为通过范围检查，不要改）
审查文件: docs/review/M6-R1-codex.md
本轮决议: M5 采纳 2（R1-B1、R1-B2 的最小修法都复核正确，M3 补算的定义与 Kelmarsh 一致）/ 反驳 0 / 暂缓 0 / 转实验 0。接受 RAMP_REPLICATED、NO_FALLBACK、STOP_L1；L1 r3 的可报告范围见 DECISIONS M5 节末段

需要你做的:
1. **审 Claude 的改动并跑测试**：models/ProgramEncoder.py 新增 feedback_breakdown（验证集误差分解，纯 numpy）和 render_feedback（固定措辞的反馈文字）。服务器跑 `python -m unittest tests.test_program_encoder tests.test_llm_encoder_search_l1`。
2. **按 STAGE_PLAN M6 实施 O2**（每次编辑 ≤150 行）：
   - 搜索：R 用 seed 2024 抽 60 个；L 10 代 × 6、base_seed_L 203400；去掉 L-blind；每个 L 提示词在 PROGRAM_RULES_TEXT 之前接 render_feedback（H 的短训练验证 MAE、展示程序的 feedback_breakdown、上一代失败原因）。第 1 代用专家程序 P 的分解。
   - 正式：H、P、R*、L* × 3 seed × 2 风场。先只用验证集早停并保存 checkpoint，再写 formal_selection_freeze.json 和 test_evaluation_started.json，最后对评估年推理。保存 cohort 和逐臂逐 seed 的预测 npz（SHA256 写进 summary）。
   - 统计同 P3：3 个 seed 的损失平均后用 paired_loss_statistics（块长 8，5000 次，seed 2024）和 rmse_difference_bootstrap；判定仍按逐 seed 均值，阈值不变。
   - 协议：protocol_id 改为 l1_llm_encoder_v4_o2，加 supersedes；summary 的 stage 写 M6。
   - 测试：冻结标记写在评估之前；预测文件能写出；反馈文字进了提示词；≥6/60 碰上限就停在搜索后。
3. **预检后运行一次**：种子 900005，L 真实生成 1 个第 1 代程序；围栏闭合并且 n_new_tokens < 1536 就启动，合法与否只记录。结果写到 output/l1_llm_encoder/o2/，r3 原样保留。
   - 交回：预检输出；H、P 短训练验证 MAE；R、L 的合法数、碰上限数、被拒原因（按代统计）；R*、L* 程序；正式 4 臂 × 2 风场的逐 seed MAE、RMSE、R² 和判定；P3 口径的 CI；top10_programs.json；预测文件清单和 SHA256。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-11：第64次，M6实现与测试完成，待真实预检 -->
轮次: R1/2 | 阶段: M6 | 分支: exp/M2 | 提交号: 本轮实现见git log
审查结论: 通过，见 docs/review/M6-R1-codex.md；Claude反馈模块只使用验证标签，未发现阻断。
已直接修复: 按第64次批准修正新版本审计：24次拟合仅验证早停，checkpoint全部写盘并登记哈希后才开始评估；保存cohort/逐seed预测；P3逐seed损失平均统计。r3原始记录未改。
运行信息: O2=R60/L60、10代×6、base_seed203400、无blind臂；正式H/P/R*/L*×3seed×2场；门槛与数据年份不变，v4_o2。
运行信息: H/P短训先产生真实预检反馈，正式搜索复用该次H/P记录，预检程序不进入候选池；seed900005一次生成，闭合且低于1536才通过。
运行信息: 本地25项19通过6项缺torch跳过；服务器TimeDARTFirst_M6_review运行同两模块25项全部通过，包含合成完整搜索/训练/冻结/推理/数组哈希及P3统计smoke。
关键数字: R60前30与r3相同；反馈只包含展示程序和上一代失败原因；≥6/60触上限禁止进入正式阶段。
运行信息: 尚未执行真实预检或正式运行；先提交冻结实现，再部署新worktree。SSH两次握手超时后恢复；其他用户GPU任务未干预。
需要 Claude 做的: 本轮执行后复核O2结果与审计留存；再基于已有成果讨论加强创新性（二区投稿目标、CCF-B方法目标），当前不新增创新实验、不触Hill of Towie封存年。
需要用户决定: 无。
