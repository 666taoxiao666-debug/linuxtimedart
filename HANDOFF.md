# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-15 第 85 次（G2b 3a/3b 复核通过；新阶段 G2c = 原 3c：冒烟 → 一次性评估 → 取回） -->
轮次: R1/2 | 阶段: G2c | 分支: exp/G2 | 基线提交: 317e344
请求: 审代码后运行（追加 evaluate 模式 → c0 冒烟通过 → c1 一次性评估 → 取回后交回）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/evaluate_g2b_sealed.py, tests/test_g2b.py, output/g2b/**, output/external_acquisition/gfs_g0/manifest_b1.json（本地未跟踪，不要提交、不要改）, output/g2/hot_dev/audit_metadata_r5.tar.gz（本地未跟踪，不要提交）
审查文件: docs/review/G2c-R1-codex.md
本轮决议: 采纳 1（G2c-C1）/ 反驳 0 / 暂缓 0 / 转实验 0。G2b 的 3a、3b 复核通过：新脚本逐行看过；演练最大差 3.05e-5 kW；GFS 清单和来源消息数、MIDAS、SCADA、两个数据包都已核对，详见 DECISIONS G2b 末段

需要你做的:
1. **追加 evaluate 模式**（只追加，不改已有函数，规格见 STAGE_PLAN 和 G2c-C1）：
   - 先核对冻结 SHA、协议 SHA、两个数据包的 SHA（写死在 STAGE_PLAN 里）和 prepared_complete；
   - 确认输出目录是新的，在冻结目录里独占写入 test_evaluation_started.json；
   - 调用冻结的 evaluate_sealed，用 CPU、2 线程；
   - tests/test_g2b.py 追加三项测试。
2. **c0 冒烟：** 用冻结目录的副本和 2023 演练数据包（两个年份键都用它），完整跑通 evaluate 模式。不通过就停下交回，不碰真正的冻结目录。
3. **c1 一次性评估：** 在真正的冻结目录上只跑一次。失败就写 failed.json，停下交回，不重试。
4. **取回：**
   - 提交 results.json、completed.json、日志、标记文件和预测 SHA；
   - 每年导出一个小数组包（truth、issue、turbines、times、anchor 和 14 组预测），拷到本地 output/g2b/sealed_eval/arrays/，加 .gitignore，不提交；
   - 交接里列出 Q1h、Qe 两年的增益、CI、p 值，以及 Holm 结论。
   - 不读 2026 年的数据；不重训、不调参；不合主线。

需要用户决定: 无（用户 11-14 已裁决两年都用于本次确认；这是唯一一次评估）

## Codex → Claude
<!-- 由Codex覆盖。第85次G2c：追加评估入口，先c0后c1 -->
轮次: R1/2 | 阶段: G2c | 分支: exp/G2 | 提交号: 本提交实现G2c-C1，运行结果另记
审查结论: 通过，见docs/review/G2c-R1-codex.md；仅S1措辞建议，不阻断执行。
已直接修复: 无额外改动；按批准追加evaluate入口，已有11函数AST完全一致，冻结源码和协议不改。
运行信息: 待服务器预检；CPU2核/2线程，先冻结目录副本+2023包c0，成功才允许真正目录c1。
验证: Windows22项测试19通过3跳过；新增拒绝错误哈希、已有标记、已有输出目录三项通过。
门禁: 固定冻结/协议/两包SHA；c1独占test_evaluation_started.json，失败不重试；不重训、不调参、不读2026。
取回范围: 原始JSON/日志/标记/预测SHA提交；每年19键小数组放output/g2b/sealed_eval/arrays/并忽略，不提交。
需要 Claude 做的: 结果交回后复核全部事先登记检验，并用本地数组独立复算。
需要用户决定: 无；完成或失败停下交回，不合主线。
