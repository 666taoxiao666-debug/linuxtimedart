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
<!-- 由Codex覆盖。第85次G2c：c0/c1完成，交回复核 -->
轮次: R1/2 | 阶段: G2c | 分支: exp/G2 | 提交号: 实现2eeb23f；结果见本提交
审查结论: 通过，docs/review/G2c-R1-codex.md；S1措辞建议不阻断。
已直接修复: 无额外改动；G2c-C1追加入口，已有11函数AST一致，冻结源码及协议未改。
运行信息: CPU2线程；c0副本+2023演练包PASS；c1正式一次COMPLETED，PID1027531，2026-11-15T14:13:02Z启动，无新增拟合/重试。
服务器run: /home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/G2/g2c_2eeb23f_r1/formal
命令: scripts/evaluate_g2b_sealed.py --mode evaluate；完整参数见output/g2b/sealed_eval/server/formal_launch.json。
验证: 服务器22项21通过1跳过；冻结checkpoint/source、28预测、两数组及results SHA通过；本地验收见local_verification.json。
关键数字: 2024/2025 Q1h增益14.9052/15.0538kW，CI[12.1928,17.5953]/[12.3972,17.7346]，p=8.4276e-27/2.4244e-29。
关键数字: 2024/2025 Qe增益3.2810/2.9307kW，CI[3.0935,3.4761]/[2.7600,3.1031]，p=1.0392e-246/6.0550e-248。
关键数字: 两年最大p再Holm：Q1h=8.4276e-27、Qe=2.0784e-246，均通过联合确认。
关键数字: 集成MAE/RMSE/R²：2024=197.275/326.035/0.764740；2025=190.102/316.959/0.770600。
关键数字: 次要N对N0增益0.9079/2.9931kW，2024 CI跨0、p=0.2158；不可声称两年均显著。
结果: output/g2b/sealed_eval/SUMMARY.md及server/；arrays/两年19键小数组仅本地、忽略不提交。
范围: 未读2026、不训练/调参/事后挑子组；两年共享模型不称独立，跨风场汇总仅描述；不合主线。
需要 Claude 做的: 独立复算本地数组、复核全部原定检验及结论边界；处理S1措辞，复核后交用户决定是否合并。
需要用户决定: 无；本轮完成后停下交回。
