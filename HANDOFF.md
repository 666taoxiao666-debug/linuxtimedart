# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-16 第 86 次（G2c 复核通过，用户同意合并 exp/G2；新阶段 X1：外部基线 + 时序大模型并行） -->
轮次: R1/2 | 阶段: X1 | 分支: exp/X1 | 基线提交: 3254c75
请求: 审方案后运行（先合并 G2，再从主线拉 exp/X1；A0–A2 先跑，A1 等 lightgbm，A3/B 等 Cursor 装好 Chronos-2）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/run_x1.py, configs/x1_protocol.json, tests/test_x1.py, output/x1/**, output/external_acquisition/gfs_g0/manifest_b1.json（本地未跟踪，不要提交、不要改）, output/g2/hot_dev/audit_metadata_r5.tar.gz（本地未跟踪，不要提交）
审查文件: docs/review/X1-R1-codex.md
本轮决议: 采纳 1（G2c R1-S1：效应以改造后工况为前提）/ 反驳 0 / 暂缓 0 / 转实验 0。G2c 我用本地数组独立复算：两年的 MAE、RMSE、R²、5 组增益与 results.json 一致到小数点后 4 位，NW z 一致到 3 位；Q1h、Qe 两年都通过。结论边界见 DECISIONS G2c 末段

需要你做的:
1. **合并（用户 2026-11-16 同意）：** 先提交本轮文档；exp/G2 用 --no-ff 合回 codex/self-evolving-wiki 并推送；再从主线拉 exp/X1。在本地 G1 克隆里切到 exp/X1。
2. **审 X1 方案**（STAGE_PLAN 全文）。确定缺陷按唯一的最小修法先做；设计分歧写成 B 或 S 编号交给我。
3. **A 线：**
   - 写 scripts/run_x1.py 和 configs/x1_protocol.json（锁 SHA），每次编辑不超过 150 行。
   - 先跑 A0、A2；lightgbm 能用就跑 A1。
   - 统计和判定按 STAGE_PLAN；N 的预测取存盘文件并核对 SHA。
4. **A3 和 B 线：** 等 Cursor 装好 Chronos-2 后核验（版本 revision、文件 SHA、在 GPU 上的冒烟），再跑 A3a、A3b（可选）、B1、B2。B2 的判定只看开发验证段。
   - 不下载、不安装；缺依赖就停下等 Cursor。
   - 不改 N 和已冻结的东西，不读 2026 年的数据。

需要用户决定: 无（Cursor 下载指令由我直接给用户）

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
