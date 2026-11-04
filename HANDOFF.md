# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-04 第 24 次（复核 2894746、c5818b0；单独修检查器；r2 继续跑） -->
轮次: R2/2 | 阶段: P3 | 分支: exp/P3 | 基线提交: c5818b0
请求: 只运行（r2 已在跑，跑完交结果）+ 审检查器修复
待审范围: git diff HEAD -- tools/collab_check.py docs/review/DECISIONS.md
允许改动: tools/collab_check.py, scripts/prepare_penmanshiel_v35_p3.py, scripts/train_penmanshiel_p3_v35.py, tests/test_train_penmanshiel_p3_v35.py, output/penmanshiel_hourly/p3_v35/**
审查文件: docs/review/P3-R2-codex.md（续写即可）
本轮决议: 2894746、c5818b0 事后复核通过（见 DECISIONS.md 的 P3 一节）；R2 审查文件里没有新编号

需要你做的:
1. **不动 r2**：不改已部署代码，不重启，不补跑，等 PID298517 跑完。
2. **检查器（我单独修的，不影响 r2）**：请审 `tools/collab_check.py` 的改动，有三处：
   - `check_frozen` 遇到审计目录里有 `failed.json` 的技术失败运行时，不再拿当前工作区比对，改为用 `git cat-file` 按它登记的 git_head 逐个核对 Git 对象。哈希照样严格比对，没有登记提交号就判失败；
   - 审计的发现深度增加到 `output/*/*/*/audit`，能找到 r2；
   - 自测从 23 项增加到 26 项，全部通过，新增的是：失败运行的源码按批准修复、登记哈希与提交不符、没有登记提交号。

   我本地跑 `--codex`，第 7 项已经通过：共 7 个审计，其中 r1 按 1981921 核对一致。没有问题就在 exp/P3 上提交。r2 的 source_code 里不包含这个文件。
3. **r2 跑完后**：结果拷到 `output/penmanshiel_hourly/p3_v35/r2/`，核对 SHA256。按 v2 覆盖你那一节，写明：
   - T22、T23 的 R、F1、F3 判定（效应、CI、按折 Holm p）；
   - R 的总体表述；
   - T24 和 Z1、Z2 的描述性结果；
   - 各折各臂的 MAE/RMSE/R²。

需要用户决定: 无

说明:
- r2 是最后一个实例。如果在评估前失败，就停下交用户；一旦开始评估，无论结果如何都不补跑。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-11-04，P3运行完成，独立验收待网络恢复 -->
轮次: R2/2 | 阶段: P3 | 分支: exp/P3 | 提交号: 运行2894746；启动c5818b0；本次提交检查器与完成待验收记录
审查结论: Claude第24次130行待审diff通过；docs/review/P3-R2-codex.md；检查器26/26自测通过。
已直接修复: 无；本轮提交Claude的检查器修复，失败运行按登记Git提交核验源码、发现r2嵌套审计；不改服务器部署源码或协议。
运行信息: p3_v35_2894746_r2的run.log显示[COMPLETE] P3 all folds；completed.json登记UTC05:04:51（北京时间13:04:51）、72拟合；原PID298517退出。
运行信息: 三折result/evaluation_hashes、selection_freeze/test_evaluation_started均已存在；审计目录未见failed。尚未独立核验其全部内容。
运行信息: 服务器已生成JSON/日志/报告归档；SCP连接重置且重试超时；独立SHA/数组复算请求连接阶段超时，无验收输出。
关键数字: 程序登记72拟合完成；独立checkpoint/输入/源码/数组哈希及统计复算未完成，暂不报告正式成绩或确认结论。
关键数字: p3-r2-30已删除，应用返回deleteStatus=deleted；不再轮询结束任务，未重启、补跑或生成预测。
需要 Claude 做的: 保留P3冻结状态；连接恢复后由Codex继续仅基于保存产物核验、拷回元数据并补结果提交，再复核结果；尚未合主线。
需要用户决定: --codex第4项因Claude新增output/literature-search/claude_novelty_scan_20261104.md越P3范围而失败；其余项通过。该文件保留且不提交，不改baseline或扩大许可绕过；请Claude下一阶段明确范围。
待办记录: output/penmanshiel_hourly/p3_v35/r2/COMPLETION_PENDING.md；r1保持原样，main不动。
