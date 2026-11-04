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
<!-- 由 Codex 覆盖。最后更新：2026-11-04，P3 r2结果回传及独立验收完成 -->
轮次: R2/2 | 阶段: P3 | 分支: exp/P3 | 提交号: 运行2894746；启动c5818b0；检查器4682704；本次为r2结果提交
审查结论: 通过；原代码审查见docs/review/P3-R2-codex.md；本次只读验收退出码0，详见output/penmanshiel_hourly/p3_v35/r2/INDEPENDENT_VERIFICATION.md。
已直接修复: 无；未修改部署源码、冻结协议或预测；临时验收脚本适配旧Python的分块SHA256读取，不属于实验改动。
运行信息: 原命令见r2/audit/launch.json；run=outputs/logs/Penmanshiel/p3_v35_2894746_r2；UTC05:04:51完成72拟合，耗时6116.33秒；没有重启、补跑或第三实例。
运行信息: 核验72个唯一拟合及选中模型、50份源码、414个输入、6个准备产物、75个数组哈希；selection冻结标记一致。
运行信息: 仅以保存数组复算三折全部指标/CI/Holm/判定，rtol=atol=1e-10一致；239份元数据大小/SHA一致，已拷至output/penmanshiel_hourly/p3_v35/r2/；CSV/权重/数组未回传。
关键数字: T22 R/F1/F3 MAE增益=15.952[12.729,18.939]/86.205[79.548,92.208]/48.709[45.077,52.190]kW；Holm p=1.93e-23/5.58e-159/8.15e-158，均通过。
关键数字: T23 R/F1/F3 MAE增益=15.011[12.001,18.253]/87.313[81.359,93.288]/42.864[38.926,47.003]kW；Holm p=3.45e-21/7.81e-180/1.45e-100，均通过；六项RMSE增益CI下界均>0。
关键数字: R在2022/2023两个时间隔离测试年确认，滚动训练集重叠，不宣称统计独立；F1/F3是完整迁移配方对对应Scratch的收益，不外推普适零样本收益。
关键数字: P-upstream三折MAE/RMSE/R²=213.631/329.317/.753407；188.839/301.960/.772537；198.866/310.791/.788672。T24只描述；全部臂及全部效应CI见r2/RESULT_SUMMARY.md。
关键数字: Z2三折MAE增益=+2.537/-2.321/-3.996kW，CI均跨0；T24及所有Z1/Z2只描述。p3-r2-30上一轮已删除，不保留结束任务轮询。
需要 Claude 做的: 复核r2/RESULT_SUMMARY.md、INDEPENDENT_VERIFICATION.md及原始JSON，确认结果措辞与负结果边界；复核后等用户同意再合主线。
需要用户决定: --codex第4项仍将两份阶段外文档列为越界：用户单独要求的docs/GPT_PAPER_INNOVATION_DISCUSSION.md，以及Claude的output/literature-search/claude_novelty_scan_20261104.md；本轮均保留不提交，不改baseline/许可绕过，其余检查结果见本次提交正文。
状态: 已解除此前网络阻塞；COMPLETION_PENDING.md标记为历史记录。r1保持原样，main不动。
