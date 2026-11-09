# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-09 第 58 次（l0b_r2 收尾复查通过；CW2 STOP_AT_SELECTION 已复核；新阶段 M4：C 线第 3 轮 CW3） -->
轮次: R1/2 | 阶段: M4 | 分支: exp/M2 | 基线提交: bd955f4
请求: 审方案后实现并运行（只审 CW3；无阻断就正式运行一次，CPU、2 个线程；不要打断 T1）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/pilot_conformal_cw3.py, tests/test_pilot_conformal_cw3.py, output/conformal_cw3/**, output/conformal_cw2/**, output/conformal_cw1/**, scripts/tune_t1.py, configs/t1_tuning_protocol.json, tests/test_tune_t1.py, output/t1_tuning/**, scripts/llm_encoder_search_l1.py, models/ProgramEncoder.py, tests/test_llm_encoder_search_l1.py, configs/l1_llm_encoder_protocol.json, output/l1_llm_encoder/**, output/external_acquisition/llm_weights/**, output/conformal_c0b/**, output/hill_of_towie_e1/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/M4-R1-codex.md
本轮决议: l0b_r2 收尾复查通过，原验收不变；CW2 的 STOP_AT_SELECTION 已独立复核，J3 未触碰；M4 暂无编号（复核记录都在 DECISIONS 的 M3 一节末尾）

需要你做的:
1. **提交**本轮的 STAGE_PLAN 和 DECISIONS。仍在 exp/M2 上，T1 在运行，不要换分支。
2. **审 CW3**，有阻断按 v2 写 M4-R1-codex.md。重点看：
   - LN-R、LJ-R 和它们的匹配对照 LN-P、LJ-P 是否只差 σ̂ 这一项；
   - 联合近邻的距离缩放只用源校准集；
   - 4 组挑选数据的合格条件；
   - J3、J4 的数据身份和因果截止；
   - 三档结论（包括"小幅但显著"）的判定逻辑。
   - 没有阻断就实现并正式运行一次。
3. **交回**：
   - CW3：挑选数据上各臂的覆盖率和 Winkler；所选候选；J3、J4 上所有方法的覆盖率、宽度、Winkler、截顶占比，C1–C3 的判定和 CI，以及最终落在哪一档；
   - T1：跑完就一并交回，没跑完就写当前进度。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-09：第58次交接，CW3方案R1审查 -->
轮次: R1/2 | 阶段: M4/CW3 | 分支: exp/M2 | 提交号: 本轮见git log（基线bd955f4）
审查结论: 1条阻断R1-B1、2条建议；docs/review/M4-R1-codex.md。CW3未实现、未部署、未启动，正式次数未消耗。
已直接修复: 无；涉及判定数据身份和结论权限，有多种修订路线，不适用唯一最小修法提速例外。
运行信息: 只读核对CW2已有校准审计与冻结Data/scale/selection调用链；未读封存原始数据，未重跑CW2。
关键数字: CW2 selection_2023的K2022源校准窗口16949；J1的K2023源校准窗口16716，两组result.json均已产生。
关键数字: 上述K2022/K2023标签已参与残差/尺度拟合并影响挑选，当前方案又将其用作J4/J3，不能称“区间方法从未用过”的新判定数据。
审查边界: 单折时间截止可正确；阻断针对跨阶段复用及独立确认，不据此否定既有点预测结果，也不声称每折存在直接未来标签输入。
通过项: 匹配臂仅替换尺度特征、距离标准差只用源集、四组资格/等权选择/平局规则、J3/J4单折时间边界。
建议: S1把CW2机制归因写成假设；S2明确4%/2%混合改善时“小幅但显著”档的真值表。
T1运行信息: SSH只读成功；PID674567，状态Rsl、elapsed02:21:55；真实命令为timedart/bin/python scripts/tune_t1.py --data-root /home/XT/nuist/pythoncode/TimeDARTFirst --out /home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Tuning/t1_a99037f_r1 --device cuda:0。
T1关键数字: 服务器日志2026-11-09 17:08:16显示iTransformer trial_36完成（该模型0起算37次），trial_37训练中；最佳trial_32验证MAE173.2604098，仅搜索验证值，非最终测试结果。
T1状态: 本次run目录仅launch/input_identity/cohorts/search，未见根目录完成标记；未完成，不中断、不重启、不改源码。
需要 Claude 做的: 在DECISIONS逐条决议R1-B1，补按风场/年份/用途的暴露清单；明确保留年份的探索性口径与后续权限，或另设隔离方案，更新STAGE_PLAN后交R2只审改动。
需要 Claude 做的: 同时明确S2混合情形；原阈值、臂数、年份未经新批准不自行改变。CW3本轮无运行结果。
需要用户决定: 无，尚在两轮内；等Claude R2修订。T1按原批准继续。
