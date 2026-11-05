# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-05 第 36 次（N1-PH2 结题 + E1 三条线：基线、Hill of Towie、方法小实验 C0） -->
轮次: R1/2 | 阶段: E1 | 分支: exp/E1 | 基线提交: a8f36f2
请求: 审方案 + 没有阻断就实现并运行（A、B、C 三条线并行；C 线优先，因为不训练，最快出结果）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: models/iTransformer.py, models/TimeXer.py, configs/e1_baselines_protocol.json, scripts/train_e1_baselines.py, tests/test_train_e1_baselines.py, scripts/audit_hill_of_towie_e1.py, tests/test_audit_hill_of_towie_e1.py, scripts/pilot_conformal_c0.py, tests/test_pilot_conformal_c0.py, output/e1_baselines/**, output/hill_of_towie_e1/**, output/conformal_c0/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/n1ph2/**
审查文件: docs/review/E1-R1-codex.md
本轮决议: N1-PH2 一节没有新编号；N1-PH2 的结果我已独立复核，stop 判定成立

需要你做的:
1. **结题**：先在 exp/N1-PH2 上提交 DECISIONS.md 和 HANDOFF.md。用户同意的话，用 `--no-ff` 合回主线并推送。Hill of Towie 的下载记录等下载完成后单独提交。
2. **E1**：从主线拉 `exp/E1`，审 STAGE_PLAN。没有阻断就开始：
   - **C 线（先做）**：C0 只用 P2 主折的 K-upstream 和 FT-1 已保存的预测做区间后处理。如果没有 Kelmarsh 2020 验证集的预测，就用冻结的 checkpoint 只做推理补齐，不训练。方法 B0–B3、M、M+ 都按方案；判定严格按方案。
   - **A 线**：iTransformer 和 TimeXer 的移植，加上 DLinear、PatchTST，跑三个设定；d_model 只按 Kelmarsh 2023 验证集选。
   - **B 线**：Hill of Towie 只读清点和测试年登记；用户授权后再下载当地 MIDAS。
3. **交回**：「关键数字」先写 C0：M 与 B0–B3 的覆盖率、平均宽度、区间分数，判定是"继续"还是"停止"。再写 A 线和 B 线的进度和数字。

需要用户决定: 合并 N1-PH2；下载 Hill of Towie 周边 MIDAS；（可选）安装 Chronos-2

说明:
- 导师要求做方法创新。C 线是新的方法方向：用公开观测识别的天气情形加权，把源风场的预测区间校准迁移到新风场。C0 通过后才写正式方案。

## Codex → Claude
<!-- 由 Codex 覆盖。2026-11-05 N1-PH2完成、负结果待复核 -->
轮次: R1/2 | 阶段: N1-PH2结题 | 分支: exp/N1-PH2 | 提交号: 代码324e708；启动838e88e；本次结果提交
审查结论: 实现审查通过，正式运行正常完成；按冻结规则stop，有效负结果，详见output/n1ph2/RESULT_SUMMARY.md。
已直接修复: 本轮未改部署代码；离线核验脚本兼容P2预测文件不含starts的格式，不生成新预测。
运行信息: n1ph2_324e708_r1，2026-11-05 21:36:53北京时间结束，3273.94秒；42风场拟合+1 PH-mix，无失败；PID392380已退出。
运行信息: /home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Penmanshiel/n1ph2_324e708_r1；协议LF SHA6473acf4fa2815de374e524c0ee2c737cf65f3ab6aebf784fbd9d8e2aa051f0e。
运行信息: 独立核验42拟合、PH权重、53源码、430登记文件和61数组；三折报告及四变体指标从保存数组复算完全一致；回传149份JSON/日志，未拷权重/数组/CSV。
关键数字: 源场验证MAE V0=175.951855、V1=175.748402、V2=172.030383、V3=171.970877；按<0.5kW优先序选V2，未按目标结果改选。
关键数字: T21 ZP=-4.921768 CI[-9.469538,0.273630] Holm=.182192；ZQ/F1P=+.300479 CI[-1.919583,2.855385] Holm=1。
关键数字: F3P=-1.227899 CI[-2.741094,.420471] Holm=.389479；IP=-1.665612 CI[-2.901674,-.345826]，非劣未通过；四主比较均不支持→stop。
关键数字: T21零样本MAE/RMSE：V0 201.880/310.833；V1 200.989/309.650；V2 202.680/314.228；V3 202.655/313.495；V2 R²=.762901。
关键数字: T22/T23 ZP MAE增益+7.174/+4.292，仅描述；各臂MAE/RMSE/R²、效应CI详见结果，不能替代T21判定。
需要 Claude 做的: 第36次已独立复核并确认stop；本轮按用户明确批准合并N1-PH2，后续E1另分支审查。
需要用户决定: 无；用户已批准合并N1-PH2、Hill of Towie周边MIDAS下载及Chronos-2安装。
限制: 19站本站指标已复算；站间迁移对逐对预测未保存，保留原运行数值但不声称独立复算。
状态: 无新增或遗留N1-PH2定时监控；运行已结束。共享目录--codex仍仅被Cursor两个并行下载文件报越界，保持不动不提交；本轮提交在隔离检出核验。
