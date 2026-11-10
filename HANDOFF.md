# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-09 第 59 次（M4 R2：R1-B1 把 J3、J4 降级为部分暴露的探索性判定，附暴露清单；S1、S2 写清） -->
轮次: R2/2 | 阶段: M4 | 分支: exp/M2 | 基线提交: ea5252d
请求: 审方案后实现并运行（只审改动；无阻断就正式运行 CW3 一次，CPU、2 个线程；不要打断 T1）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/pilot_conformal_cw3.py, tests/test_pilot_conformal_cw3.py, output/conformal_cw3/**, output/conformal_cw2/**, output/conformal_cw1/**, scripts/tune_t1.py, configs/t1_tuning_protocol.json, tests/test_tune_t1.py, output/t1_tuning/**, scripts/llm_encoder_search_l1.py, models/ProgramEncoder.py, tests/test_llm_encoder_search_l1.py, configs/l1_llm_encoder_protocol.json, output/l1_llm_encoder/**, output/external_acquisition/llm_weights/**, output/conformal_c0b/**, output/hill_of_towie_e1/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/M4-R2-codex.md
本轮决议: 采纳 3（R1-B1、S1、S2）/ 反驳 0 / 暂缓 0 / 转实验 0

需要你做的:
1. **提交**本轮的 STAGE_PLAN 和 DECISIONS（仍在 exp/M2 上，T1 在运行）。
2. **只审改动部分**，这是第 2 轮，也是最后一轮；仍有分歧就写明，交用户决定：
   - J3、J4 改名为"部分暴露的探索性判定"，暴露清单是否完整、准确；
   - 方法主张只能等 Hill of Towie 联合确认；
   - 结论档位：方法档要求两组都 ≥3%；只要不是两组都到 3%、但两组 CI 下界都 > 0，就归"小幅但显著"。
   - 年份、阈值、臂都不变。
3. **无阻断就实现 CW3 并正式运行一次**。结果文件里写明"探索性判定"；交回内容同第 58 次。T1 跑完就一并交回，没跑完就写进度。
4. **L1 的权重已到**（output/external_acquisition/llm_weights/manifest.json，Cursor 2026-11-09 交回；STAGE_PLAN 已同步）：
   - 现在只读核对：49 个文件的 SHA256、revision 和 llm_l1 环境，然后把 manifest 一并提交。
   - T1 跑完后，再按 `git show a1ebfa0:docs/STAGE_PLAN.md` 的 L1 规则实现并运行（规则已在 M1 R2 审定，提速规则适用），结果记在 DECISIONS 的 M1 一节。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-10：按用户要求只读核验并汇总服务器已运行任务 -->
轮次: M2结果验收 | 阶段: T1完成；M4 R2待审 | 分支: exp/M2 | 提交号: 本轮见git log；运行a99037f
审查结论: T1只读验收通过，决定RETAIN_EXISTING；CW3第59次R2尚未复审或实施，本轮不消耗新运行预算。
已直接修复: 无；新增验收脚本和结果记录，不修改冻结源码、协议或数据。
运行信息: T1目录/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Tuning/t1_a99037f_r1；2026-11-09 21:28:32北京时间完成，24096.86秒。
运行信息: PID674567已退出；真实相关Python命令扫描未见T1/CW1/CW2/CW3/L1存活实例。completed=COMPLETED、failed不存在。
运行信息: verify_saved.py经SSH标准输入只读执行；218源码、260输入、1200拟合产物、74评估数组SHA256核对通过；冻结链接及时间顺序一致。
运行信息: 逐种子指标及五种子集成从保存数组重算一致，未训练、未生成预测；取回737个JSON/日志，不取原始数据、权重或数组。
关键数字: 180/180搜索COMPLETE，60/60正式拟合；result SHA256=bd33ff0596bff1ff55f8e8f825f2d830e181093a50a339901cd2c7804786261d。
关键数字: Kelmarsh2022 TSMixer原/调参集成MAE=158.0902/157.1188，RMSE=239.9439/242.3934，R²=0.830145/0.826659。
关键数字: Penmanshiel2021 TSMixer原/调参集成MAE=178.8555/177.5431，RMSE=294.5795/298.0361，R²=0.792284/0.787380。
关键数字: 两折MAE仅下降0.6145%/0.7338%，RMSE增加2.4495/3.4566kW；均未过“MAE下降≥2%且RMSE不增”，保留原配置。
关键数字: 三模型各60次搜索，TSMixer/iTransformer/TimeXer所选trial59/48/38；全部比较和参数见r1/tuning_freeze.json及RESULT_SUMMARY.md。
证据: output/t1_tuning/{RESULT_SUMMARY.md,verification.json,metadata_sha256.json,verify_saved.py,r1/}；本地原始元数据与远端字节哈希一致，Git换行按LF核对。
范围限制: 回顾性跨风场开发，不作2021前瞻确认；五种子集成与历史单种子/不同样本队列不可直接对比；未宣称统计显著性。
其他线: CW1已完成STOP_CW1；CW2已完成STOP_AT_SELECTION。CW3未启动；L1仅权重交付清单到达，本轮未核验/运行。
需要 Claude 做的: 复核T1的保存数组验收、完整两折三模型结果及RETAIN_EXISTING结论，在DECISIONS M2登记；再按第59次交接推进M4 R2与L1。
需要用户决定: 无；本轮未合主线，仍需Claude复核及用户同意。Claude的STAGE_PLAN/DECISIONS未提交改动保留，不冒充本轮已审通过。
