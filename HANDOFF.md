# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-09 第 55 次（C0b STOP 已复核；新阶段 M2：C 线第 1 轮 CW1 + T1 并行，L1 等权重。取代未执行的第 54 次） -->
轮次: R1/2 | 阶段: M2 | 分支: exp/M2 | 基线提交: a1ebfa0
请求: 审方案后实现并运行（CW1、T1 都审；无阻断就各正式运行一次，二者并行，提速规则适用）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/pilot_conformal_cw1.py, tests/test_pilot_conformal_cw1.py, output/conformal_cw1/**, scripts/tune_t1.py, configs/t1_tuning_protocol.json, tests/test_tune_t1.py, output/t1_tuning/**, scripts/llm_encoder_search_l1.py, models/ProgramEncoder.py, tests/test_llm_encoder_search_l1.py, configs/l1_llm_encoder_protocol.json, output/l1_llm_encoder/**, output/external_acquisition/llm_weights/**, output/conformal_c0b/**, output/external_acquisition/gfs_g0/**, output/hill_of_towie_e1/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/M2-R1-codex.md
本轮决议: C0b STOP 已独立复核（DECISIONS 的 M1 末节）；按用户要求 C 线续做最多 3 轮；C2、C3 的门槛结构在运行前调整并登记（见 DECISIONS 的 M2 一节），新意门槛没有降低

需要你做的:
1. **提交与分支**：在 exp/M1 上提交本轮的 STAGE_PLAN 和 DECISIONS。用户同意后，把 exp/M1 用 --no-ff 合回主线，再从主线拉 exp/M2。
2. **审 M2 方案**（CW1 + T1），有阻断按 v2 写审查文件。重点看：
   - CW1：反向 J2 的数据身份和因果截止（Penmanshiel 2023 校准 → Kelmarsh 2024）；W1–W4 的实现是否和 B3-A 同处理；W4 的第二个对照；挑选规则；selection_freeze 必须先于判定数据；CI 的算法。
   - T1：共同窗口及退回 24 h 的条件；三个模型预算相等；tuning_freeze 必须先于评估年。
   - 没有阻断就各正式运行一次。CW1 用 CPU、2 个线程，T1 用 GPU，二者并行。服务器上没有 Optuna 就停下 T1，交回"需 Cursor 安装"，CW1 照常运行。
3. **交回**：
   - CW1：挑选年各变体的覆盖率和 Winkler；所选变体；J1、J2 上所有方法的覆盖率、宽度、Winkler、截顶占比，以及 C1–C3 的判定和 CI。
   - T1：最优配置、试验记录，两个开发年上集成对集成的对比和判定。
   - L1 仍然等权重。

需要用户决定: 是否同意把 exp/M1 合回主线

## Codex → Claude
<!-- 由Codex覆盖。2026-11-09 第53次：M1 R2通过，C0b单次完成并验收STOP -->
轮次: R2/2 | 阶段: M1 | 分支: exp/M1 | 提交号: 实现35e8e12；本轮结果见git log
审查结论: 通过，无遗留阻断；docs/review/M1-R2-codex.md；R1-B1及S1–S6已核对本轮改动。
已直接修复: 无额外设计变更；按用户及第53次批准实现scripts/pilot_conformal_c0b.py和tests/test_pilot_conformal_c0b.py；冻结C0未改。
运行信息: 本地/服务器各7测试通过，服务器--smoke通过；正式一次PID660408已完成退出，169.806秒，无重启/补跑。
运行信息: /home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Conformal/c0b_35e8e12_r1；python scripts/pilot_conformal_c0b.py --data-root /home/XT/nuist/pythoncode/TimeDARTFirst --out <该run目录>；CPU线程2。
运行信息: 2022选A，selection_freeze SHA256 def24a5d386820514ec845464ff8b922621a7854af47cff20a7b065aa5f5f307早于2023计算；评估只运行选中臂及规定对照。
运行信息: 只读验收30数组/286输入/54源码哈希；保存数组重算指标一致，2022的9预测包与旧C0逐元素一致；本地output/conformal_c0b/{RESULT_SUMMARY.md,verification.json,r1/}。
关键数字: 2023三seed均值，M-A覆盖89.5002%、宽854.9699、Winkler1280.1556、cap0.007465%；B3-A覆盖89.4561%、宽855.2339、Winkler1283.9894、cap0%。
关键数字: B0/B1/B2-A覆盖82.8470%/94.4837%/89.3204%，Winkler1636.8964/1610.8071/1497.0523；均cap0%，各h详见seed JSON。
关键数字: C1=true，C2=false，C3=false，STOP；相对匹配B3改善0.298583%，不足5%/3%门槛；有效负结果，C线不再优化。
关键数字: 每seed 2022/2023评估窗口37741/36838；两校准截止2021/2022年末23:00；无点模型训练，不解封Hill of Towie。
需要 Claude 做的: 复核35e8e12实现、验收和STOP结论，在DECISIONS登记C0b有效负结果；L1继续等Cursor模型revision/环境，未实现/运行。完整摘要见RESULT_SUMMARY.md。
需要用户决定: 无；只提交推送exp/M1，不合主线，main不动；未创建定时任务。
