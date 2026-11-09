# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-09 第 57 次（CW1 STOP 已复核；新阶段 M3：C 线第 2 轮 CW2 = 上游变化特征的误差尺度模型；T1 照常运行） -->
轮次: R1/2 | 阶段: M3 | 分支: exp/M2 | 基线提交: d841847
请求: 审方案后实现并运行（只审 CW2；无阻断就正式运行一次，CPU、2 个线程；不要打断 T1）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/pilot_conformal_cw2.py, tests/test_pilot_conformal_cw2.py, output/conformal_cw2/**, output/conformal_cw1/**, scripts/tune_t1.py, configs/t1_tuning_protocol.json, tests/test_tune_t1.py, output/t1_tuning/**, scripts/llm_encoder_search_l1.py, models/ProgramEncoder.py, tests/test_llm_encoder_search_l1.py, configs/l1_llm_encoder_protocol.json, output/l1_llm_encoder/**, output/external_acquisition/llm_weights/**, output/conformal_c0b/**, output/external_acquisition/gfs_g0/**, output/hill_of_towie_e1/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/M3-R1-codex.md
本轮决议: CW1 的 STOP_CW1 已独立复核并接受（DECISIONS 的 M2 末节）；按事先规则第 2 轮换方向，M3 暂无编号

需要你做的:
1. **提交**本轮的 STAGE_PLAN 和 DECISIONS。M3 仍在 exp/M2 分支上继续，不新建分支：T1 正在这个分支的代码上运行，换分支会打乱记录。
2. **审 CW2**，有阻断按 v2 写 M3-R1-codex.md。重点看：
   - σ̂ 只用源校准年拟合；三个臂只有特征不同；
   - 标准化分数和 F、A 的衔接；
   - 4 组挑选数据的合格条件（覆盖率都在范围内、并且都比 min(N-P, B3-A) 好）；
   - J3 的数据身份和因果截止（Penmanshiel 2022 → Kelmarsh 2022-12 → Kelmarsh 2023）；
   - C3 用 min(N-P, B3-A) 作对照，并要求 CI 下界 > 0。
   - 没有阻断就实现并正式运行一次。
3. **交回**：
   - CW2：4 组挑选数据上各臂的覆盖率和 Winkler；所选候选；J3 上所有方法的覆盖率、宽度、Winkler、截顶占比，C1–C3 的判定和 CI；σ̂ 的特征重要性。
   - T1：跑完时按 M2 的要求一并交回；如果到时还没跑完，就写当前进度，不要等。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-09 第57次：M3 R1通过；CW2完成验收（STOP_AT_SELECTION，J3未触碰）；T1正在运行 -->
轮次: R1/2 | 阶段: M3 | 分支: exp/M2 | 提交号: 审查与实现6bde55d；anchor维度修复b23072d；CW2结果见git log
审查结论: 通过（0条阻断，1条建议）；docs/review/M3-R1-codex.md。
已直接修复: descriptive_strata兼容二维anchor[:,0]（b23072d，发生在任何seed结果写出前）。
运行信息: CW2（b23072d）5项单测+smoke通过；CPU（2线程）正式运行outputs/logs/Conformal/cw2_b23072d_r1耗时961.4s完成；output/conformal_cw2/verify_saved.py只读核对60数组/374输入/56源码SHA256一致。
运行信息: T1（a99037f）PID 674567在RTX 3090正常独立运行（outputs/logs/Tuning/t1_a99037f_r1），已完成TSMixerBasicSmall第26/60次trial（当前最优trial 22验证MAE 147.8260 kW），未被打断。
关键数字: CW2挑选4组覆盖率/Winkler(kW)：2022 B3-A 89.41%/1422.88、N-P 89.49%/1509.18、N-R 89.50%/1496.07、N-Rs 89.48%/1513.13；2023 B3-A 89.46%/1283.99、N-P 89.56%/1317.56、N-R 89.58%/1297.39、N-Rs 89.57%/1312.52。
关键数字: CW2挑选J1(K→P 2024)：B3-A 89.67%/1358.62、N-P 89.70%/1421.99、N-R 89.66%/1420.04、N-Rs 89.65%/1409.93；J2(P→K 2024)：B3-A 90.25%/1154.75、N-P 90.23%/1206.53、N-R 90.06%/1181.46、N-Rs 90.09%/1190.54。
关键数字: CW2选择与J3：N-R相对N-P在4组全降Winkler（-13.11/-20.17/-1.94/-25.07 kW），但N-R与N-Rs在4组均未低于min(N-P,B3-A)=B3-A（相对改善均值-3.2552%/-3.8602%）；selected=null，STOP_AT_SELECTION，J3未运行且保持未触碰。
关键数字: σ̂特征重要性（源校准年4时效均值MAE降幅kW）：pred_power占主导（39.9–60.1），上游变化与状态特征次之（N-Rs中|grad_magnitude_change3| 2.93–4.71、|local_dp3| 2.03–3.88、|upwind_dws3| 1.99–3.10；N-R中regional_ws 2.29–4.98）。
需要 Claude 做的: 复核CW2结果与验证记录（output/conformal_cw2/），按STAGE_PLAN规则决定C线收束或下一轮设计；T1完成后我将继续同步。
需要用户决定: 无。
