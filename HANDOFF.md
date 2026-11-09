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
<!-- 由Codex覆盖。2026-11-09 第56次：M2 R2通过；CW1已完成并验收（STOP_CW1）；T1已在服务器GPU正式运行 -->
轮次: R2/2 | 阶段: M2 | 分支: exp/M2 | 提交号: CW1实现0b3eaf9；T1实现a99037f；CW1结果与交接见git log
审查结论: 通过（0条阻断）；R1-B1与S1–S5全部按要求修订；docs/review/M2-R2-codex.md。
已直接修复: 无。
运行信息: CW1（0b3eaf9）8项单测+smoke通过；正式运行outputs/logs/Conformal/cw1_0b3eaf9_r1耗时424.5s完成，verify_saved.py核对74数组/374输入/55源码SHA256及CI一致（output/conformal_cw1/）。
运行信息: T1（a99037f）核对Optuna 5.0.0（existing_packages_changed=false），4项单测+smoke通过；已在RTX 3090独立启动唯一实例PID 674567（outputs/logs/Tuning/t1_a99037f_r1），正运行180次搜索+60次正式拟合。
关键数字: CW1挑选（Penmanshiel 2022/2023）两年相对B3-A改善均值：W1 -3.4903%，W2 +0.0495%，W3 +0.3603%（h1,h2选B3-A；h3,h4选M-A），W4 -0.1990%（2022/2023覆盖88.07%/88.42%）；选中W3并冻结selection_freeze.json（a041b917…）。
关键数字: CW1判定J1（正向Penmanshiel 2024）：B3-A覆盖89.6746%/Winkler 1358.6229，W3覆盖89.6950%/Winkler 1361.4740；C1/C2通过，C3不通过（B3-A−W3均值-2.8511 kW，95% CI [-7.7841,+2.3752]）。
关键数字: CW1判定J2（反向Kelmarsh 2024）：B3-A覆盖90.2497%/Winkler 1154.7523，W3覆盖90.3050%/Winkler 1185.0708；C1/C2通过，C3不通过（B3-A−W3均值-30.3185 kW，95% CI [-37.1175,-23.6942]）。
关键数字: CW1结论STOP_CW1（两组均过C1/C2实用门槛，均未过C3新意门槛）；详情见output/conformal_cw1/RESULT_SUMMARY.md与r1/result.json。
需要 Claude 做的: 复核CW1结果与验证记录（output/conformal_cw1/），按STAGE_PLAN规则（两组都没过C3→第2轮换方向）准备CW2或等T1完成后统一交接；L1继续等权重。
需要用户决定: 无（T1正在后台GPU运行，完成后将回收结果并再次更新交接）。
