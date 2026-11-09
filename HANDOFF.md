# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-09 第 56 次（M2 R2：R1-B1 的 W4 改为过去固定窗口拟合；S1–S5 写清；Optuna 交 Cursor） -->
轮次: R2/2 | 阶段: M2 | 分支: exp/M2 | 基线提交: fec8edc
请求: 审方案后实现并运行（只审改动；CW1 无阻断就正式运行一次；T1 等 Cursor 交回 Optuna 版本后再运行）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/pilot_conformal_cw1.py, tests/test_pilot_conformal_cw1.py, output/conformal_cw1/**, scripts/tune_t1.py, configs/t1_tuning_protocol.json, tests/test_tune_t1.py, output/t1_tuning/**, scripts/llm_encoder_search_l1.py, models/ProgramEncoder.py, tests/test_llm_encoder_search_l1.py, configs/l1_llm_encoder_protocol.json, output/l1_llm_encoder/**, output/external_acquisition/llm_weights/**, output/conformal_c0b/**, output/external_acquisition/gfs_g0/**, output/hill_of_towie_e1/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/M2-R2-codex.md
本轮决议: 采纳 6（R1-B1、S1–S5）/ 反驳 0 / 暂缓 0 / 转实验 0

需要你做的:
1. **提交**本轮的 STAGE_PLAN 和 DECISIONS。
2. **只审改动部分**，这是第 2 轮，也是最后一轮；仍有分歧就写明，交用户决定：
   - R1-B1：W4 的填充值、标准化、k-means 和功率四分位边界，只用"源校准年 + 目标风场上一年"的无标签特征拟合，然后冻结；测试要求历史相同、未来后缀不同时结果完全相同。
   - S1–S5：W4 的状态键和半径来源；W2 的拟合登记；挑选、W3 平局和 CI 的规则；W4 两个对照的差值方向；T1 的回顾性措辞。
3. **运行**：
   - CW1 无阻断就实现并正式运行一次（CPU、2 个线程），交回内容同第 55 次。
   - T1 等 output/t1_tuning/optuna_install.json 显示 timedart 环境里 Optuna 能导入、版本已登记、已有包没有变化之后，再按方案实现并运行（GPU）。如果安装改动了已有包，停下交用户。
   - L1 仍然等权重。

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
