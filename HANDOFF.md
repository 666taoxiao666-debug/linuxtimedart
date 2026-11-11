# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-10 第 63 次（用户裁决：L1 最后一次 r3 + Penmanshiel 只读爬坡复现，两线并行） -->
轮次: R1/2 | 阶段: M5 | 分支: exp/M2 | 基线提交: 0bbf21a
请求: 审代码后运行（审 Claude 的两处改动；实施 L1 r3 修复并运行一次；并行跑只读复现）
待审范围: git diff HEAD -- models/ProgramEncoder.py tests/test_program_encoder.py scripts/ramp_diagnostics.py tests/test_ramp_diagnostics.py docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: models/ProgramEncoder.py, tests/test_program_encoder.py, scripts/ramp_diagnostics.py, tests/test_ramp_diagnostics.py, scripts/llm_encoder_search_l1.py, tests/test_llm_encoder_search_l1.py, configs/l1_llm_encoder_protocol.json, output/l1_llm_encoder/**, output/ramp_diagnostics/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**（后三项是本地历史未跟踪文件，只为通过范围检查，不要改）
审查文件: docs/review/M5-R1-codex.md
本轮决议: 采纳 1（R2-B1，用户裁决授权最后一次 r3）/ 反驳 0 / 暂缓 0 / 转实验 0。另外我改正了自己定的预检规则：r3 预检只查技术项，见 DECISIONS M4 节末段

需要你做的:
1. **审 Claude 的改动并跑测试**：
   - models/ProgramEncoder.py 新增 PROGRAM_STRUCTURE_TEXT（单名赋值、唯一末尾 return、P[i] 下标范围），接在 PROGRAM_RULES_TEXT 末尾；
   - 新文件 scripts/ramp_diagnostics.py（纯 numpy，判定 M1–M3 见 STAGE_PLAN），本地对 Kelmarsh 2024 复算与我的诊断一致。
   - 服务器跑 `python -m unittest tests.test_program_encoder tests.test_ramp_diagnostics tests.test_llm_encoder_search_l1`。
2. **L1 r3（最后一次，按 STAGE_PLAN「L1 r3 的修法」）**：
   - 实施：BLIND_EXAMPLE 三行解包改成逐行单变量赋值；L-blind 规则末尾接 pe.PROGRAM_STRUCTURE_TEXT；测试要求 EXPERT_PROGRAM、BLIND_EXAMPLE 都过 validate_source 和 check_program；protocol_id 改为 l1_llm_encoder_v3，加 supersedes。
   - 预检：种子 900003（L）、900004（L-blind），两个输出都闭合围栏且 n_new_tokens < 1536 就通过；validate_source 只记录不卡启动。不通过 → L1 关闭，不再修。
   - 通过后从头完整跑一次（H、P、R、L、L-blind → 正式阶段）；r3 写到 output/l1_llm_encoder/r3/，r1、r2 预检原样保留。
   - 交回：预检输出；H、P 短训练验证 MAE；各臂合法数、碰上限数、被拒原因分布；R*、L*、L-blind* 程序；正式阶段两个风场 5 臂的 MAE、RMSE、CI 和判定；top10_programs.json。
3. **Penmanshiel 只读复现（CPU，和 L1 并行，零训练）**：
   - 先核对 P3 r2 各折 cohort.npz 的键名；名字不同就用 --truth-key 等参数指定；没有 PersistenceHour 就停下交回。
   - 判定：`python scripts/ramp_diagnostics.py --out output/ramp_diagnostics/penmanshiel_T22_T23.json --fold T22 <T22>/cohort.npz "<T22>/P-upstream_s*.npz" "<T22>/P-last10_s*.npz" --fold T23 <T23 同上>`
   - 描述：T24 同样的命令单独输出到 penmanshiel_T24_descriptive.json，不进判定。
   - 交回：两个 JSON 和控制台输出，以及输入文件的 SHA256（脚本已写进 JSON）。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-11：第63次，L1 r3和CPU统计均结束，结果待复核 -->
轮次: R1/2 | 阶段: M5 | 分支: exp/M2 | 提交号: L1运行723a1e8；M3补齐99b4675（结果记录见git log）
审查结论: L1通过；爬坡2项确定缺陷按最小修法先行，见 docs/review/M5-R1-codex.md；无遗留启动阻断。
已直接修复: 723a1e8修复M1保留完整时间轴的零事件簇；缺测元数据缺失时M3为UNAVAILABLE而不是无缺测；NPZ句柄关闭。未改阈值/年份/臂。按批准修复blind示例、对称结构提示和v3技术预检。
运行信息: worktree /home/XT/nuist/pythoncode/TimeDARTFirst_M5_723a1e8；服务器3个测试模块共30项，29通过，1个本地Kelmarsh数组回归因文件不在该worktree跳过；含两个提示示例的动态验证、训练smoke。该回归本地通过。
运行信息: L1 run /home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/LLM_L1/l1_723a1e8_r3；PID840060，完整命令见 output/l1_llm_encoder/r3/dispatch.json；CUDA，独立会话，只启动一次；源码/协议与预检哈希匹配，49权重文件核验通过。
运行信息: L1于2026-11-10 09:18:51 UTC完成，PID840060已退出，completed标记STOP_L1；30次正式拟合记录齐全，三seed指标均值复算通过；JSON/log取回output/l1_llm_encoder/r3/，没有重启或r4。
运行信息: 搜索合法数R/L/blind=30/14/5（各30槽）；两LLM臂触token上限均0。H/P短训验证MAE=150.9910/153.3405；程序和top10已归档。
运行信息: L1三seed均值MAE/RMSE/R2：Kelmarsh H=160.020/242.169/.82526，L=160.392/244.272/.82221；Penmanshiel H=184.410/299.395/.78480，L=184.355/299.958/.78400。全5臂见l1_summary.json。
运行信息: L对H集成误差增益CI：Kelmarsh -0.524[-1.720,.472]kW；Penmanshiel -0.418[-2.000,1.165]；未证明优于H，STOP_L1。该集成口径不同于逐seed指标均值，不混用。
运行信息: 审计限制：脚本766行已eval，769/777行才写formal冻结/测试开始标记；run未保存正式预测数组或checkpoint，不能独立复算CI。保留原产物，未发现据测试选参的直接证据，但不宣称完整独立验收；summary阶段标签仍M4。
运行信息: CPU只读run为数据根 outputs/logs/Ramp_M5/ramp_723a1e8；结果及日志已取回 output/ramp_diagnostics/，SHA256与服务器一致；锚点persistence=PersistenceHour，窗口starts。无训练、新预测或封存数据访问。
关键数字: T22 M1改善13.1%，增益CI[53.2,66.8]kW；M2 AUC增益+0.052，CI[0.037,0.067]；均通过。
关键数字: T23 M1改善12.6%，增益CI[48.1,64.2]kW；M2 +0.051，CI[0.032,0.070]；均通过；开发复现结论RAMP_REPLICATED，不是独立确认。
关键数字: T24仅描述：M1改善13.0%，CI[49.2,65.6]kW；M2 +0.060，CI[0.040,0.081]；未生成总判定。
运行信息: 用户本轮明确授权补齐并运行M3；只读既存prepared/upstream.csv（SHA256 6b7dfc421ced15bfd927c417471019745e947877bcde27a8ad0988a0fdafa208），沿用原11特征any-nonfinite规则，首目标时刻-1h对齐。新worktree TimeDARTFirst_M3_99b4675，不改L1部署目录。
运行信息: M3补算完成，run为数据根 outputs/logs/Ramp_M5/m3_99b4675，本地 output/ramp_diagnostics/m3_recovered/；新旧M1/M2逐字段完全相同，JSON传输哈希一致；9项本地测试通过，服务器8通过1本地数组回归跳过。
关键数字: M3整体MAE增益T22=-0.9661，CI[-1.6898,-0.3084]kW，缺测4408/37741；T23=-0.7485，CI[-1.4889,-0.0543]，缺测5002/36838；判定NO_FALLBACK，开发复现RAMP_REPLICATED。
关键数字: T24 M3仅描述=-0.9361，CI[-1.6449,-0.2630]kW，缺测1567/13449；不纳入总判定。旧UNDETERMINED文件保留，由m3_recovered结果补全。
需要 Claude 做的: 复核R1-B1/B2及M3恢复对齐；按NO_FALLBACK不加入缺测回退。复核L1完整结果/top10及审计限制，决定结果可报告范围；维持STOP_L1，无r4。审核完成后再请用户决定是否合并exp/M2，当前未合主线。
需要用户决定: 无。
