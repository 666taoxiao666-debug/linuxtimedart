# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-10 第 62 次（CW3 复核通过，C 线结束；L1 r1 技术失败，批准修复后完整重跑 r2） -->
轮次: R2/2 | 阶段: M4 | 分支: exp/M2 | 基线提交: af938db
请求: 审代码后运行（审 Claude 改的 extract_code，实施 r2 修复，预检通过后启动 L1 r2）
待审范围: git diff HEAD -- models/ProgramEncoder.py tests/test_program_encoder.py docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: models/ProgramEncoder.py, tests/test_program_encoder.py, configs/l1_llm_encoder_protocol.json, scripts/llm_encoder_search_l1.py, tests/test_llm_encoder_search_l1.py, output/l1_llm_encoder/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**（后三项是本地历史未跟踪文件，只为通过范围检查，不要改）
审查文件: docs/review/M4-R2-codex.md（续写一节"第 62 次"）
本轮决议: 无新编号。CW3 的 STOP_AT_SELECTION 复核通过（本地复算一致）；L1 r1 同意是技术缺陷，批准你提的三项修复，另加两项保障。见 DECISIONS M4 节末两段

需要你做的:
1. **审 extract_code 的改动**（models/ProgramEncoder.py:358，未闭合围栏取围栏之后的全部文字），在服务器跑 `python -m unittest tests.test_program_encoder`（11 项）。
2. **按 STAGE_PLAN「r2 的修法」「r2 的运行规则」实施**（每次编辑 ≤150 行）：
   - 协议：max_new_tokens 1536；L、L-blind 提示词末尾都加 "Do not write comments or docstrings."；protocol_id 改为 l1_llm_encoder_v2，加 supersedes。
   - 脚本：回退 val_mae 用 null；worker 返回每个输出的 n_new_tokens，候选记录 hit_max_new_tokens；L 或 L-blind 有 ≥3/30 碰到上限，就在搜索后停下，不冻结选择、不进正式阶段。
   - 测试：全部失败时 JSON 能写出；门槛 2/30 继续、3/30 停。
3. **预检后启动 r2**：用搜索种子以外的种子，L、L-blind 各真实生成 1 个第 1 代程序，都要闭合围栏并通过 validate_source。通过后从头完整重跑（H、P、R、L、L-blind → 正式阶段），只跑一个实例。
   - r2 写在 output/l1_llm_encoder/ 顶层；r1 留在 r1_failed_max_tokens_512/。
   - 交回：预检输出；H、P 的短训练验证 MAE；各臂的合法数和碰上限数；R*、L*、L-blind* 的程序；正式阶段两个风场 5 臂的 MAE、RMSE、CI 和判定；top10_programs.json。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-10：第62次修复已实施；真实预检失败，未启动正式r2 -->
轮次: R2/2 | 阶段: M4 | 分支: exp/M2 | 提交号: 6c1895d（本次结果记录见git log）
审查结论: extract_code及批准的r2修复通过；真实预检0/2通过，按启动门禁停止，见 docs/review/M4-R2-codex.md 第62次。
已直接修复: 6c1895d 实施v2/1536、两臂相同禁注释提示、null回退、token计数、3/30门禁与真实预检；未改阈值/臂/年份/训练配方。
运行信息: 服务器 /home/XT/nuist/pythoncode/TimeDARTFirst_L1_6c1895d；timedart 执行 python -m unittest tests.test_program_encoder tests.test_llm_encoder_search_l1，18/18通过、无跳过，含训练smoke。
运行信息: 数据根 /home/XT/nuist/pythoncode/TimeDARTFirst；run为其 outputs/logs/LLM_L1/l1_6c1895d_r2；仅执行 --preflight（PID838095，已退出），没有启动正式实例、没有冻结搜索选择。
运行信息: 真实权重49文件/15,303,338,337字节哈希核对通过；预检JSON及日志取回 output/l1_llm_encoder/ 顶层，r1原样保留。
关键数字: L seed900001生成1031 tokens，闭合围栏，但INIT_P有17项（上限16）；L_blind seed900002生成579 tokens，闭合围栏，但第4行解包赋值被AST拒绝；两者均未碰1536上限。
关键数字: protocol SHA256=52b4420af1c4e2c8711716f0c31f4fbc1b55be319e776ca8492c9db8844aa4fd；script SHA256=2d11e6155a794e8fbbd845905187883468f0bfd5ee3feba45c44bafa9a3a80e4。
需要 Claude 做的: 复核预检证据；BLIND_EXAMPLE（脚本371行）自身含白名单禁止的解包赋值，却标成valid。给出下一版最小提示修正和有限预检规则，不放宽AST、不靠换seed反复抽到通过；当前未实施额外改动。
需要用户决定: 本轮已到R2上限且预检失败，正式r2未启动；后续提示调整及再次真实预检须明确裁决范围，不能作为本轮继续重试。
