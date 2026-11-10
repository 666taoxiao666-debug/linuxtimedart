# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-10 第 61 次（L1 程序沙箱模块由 Claude 写好，交你审查并跑 torch 测试；搜索脚本分段写） -->
轮次: R2/2 | 阶段: M4 | 分支: exp/M2 | 基线提交: aae8538
请求: 审代码后运行（先审 Claude 写的 L1 模块，再写搜索脚本并启动 L1；CW3 照第 60 次继续）
待审范围: git diff HEAD -- models/ProgramEncoder.py tests/test_program_encoder.py docs/STAGE_PLAN.md
允许改动: models/ProgramEncoder.py, tests/test_program_encoder.py, scripts/llm_encoder_search_l1.py, tests/test_llm_encoder_search_l1.py, configs/l1_llm_encoder_protocol.json, output/l1_llm_encoder/**, scripts/pilot_conformal_cw3.py, tests/test_pilot_conformal_cw3.py, output/conformal_cw3/**, output/external_acquisition/llm_weights/**, output/t1_tuning/**, output/conformal_cw2/**, output/conformal_cw1/**, output/conformal_c0b/**, output/hill_of_towie_e1/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/M4-R2-codex.md（续写 L1 代码审查）
本轮决议: 无新编号；上一轮你写 ProgramEncoder.py 时因 Gemini 网关的 max_output_tokens 被截断，文件没生成。按用户要求改为 Claude 写这个自包含模块，你来审

需要你做的:
1. **审 models/ProgramEncoder.py**（按 STAGE_PLAN M1 的 L1 规则）。本地 11 项测试里，静态的 7 项已通过，torch 的 4 项在沙箱里跳过，请在服务器 timedart 环境运行 `python -m unittest tests.test_program_encoder`。
   - 发现确定缺陷：按"唯一最小修法先行"直接改，写进审查文件。设计上的分歧：写成 B/S 编号交我。
   - 内容：AST 白名单、子进程检查（形状、有限、梯度、batch 不变性、站点置换不变性、可变站数、训练规模探针）、专家 P（= EXPERT_PROGRAM）、语法 R（sample_grammar_programs，排除专家组合）、ProgramTSMixerModel（布局与 H 相同，K=11 时 backbone 参数量 8126）。
   - 子进程内存上限 8 GB（RLIMIT_AS）：如果专家程序因为 import torch 的虚拟内存就 CRASH，把上限调高作为技术修正，写进审查文件，不改语义。
2. **写 scripts/llm_encoder_search_l1.py**（每次编辑不超过 150 行，先写骨架再逐段补，避免输出上限再次截断）：
   - 数据构造 obs/geo；fit-period 归一化（fit_encoder_normalization）；训练接入（模型的 forward 是 (x, obs)）；Qwen 调用和代码提取（extract_code）；5 代 × 6 的进化循环；正式阶段和判定。
   - 提示词末尾附上 PROGRAM_RULES_TEXT，让大模型知道可用的名字和 dim 规则，减少被拒的程序。
   - 你配置里 L_blind 的提示词写的是 "P [16]"，请改成 "P [len(INIT_P)]"。
3. **核对并修正你的 configs/l1_llm_encoder_protocol.json**：weights_dir 写的是 datasets/external_models/...，交付清单记的是 llm_weights/Qwen2.5-Coder-7B-Instruct，以实际路径为准。之后启动 L1，CW3 照第 60 次运行；交回内容同第 60 次。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-10：CW3完成(STOP_AT_SELECTION)；L1 r1因max_new_tokens=512截断在搜索阶段中断 -->
轮次: R2/2 | 阶段: M4 | 分支: exp/M2 | 提交号: 36a142d（结果归档见git log）
审查结论: 通过（0条阻断），见 docs/review/M4-R2-codex.md；unittest 23/23 与双 --smoke 均通过。
已直接修复: 修正 configs/l1_llm_encoder_protocol.json 中 weights_dir 与 L_blind 提示词 P [len(INIT_P)]。
运行信息: CW3（CPU 2线程，run目录 outputs/logs/Conformal_CW3/cw3_36a142d_r1，耗时 765.1s）已完成并核验 SHA256，29 个结果文件已归档至 output/conformal_cw3/。
运行信息: L1 r1（GPU，run目录 outputs/logs/LLM_L1/l1_36a142d_r1）已完成 H、P 及 R（30/30 合法完成短训练，最优 R_15 val_mae=152.21 kW），但在 L/L-blind 搜索末尾因技术缺陷中断（未进入正式阶段、未生成 test_evaluation_started.json），日志与候选已归档至 output/l1_llm_encoder/r1_failed_max_tokens_512/。
关键数字: CW3 挑选集平均相对 Winkler 收益 LN-R=-0.34%、LJ-R=-0.90%（selection_2022 B3-A=1422.88 / LN-P=1424.40 / LN-R=1433.93 / LJ-P=1421.56 / LJ-R=1434.39 kW；J1 B3-A=1358.62 / LN-P=1363.99 / LN-R=1380.74 / LJ-P=1359.38 / LJ-R=1384.31 kW），无候选过门禁，按冻结规则判定 STOP_AT_SELECTION（C线收尾）。
关键数字: L1 r1 中断根因：EXPERT_PROGRAM 约 620 tokens，configs/l1_llm_encoder_protocol.json 的 max_new_tokens=512 导致 Qwen2.5-Coder-7B 生成的 60 个候选有 59 个在 return 前被截断而缺闭合 ```，extract_code 原样返回首行 ```python 触发 line 1 syntax error，随后回退值 float('inf') 触发 JSON allow_nan=False 报错。
需要 Claude 做的: 1) 复核 CW3 结果（STOP_AT_SELECTION，C 线结束）；2) 审核并批准 L1 技术修复后启动 r2：将 max_new_tokens 调为 1536（并在提示词要求不写注释）、extract_code 剥离未闭合首行 ```python、回退 val_mae 用 None 代替 inf，可复用 r1 已完成的 R_candidates 或重跑完整 L1。
需要用户决定: 无
