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
<!-- 由Codex覆盖。2026-11-10：ProgramEncoder审查+23项测试与双smoke通过；CW3与L1正式运行已启动 -->
轮次: R2/2 | 阶段: M4 | 分支: exp/M2 | 提交号: 36a142d（运行后见git log）
审查结论: 通过（0条阻断），见 docs/review/M4-R2-codex.md；models/ProgramEncoder.py 审查通过，RLIMIT_AS=8GB 在服务器实测正常无需调高。
已直接修复: 修正 configs/l1_llm_encoder_protocol.json 中 weights_dir=llm_weights/Qwen2.5-Coder-7B-Instruct 及 L_blind 提示词 P [len(INIT_P)]。
运行信息: 服务器 worktree /home/XT/nuist/pythoncode/TimeDARTFirst_M4_36a142d；unittest 23/23 通过（含 ProgramEncoder 11/11、CW3 7/7、L1 5/5），CW3 与 L1 的 --smoke 均通过。
运行信息: CW3 正式实例已启动（CPU 2 线程，PID 823712），run 目录 outputs/logs/Conformal_CW3/cw3_36a142d_r1，launch.json 已归档至 output/conformal_cw3/launch.json。
运行信息: L1 正式实例已启动（GPU，PID 823713），run 目录 outputs/logs/LLM_L1/l1_36a142d_r1，audit/launch.json 已归档至 output/l1_llm_encoder/audit/launch.json。
关键数字: L1 预检 49 个权重文件（15,303,338,337 字节）与 328 个输入数据文件 SHA256 全部通过；Kelmarsh 站点张量 (58465, 36, 7)，Penmanshiel (49705, 11, 7)。
需要 Claude 做的: 等待 CW3 与 L1 正式运行完成后复核结果（完成后我将取回结果并更新本节）。
需要用户决定: 无
