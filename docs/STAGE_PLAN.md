# STAGE_PLAN — 只写下一个阶段

作者：Claude（Codex 只提意见，不改本文件）。上限 150 行：要加新内容时，先删旧内容。
内容只从现有代码和结果中提取；拿不准的地方标「待确认」；事先设定的阈值标「事先设定」。
存档：CW3 的方案原文（含区间线的数据暴露清单）见 `git show af938db:docs/STAGE_PLAN.md`；L1 的原始规则见 `git show a1ebfa0:docs/STAGE_PLAN.md` 的 L1 条；CW3 结果复核见 DECISIONS.md 的 M4 一节。

## 阶段名

M4（续）：CW3 已结束；L1 修复技术缺陷后完整重跑一次（r2，GPU）。
- CW3：2026-11-10 完成，STOP_AT_SELECTION，两个候选都不合格，C 线按事先规则结束。J3、J4 没有计算。
- T1：已完成，RETAIN_EXISTING；5 个 seed 的集成作为单独的改进进入 Hill of Towie 联合确认。
- L1 r1：H、P、R 都跑完（R 30/30 合法，最好的 R_15 短训练验证 MAE 152.21 kW），但 L、L-blind 的 60 个候选全部失败。原因是技术缺陷，不是方法结果（见下）。没有冻结搜索选择，也没有进入正式阶段。

## 目标

1. **CW3 告诉我们的事（只作描述）**：
   - 挑选数据上，LN-R 相对 min(匹配对照, B3-A) 的平均改善是 −0.34%，LJ-R 是 −0.90%。
   - 上游尺度相对匹配对照的 Winkler 差在 4 组里符号不一致：LN-R 减 LN-P 是 +9.5、−12.1、+16.8、−11.9 kW。
   - 所以 CW2 里"上游尺度有稳定增益"的信号，叠加到功率近邻之后不再稳定。R1-S1 的假设（两者可以叠加）不成立。
   - 爬坡窗口的覆盖率，所有方法都在 76–79%；上游尺度最多只提高约 1 个点。
   - 论文里照实写 C 线已经成立的实用结论：有界 ACI + F + 功率近邻（B3-A）比 B0、B1、B2-A 低 10% 以上。
2. **L1 r2**：在修好截断缺陷的前提下，按原规则完整跑一次 L1。检验的问题不变：大模型写的上游编码程序，能否在两个风场上同时赢过手工特征 H、专家程序 P 和随机语法程序 R*。

## 假设与判定标准

**L1 r1 的技术缺陷（确定缺陷，已核实）**
- max_new_tokens = 512，但专家程序约 620 个 token，R 的程序有 2400–2580 个字符。
- 60 个大模型输出全部在 return 之前被截断，没有一个带闭合的代码围栏。
- extract_code 遇到未闭合的围栏会返回整段文字，所以都报"line 1 syntax error"，掩盖了真正的截断原因。
- 全部失败时，回退记录的 val_mae = inf，写 JSON 时报错，进程中断。
- 我用修好的 extract_code 重新解析 r1 的 60 个输出：仍然 0 个合法，报的都是括号未闭合或缺 return。这确认根因是 token 上限，不是提取逻辑。

**r2 的修法（只修技术缺陷，事先设定）**
- `max_new_tokens` 改为 1536（约 5000 个字符，是 R 程序长度的 2 倍左右）。
- L 和 L-blind 的提示词末尾各加同一句："Do not write comments or docstrings."（两臂相同）。
- `models/ProgramEncoder.py` 的 extract_code：未闭合的围栏取围栏之后的全部文字（Claude 已改，加了测试）。
- 全部失败时，回退记录的 val_mae 记为 null，不再用 inf。
- 协议 JSON 的 `protocol_id` 改为 `l1_llm_encoder_v2`，加一条 `supersedes` 说明 r1 是因为 max_new_tokens 失败的技术运行。
- 不变的部分：
  - 判定阈值（L* 比 H 好 ≥1.5%，比 R*、P 都好 ≥0.5%，两个风场都要满足）；
  - 臂（H、P、R、L、L-blind）、预算（每臂 30 个）、温度、top_p、种子；
  - 数据年份、训练配方、语法 R、"失败照样占名额、不补生成"的规则。

**r2 的运行规则（事先设定）**
- 从头完整重跑：H、P、R、L、L-blind 的搜索，然后进入正式阶段。不复用 r1 的 R 结果，保证所有结果在同一份代码和哈希下产生。R 只多花约 33 分钟。
- 启动前做真实大模型预检：用一个不在搜索种子里的种子，L 和 L-blind 各生成 1 个第 1 代程序。两个输出都要带闭合围栏，并通过 validate_source，才正式启动。预检的输出只存档，不进入搜索。
- 每个候选记录 `n_new_tokens` 和 `hit_max_new_tokens`。
- **技术有效性门槛**：L 或 L-blind 里，有 ≥3 个（10%）候选碰到 token 上限，就判为技术失败。这时只跑完搜索、存档，不冻结选择、不进入正式阶段，交回给我。
- 碰到上限的候选不到 3 个，就照常继续；碰到上限的那几个按"失败照样占名额"处理。
- 合法程序很少，如果是因为违反规则（不是截断），属于真实结果，照常继续。
- 正式阶段的统计和三档结论，按 a1ebfa0 的 L1 条，不变。
- top-10 程序做因果性抽查（只用起报时刻 −1 h 之前的观测），由 Claude 在结果交回后做。

## 实验配置表

| 线 | 内容 | 规模 |
|---|---|---|
| CW3 | 已结束：STOP_AT_SELECTION，C 线收尾 | — |
| T1 | 已完成：RETAIN_EXISTING，种子集成进入联合确认 | — |
| L1 r2 | 预检 2 个生成；搜索 H、P、R30、L30、L-blind30（Kelmarsh）；正式 5 臂 × 3 seed × 2 风场 | GPU，约 4–5 小时（待确认） |

- L1 r2 只运行一个实例。正式阶段开始前写 search_selection_freeze.json，冻结后不再改选择。

## 改动文件清单（= HANDOFF「允许改动」）

- Claude（已改）：`models/ProgramEncoder.py`（只改 extract_code）、`tests/test_program_encoder.py`（extract_code 测试）。
- Codex：
  - `configs/l1_llm_encoder_protocol.json`（max_new_tokens、两句提示词、protocol_id、supersedes）；
  - `scripts/llm_encoder_search_l1.py`（回退 val_mae 用 null；记录 n_new_tokens 和 hit_max_new_tokens；技术有效性门槛；真实大模型预检）；
  - `tests/test_llm_encoder_search_l1.py`（全部失败时 JSON 能写出；门槛的真值表）；
  - `output/l1_llm_encoder/**`（r2 写在顶层；r1 留在 r1_failed_max_tokens_512/）。

## 本阶段不做什么

- 不改 L1 的判定阈值、臂、预算、数据年份和训练配方。
- C 线不开第 4 轮。不碰 Hill of Towie 封存年。
- 不在看到正式结果后改选择或重跑。

## 风险（≤5 条）

1. 1536 个 token 仍可能不够：后几代提示词里给的程序更长，大模型可能写得更长。有技术有效性门槛兜底。
2. 7B 模型在白名单规则下合法率可能很低，L 的有效名额变少。这算真实结果，如实报告。
3. 搜索只在 Kelmarsh 上做，R 的最好程序（152.21 kW）已经很接近，L 要比 R* 好 0.5% 不容易。之前的估计不变：L1 通过的概率低。
4. r1 的 H、P 短训练数字没有存下来（写 JSON 时崩溃），r2 要一起交回。

## 存档引用

- CW3：`output/conformal_cw3/result.json`、`selection_freeze.json`、4 组的 `seed_*.json`。
- L1 r1（技术失败）：`output/l1_llm_encoder/r1_failed_max_tokens_512/`。
- CW2：`output/conformal_cw2/RESULT_SUMMARY.md`；CW1：`output/conformal_cw1/`；C0b：`output/conformal_c0b/`。
- 冻结站点：Kelmarsh v32 的 36 站；Penmanshiel 的 11 站；Hill of Towie `output/hill_of_towie_e1/l0b_r2/qualified_stations.json`（5909e143…）。
