# STAGE_PLAN — 只写下一个阶段

作者：Claude（Codex 只提意见，不改本文件）。上限 150 行：要加新内容时，先删旧内容。
内容只从现有代码和结果中提取；拿不准的地方标「待确认」；事先设定的阈值标「事先设定」。
存档：M4（CW3 结果、L1 r2 修法）见 `git show 0bbf21a:docs/STAGE_PLAN.md`；CW3 方案原文和区间线暴露清单见 `git show af938db:docs/STAGE_PLAN.md`；L1 原始规则见 `git show a1ebfa0:docs/STAGE_PLAN.md` 的 L1 条。

## 阶段名

M5：L1 最后一次运行（r3，GPU）+ Penmanshiel 只读爬坡复现（CPU）。两条线并行。
- 用户 2026-11-10 裁决：M4 已到两轮上限，授权 L1 最后跑一次；无论结果如何，跑完 L1 就关闭。并行只做 Penmanshiel 只读统计。
- 已结束：CW3（STOP_AT_SELECTION，C 线结束）；T1（RETAIN_EXISTING，种子集成进入联合确认）。

## 目标

1. **L1 r3**：问题不变。大模型写的上游编码程序，能否在两个风场上同时赢过手工特征 H、专家程序 P 和随机语法程序 R*。
   - 已知的描述性证据（r1，单 seed 短训练）：R 的 30 个程序里，13 个离最好的（R_15，152.21 kW）不到 1%；前 10 名只差 0.66 kW。通道数 ≤9 的明显更差（157–160 kW）。
   - r2 预检里，L 的输出基本是专家程序的改写。这些都提示 L1 通过的概率低，跑完主要是补齐"同预算下大模型搜索 vs 随机搜索 vs 手工特征"的对照。
2. **Penmanshiel 只读统计**：检验 Kelmarsh 2024 上看到的"上游增益集中在爬坡"能否在第二个风场复现，并决定 Hill of Towie 协议要不要加"上游缺测时回退"的规则。零训练、不碰新数据。

## 假设与判定标准

**L1 r3 的修法（只修确定缺陷，事先设定）**
- R2-B1 的两个原因：
  - 给 L-blind 看的 BLIND_EXAMPLE 本身用了解包赋值，违反白名单（我本地复核：validate_source 拒绝，line 4）；
  - 两臂的规则文字都没写"每行只能给一个名字赋值"，但白名单强制执行这一条。
- 修法：
  - Claude（已改）：`models/ProgramEncoder.py` 新增 PROGRAM_STRUCTURE_TEXT，写清单名赋值、唯一的末尾 return、P[i] 的下标范围，并接在 PROGRAM_RULES_TEXT 末尾（L 用）。
  - Codex：BLIND_EXAMPLE 的三行解包改成逐行单变量赋值，其余不变；L-blind 的规则文字末尾接同一个 pe.PROGRAM_STRUCTURE_TEXT；加测试，要求 EXPERT_PROGRAM 和 BLIND_EXAMPLE 都通过 validate_source 和 check_program。
  - 协议 protocol_id 改为 l1_llm_encoder_v3，supersedes 写明 v2 因 BLIND_EXAMPLE 缺陷停在预检。
- **预检只查技术项（我改正自己的规则）**：
  - r2 的预检要求唯一的生成必须合法。但协议本来就规定不合法的程序照样占名额，合法率正是要测的结果，所以用 1 个样本卡合法性是错的。
  - r3 用新种子 900003（L）和 900004（L-blind），都不在搜索种子范围内。两个输出都闭合围栏、并且 n_new_tokens < 1536，就算通过。validate_source 的结果只记录，不卡启动。
  - 预检不通过 → L1 关闭，记为"技术上未完成"，不再修。
- 不变的部分：
  - 判定阈值（两个风场上，L* 比 H 好 ≥1.5%，比 R*、P 都好 ≥0.5%）；
  - 臂、预算（每臂 30 个）、温度、top_p、搜索种子；
  - 数据年份、训练配方、语法 R、"失败照样占名额、不补生成"；
  - max_new_tokens 1536 和"不写注释"；
  - 搜索中 ≥3/30 碰到上限就判技术失败。触发时 L1 关闭，记为"技术上未完成"，不再有 r4。
- 正式阶段的统计和三档结论按 a1ebfa0 的 L1 条。top-10 程序的因果性抽查由 Claude 在结果交回后做。

**Penmanshiel 只读复现（事先设定，阈值在 Penmanshiel 运行前固定）**
- 脚本：`scripts/ramp_diagnostics.py`（Claude 写，纯 numpy）。本地在 Kelmarsh 2024 上复算，与我 11-10 的诊断数字一致（tests/test_ramp_diagnostics.py 有回归测试）。
- 数据：P3 r2 的 T22、T23 两折，候选 P-upstream，参照 P-last10，各 3 个 seed 的平均预测；锚点是 PersistenceHour；额定 2050 kW。
- 判定（每折单独算）：
  - M1：h2–h4 中真实变化 ≥10% 额定的格子，MAE 相对改善 ≥10%，且 kW 增益的 95% CI 下界 > 0。
  - M2：h4 上 ±30% 额定的爬坡，用预测变化做得分；上爬、下爬 AUC 的平均值提升 ≥0.05，且 CI 下界 > 0。
  - M3：上游缺测窗口改用 P-last10 的预测；全体 MAE 的增益 > 0，且 CI 下界 > 0。
  - CI：起报时刻的移动块 bootstrap（块长 8，5000 次，seed 2024），整簇重抽。
- 结论（事先设定）：
  - T22、T23 都过 M1 和 M2 → RAMP_REPLICATED：论文主线用"上游观测 → 爬坡"的写法，爬坡指标进 Hill of Towie 协议当次要终点。
  - 否则 → 爬坡分析只作为 Kelmarsh 的描述。
  - 两折都过 M3 → ADOPT_FALLBACK：回退规则进 Hill of Towie 的主检验；否则不加。
- 参考（只作描述，不进判定）：Kelmarsh 2024 上，M1 是 17.9%（[63.6, 77.4] kW）通过，M2 是 +0.069（[+0.049, +0.089]）通过；M3 是 +0.12 kW（[−0.02, +0.37]），不采纳，缺测窗口只占 0.6%。
- T22、T23 是 P3 的测试年，已经用过，这里只是开发数据上的复现，不是独立确认。T24 只作描述，单独输出，不进判定。

## 实验配置表

| 线 | 内容 | 规模 |
|---|---|---|
| L1 r3 | 预检 2 个生成；搜索 H、P、R30、L30、L-blind30（Kelmarsh）；正式 5 臂 × 3 seed × 2 风场 | GPU，约 4–5 小时（待确认） |
| 只读复现 | T22、T23 判定；T24 描述；只读已有预测 | CPU，每折约 1 分钟 |

- L1 r3 只运行一个实例；正式阶段开始前写 search_selection_freeze.json。
- 只读复现先核对 cohort.npz 的键名（truth、PersistenceHour、issue_upstream_missing、times、window_starts）。名字不同就用命令行参数指定；没有 PersistenceHour 就停下交回，不用别的锚点替代。

## 改动文件清单（= HANDOFF「允许改动」）

- Claude（已改）：`models/ProgramEncoder.py`（PROGRAM_STRUCTURE_TEXT）、`tests/test_program_encoder.py`、`scripts/ramp_diagnostics.py`、`tests/test_ramp_diagnostics.py`（新文件：现有脚本里没有爬坡分层统计，冻结的 utils 不宜改，所以单独写）。
- Codex：`scripts/llm_encoder_search_l1.py`（BLIND_EXAMPLE、L-blind 规则文字、预检门槛）、`tests/test_llm_encoder_search_l1.py`、`configs/l1_llm_encoder_protocol.json`、`output/l1_llm_encoder/**`、`output/ramp_diagnostics/**`。

## 本阶段不做什么

- 不改 L1 的判定阈值、臂、预算、年份、训练配方；没有 r4。
- 只读复现不训练、不改已有预测文件、不碰 Hill of Towie。
- 不在看到结果后改 M1–M3 的阈值。

## 风险（≤5 条）

1. 7B 模型在白名单下的合法率可能很低，L 的有效名额少。这算真实结果，照实报告合法率和被拒原因。
2. 后几代提示词里的程序更长，生成可能碰到 1536 的上限，由 3/30 门槛兜底。
3. M1 按真实变化事后分层，只能说明"增益在哪里"，不能当成可部署的开关。
4. Penmanshiel 缺测在时间上成片出现时，M3 的 CI 会偏宽，可能判不出采纳。

## 存档引用

- L1：r1 `output/l1_llm_encoder/r1_failed_max_tokens_512/`；r2 预检 `output/l1_llm_encoder/preflight.json`、`R2_PREFLIGHT.md`。
- CW3：`output/conformal_cw3/result.json`；P3：`output/penmanshiel_hourly/p3_v35/r2/`。
- 冻结站点：Kelmarsh v32 的 36 站；Penmanshiel 的 11 站；Hill of Towie `output/hill_of_towie_e1/l0b_r2/qualified_stations.json`（5909e143…）。
