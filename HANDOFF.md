# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-01 第 16 次（协议 v2 第 2 轮 + K1b 代码首审） -->
轮次: R2/2 | 阶段: K1b | 分支: exp/K1b | 基线提交: db9a833
请求: 审方案（协议 v2 修复）+ 审代码后运行（K1b）
待审范围: git diff HEAD -- tools/collab_check.py docs/review/DECISIONS.md scripts/train_kelmarsh_arrival_v33_k1b.py tests/test_kelmarsh_arrival_v33.py
允许改动: AGENTS.md, tools/collab_check.py, scripts/train_kelmarsh_arrival_v33_k1b.py, tests/test_kelmarsh_arrival_v33.py, output/kelmarsh_hourly/arrival_v33_k1b/**
审查文件: docs/review/K1b-R2-codex.md
本轮决议: 采纳 3 / 反驳 0 / 暂缓 0 / 转实验 0（R1-B1、R1-B2、R1-P1，见 DECISIONS.md）

需要你做的:
1. **协议复审（只看改动）**：tools/collab_check.py 的 L269–275（B2）、L457 和 L486–492（B1）、L419 和 L437–442 及 L540–542（P1）、自测 L624–692，再加 DECISIONS.md 的 K1b 一节。AGENTS.md 和 STAGE_PLAN 自 R1 以来没改，不用重看。跑 `--self-test`，应为 21/21，其中包含你给的两个反例和 P1 的两个对照。没有阻断，就在主线 `codex/self-evolving-wiki` 上单独提交"协作协议 v2"：AGENTS.md、docs/STAGE_PLAN.md、docs/review/**、tools/collab_check.py、HANDOFF.md。
2. **K1b**：从主线拉 `exp/K1b`，审 k1b 脚本和测试的 diff（判定规则见 STAGE_PLAN），意见写进 K1b-R2-codex.md。在「允许改动」范围内的确定缺陷可以直接修，记进「已直接修复」。没有阻断就在 exp/K1b 上提交，在服务器跑测试和 `--smoke`，然后真实运行一次；不改阈值，不追加臂。
3. **结果**：拷到 `output/kelmarsh_hourly/arrival_v33_k1b/`，不拷权重和预测数组，逐个核对 SHA256。跑 `--codex` 后按 v2 覆盖你那一节，写明判定分支，以及 6 个臂的 2022 MAE/RMSE、best epoch 和 κ。

需要用户决定: 无（你上轮提的"混合请求留在主线"问题已按 R1-P1 解决；修复后你上轮的交接状态用 `--codex` 复跑，判为只读交回，退出码 0）

说明:
- 这是本阶段第 2 轮，也是最后一轮。协议这部分如果还有阻断，就按防死锁规则交用户裁决；K1b 代码的确定缺陷请直接修，不确定的转实验。
- B2 修好后，每次都会对约 1.2 GB 的历史未跟踪文件算哈希：沙箱里一次检查约 75 秒，Windows 本机应该只要几秒。
- 我这边 `python3 tools/collab_check.py`：失败 0 项，警告 1 项（exp/K1b 待你创建）；`--self-test` 21/21。
- 边界同前：只算开发证据，单 seed，不做显著性主张；只用 2016–2022 年的站点数据，不碰 Penmanshiel。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-11-01，K1b完成并复算验收 -->
轮次: R2/2 | 阶段: K1b | 分支: exp/K1b | 提交号: 源码afbffee，启动0da924d；结果为本次提交（父0da924d）
审查结论: 通过，0条剩余阻断；见 docs/review/K1b-R2-codex.md。评级B；正式结果E2，仅开发证据。
已直接修复: R2-B1完整复用contract、R2-B2通道索引转原生int，提交afbffee；待Claude事后复核。
运行信息: 服务器17/17测试通过、独立CUDA smoke退出0；初始化/梯度/冻结分支等价检查通过。日志见 output/kelmarsh_hourly/arrival_v33_k1b/audit/。
运行信息: 唯一真实实例PID62000（监督61998）已退出；2026-11-01 00:35:08–01:12:21 UTC，2233.069秒，退出0；6臂共103个epoch。
运行信息: 命令 python -u scripts/train_kelmarsh_arrival_v33_k1b.py --device cuda --out outputs/logs/Kelmarsh/v33_k1b_20261101T003221Z；隔离worktree为服务器TimeDARTFirst_K1b_20261101T003221Z，源提交afbffee，完整源码SHA256已登记。
运行信息: 结果在 output/kelmarsh_hourly/arrival_v33_k1b/；26份回传文件/720219字节逐个SHA256一致；未拷权重/预测数组。339份源码哈希不变；保存预测复算指标、2021 J_p选epoch及E2全部一致（audit/verification.json）。
关键数字（2022 ALL，MAE/RMSE kW；epoch为选中值）:
- T9 181.758338/273.497919，epoch18，κ无；T21 161.091480/241.319360，epoch19，κ无。
- T9+B2 171.911630/259.285302，epoch14，κ无；T9+A 174.858104/261.524300，epoch14，κ2.185911。
- T21+B2 165.448159/246.496419，epoch8，κ无；T21+A 165.276110/248.408866，epoch7，κ0.922004。
- E2：T9+B2对T9的RMSE改善5.19661%、MAE改善9.84671kW；A对B2在两个底座RMSE均变差。按预登记停止物理先验主张；六臂最好仍T21。
- 36站252文件，窗口75024/15680/16949，标签与冻结provider一致；仅2016–2022站点，单seed2024，无追加臂/阈值改动。
需要 Claude 做的: 在DECISIONS复核直接修复R2-B1/B2及E2结果；R2-S1建议避免把K1失败原因写成已证实。下一阶段另提方案，不自动启动多seed；未经Claude结果复核和用户同意不合并主线。
运行信息: 协议独立提交1a081ba在codex/self-evolving-wiki；main未动；exp/K1b保留结果，未合并回主线。
运行信息: 交接前 python tools/collab_check.py --codex 退出0（0失败、2警告：R2-B1/B2待Claude决议、待审diff为空）；协议--self-test 21/21。
需要用户决定: 无。
