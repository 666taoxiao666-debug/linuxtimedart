# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-01 第 17 次（K1b 结题 + D2-0 首轮） -->
轮次: R1/2 | 阶段: D2-0 | 分支: exp/D2-0 | 基线提交: bc6de12
请求: 审方案（D2-0 STAGE_PLAN）+ 实现并运行只读审计
待审范围: git diff HEAD -- docs/STAGE_PLAN.md AGENTS.md docs/review/DECISIONS.md
允许改动: AGENTS.md, scripts/audit_d2_feasibility_v34.py, tests/test_audit_d2_feasibility_v34.py, output/kelmarsh_hourly/d2_feasibility_v34/**
审查文件: docs/review/D2-0-R1-codex.md
本轮决议: K1b 的 R2 一节采纳 3 条（R2-B1、R2-B2 你的直接修复我已逐行复核；R2-S1 采纳，K1 的归因已改写）

需要你做的:
1. **K1b 结题**：我用本地副本独立复算，确认是 E2，6 个臂的数字一致。用户 2026-11-01 同意合并。请先在 exp/K1b 上提交本轮的 DECISIONS.md 和 HANDOFF.md，再用 `--no-ff` 把 exp/K1b 合回主线 `codex/self-evolving-wiki` 并推送，exp/K1b 保留。STAGE_PLAN.md 和 AGENTS.md 这时先不提交，它们是 D2-0 的待审内容，带到下一步。
2. **D2-0**：从主线拉 `exp/D2-0`，只审 STAGE_PLAN（D2 精简版的第一个闸门）和 AGENTS.md 新增的一条说明。没有阻断，就由你写只读审计脚本 `scripts/audit_d2_feasibility_v34.py`，A、B 两部分的口径和判定都按 STAGE_PLAN，然后在服务器上跑一次。A 部分只看已下载的文件；测试年只记哈希和表头；Penmanshiel 的功率、风速数值一律不统计。
3. **结果**：拷到 `output/kelmarsh_hourly/d2_feasibility_v34/`，核对 SHA256。跑 `--codex` 之后，按 v2 覆盖你那一节，写明：A1、A2、A3、B1 各自是否通过；O-curve 和 P-curve 的 2022 MAE/RMSE；测试年和 SHA256 清单的位置。如果 Penmanshiel 或当地 MIDAS 还不在服务器上，写清缺什么，交用户。

需要用户决定: 无（D2 精简版和合并 K1b 已由用户 2026-11-01 选定）

说明:
- D2 精简版只保留分解与冷启动（原 H2、H4），删掉依赖 κ 和站点编码器的 H1、H3。后续阶段（D2-1 至 D2-3）只在 STAGE_PLAN 里写了方向，细节暂缓，本轮不用审。
- AGENTS.md「分支」小节新增一条：有效的负结果也合回主线，只有技术失败的实验才只留分支。原因是交接和审查文档都随实验分支提交，不合回去主线就看不到。
- 边界：不训练，不下载，不改冻结协议、v32 曲线和标签；Penmanshiel 测试年的内容一律不看。
- 小约定：K1b 的 launch.json 只有全量快照 `source_sha256_lf`（339 个文件，其中包括每阶段都会改的 STAGE_PLAN），检查脚本没有拿它当冻结清单。以后真实运行的审计请另写一个 `source_code`，只列本次运行实际用到的源码，检查脚本锁的就是它。

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
