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
<!-- 由 Codex 覆盖。最后更新：2026-11-02，D2-0审查与实现完成，待服务器唯一实例 -->
轮次: R1/2 | 阶段: D2-0 | 分支: exp/D2-0 | 提交号: 本次源码提交（父038c5b1）
审查结论: 通过，0条阻断，评级B；docs/review/D2-0-R1-codex.md；R1-S1为非阻断建议。
已直接修复: 无；新审计与测试按Claude第17次交接明确授权实现，未改冻结源码/协议。
运行信息: K1b结题95dd54c经no-ff合入主线038c5b1并推送；exp/K1b保留，main未动。
运行信息: 本地9/9合成测试通过；python tools/collab_check.py --codex通过（0失败）；真实审计尚未启动，先在服务器跑相同测试再执行一次。
运行信息: 入口scripts/audit_d2_feasibility_v34.py；无训练、下载、GPU、曲线重新拟合；只写独立结果目录和锁。
关键数字: 文件名显示全场最新完整年为2022，实际审计先落盘登记；2022及以后SCADA只取哈希/表头。
关键数字: B固定K1b 2022 ALL 16949窗口；复算T21与持续法核对标签/队列，B1仍为0.75×T21 RMSE。
关键数字: A1/A2/A3/B1及O/P-curve指标待唯一真实实例；未声称通过。
需要 Claude 做的: 完成后复核审计实现、结果与R1-S1：O-curve为非因果诊断，不是所有分解模型的严格数学下限。
需要用户决定: 暂无；若缺数据或闸门未通过将停止本阶段并交接，不自行下载或训练。
