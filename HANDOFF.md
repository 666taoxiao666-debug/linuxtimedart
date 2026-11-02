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
<!-- 由 Codex 覆盖。最后更新：2026-11-02，D2-0唯一实例技术失败，待R2决议 -->
轮次: R1/2 | 阶段: D2-0 | 分支: exp/D2-0 | 提交号: 源码2eba47e；结果为本次提交（父2eba47e）
审查结论: 运行前方案/代码通过；真实执行后2条阻断，评级C；docs/review/D2-0-R1-codex.md（R1-B1/B2）。不构成路线科学负结果。
已直接修复: 无；源码运行后按哈希保留；R1-P1/P2为待Claude审核、未实施的新版本修复/判定提案。
运行信息: K1b结题95dd54c已no-ff合入并推送主线038c5b1；exp/K1b保留。D2-0技术失败不合主线，main未动。
运行信息: 本地及服务器各9/9合成测试通过，未覆盖真实静态CSV的15行全空行；真实运行失败优先于测试通过。
运行信息: 唯一真实PID111916，01:42:03 UTC启动，耗时41.093秒后退出；A/B均UNASSESSABLE。未捕获OS退出码，不声称退出码0；无训练/下载/GPU/重拟合/重跑。
运行信息: 命令python -u scripts/audit_d2_feasibility_v34.py --data-root /home/XT/nuist/pythoncode/TimeDARTFirst --out outputs/logs/Kelmarsh/d2_feasibility_v34_20261102T014029Z --k1b-predictions outputs/logs/Kelmarsh/v33_k1b_20261101T003221Z/arms/T21/predictions.npz；完整绝对参数在结果audit/*.process.json。
运行信息: 服务器隔离工作区TimeDARTFirst_D2_20261102T014029Z；结果output/kelmarsh_hourly/d2_feasibility_v34/，回传8文件64122字节逐个SHA256一致，5项登记源码哈希未变。
关键数字: A1/A2/A3均无法判定：A在静态元数据29行（14设备+15全空行）解析时中断；Penmanshiel SCADA表头/正文与当地MIDAS均未打开，不能声称当地站不足。
关键数字: B1无法判定：16949固定窗口中4窗口/5个风机小时缺未来风速，h1–4为0/2/2/1；无删行或插补。O/P-curve MAE/RMSE均未产出，不填估计值。
关键数字: B触发缺风速保护前，已核对K1b starts及复算T21/持续法参照；T21既有MAE/RMSE仍161.091480/241.319360 kW。
关键数字: 测试年2022先按14台风机文件名落盘登记；2022/23/24保护；test_year_registration.json为登记，input_hashes.json为输入SHA清单，audit/launch.json仅锁源码。
关键数字: continue_D2=false仅表示无法验收，不表示分解路线已被证伪；详见summary.md和audit/diagnosis.json。
需要 Claude 做的: 按R2逐条决议R1-B1/B2、R1-P1/P2；明确新v34_r2文件允许范围及补跑授权。建议只剔除全空元数据行；B保原队列以冻结曲线输出范围计算5格缺风速误差上下界，不改标签或插补，界跨阈值则仍无法判定。
运行信息: 交接前python tools/collab_check.py --codex退出0：0失败、2警告（待Claude决议、待审diff空）；原运行日志及失败记录保留。
需要用户决定: 新判定口径/补跑须下一轮批准；本轮一次实例已用，不擅自扩大执行范围或启动D2-1。
