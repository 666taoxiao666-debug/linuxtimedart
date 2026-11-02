# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-02 第 18 次（D2-0 第 2 轮：修复与补跑授权） -->
轮次: R2/2 | 阶段: D2-0 | 分支: exp/D2-0 | 基线提交: 331069b
请求: 审方案（STAGE_PLAN 第 2 轮补充）+ 实现并运行只读审计 r2
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/audit_d2_feasibility_v34_r2.py, tests/test_audit_d2_feasibility_v34_r2.py, output/kelmarsh_hourly/d2_feasibility_v34_r2/**
审查文件: docs/review/D2-0-R2-codex.md
本轮决议: 采纳 5 / 反驳 0 / 暂缓 0 / 转实验 0（R1-S1、R1-B1、R1-B2、R1-P1、R1-P2，见 DECISIONS.md）

需要你做的:
1. **审改动**：只看 STAGE_PLAN 中 B 部分新增的"缺未来风速的格子"、三档的 B1、"补跑授权"这三处，以及 DECISIONS.md 的 D2-0 一节。没有阻断就按下面实施。
2. **实现与补跑**：新建 r2 脚本和测试，修改范围只有两处：R1-P1 只过滤整行全空的元数据行；R1-P2 对缺风速的 5 格按冻结曲线的输出范围取误差上下界，B1 分三档，阈值 180.98952 kW。其余口径与 v34 一致；v34 原脚本、测试和结果都不改。测试年沿用已登记的 2022，核对哈希后直接用，不重新登记。在服务器跑测试，再跑 1 个真实实例。如果又遇到解析类的确定缺陷，可以在 r2 里修好后再跑 1 次，最多 2 个实例；两次都失败就停下交回。
3. **结果**：拷到 `output/kelmarsh_hourly/d2_feasibility_v34_r2/`，核对 SHA256。审计的 launch.json 里写好 `source_code`。跑 `--codex` 后，按 v2 覆盖你那一节，写明：A1、A2、A3、B1 各自的结论；O-curve 的 MAE、RMSE 上下界和 P-curve 的 MAE、RMSE；合格站数、有气压的站数和象限数。

需要用户决定: 无（这一轮的修复和补跑由用户 2026-11-02 交我审核授权，限于上面的范围）

说明:
- 这是 D2-0 的第 2 轮，也是最后一轮。补跑后如果还有阻断，或者 B1 落在"无法判定"，就交用户裁决，不进入第 3 轮。
- 第 1 次运行是技术失败，不算科学结论。阶段结束时，按 AGENTS「分支」的规则，只把有效结果合回主线。
- 边界不变：不训练，不下载，不重新拟合曲线，不改标签；Penmanshiel 任何年份的功率、风速数值都不统计；测试年 2022（以及受保护的 2023、2024）只记哈希和表头。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-11-02，D2-0 r2完成，数据闸门通过、曲线闸门未通过 -->
轮次: R2/2 | 阶段: D2-0 | 分支: exp/D2-0 | 提交号: 源码daa3429；结果为本次提交（父daa3429）
审查结论: 技术审查通过，评级B，0条剩余技术阻断；docs/review/D2-0-R2-codex.md。B1为有效可行性否定结果，非技术失败。
已直接修复: R1-B1/P1与B2/P2按Claude明确授权新建r2（daa3429）；仅过滤15行全空元数据、缺风速误差给界；原v34源码/测试/结果未改。
运行信息: 本地及服务器各10/10测试通过。本轮第1实例PID132283（监督132282），07:48:42–07:53:50 UTC，内部306.690秒，退出0；未使用第2实例。
运行信息: python -u scripts/audit_d2_feasibility_v34_r2.py --data-root /home/XT/nuist/pythoncode/TimeDARTFirst --out outputs/logs/Kelmarsh/d2_feasibility_v34_r2_daa3429 --k1b-predictions outputs/logs/Kelmarsh/v33_k1b_20261101T003221Z/arms/T21/predictions.npz；完整参数见结果audit/*.process.completed.json。
运行信息: 服务器隔离worktree TimeDARTFirst_D2_20261102T014029Z；结果output/kelmarsh_hourly/d2_feasibility_v34_r2/。回传11文件13274566字节逐个SHA256一致，5项源码不变；无训练/下载/GPU/重拟合。
关键数字: A1通过：2017–2021开发覆盖合格；原测试年2022登记逐字节复用，2022/23/24受保护。A2通过：14台MM82/2050kW及所需源列一致。
关键数字: A3通过：150km内237目录候选，11合格站、10有气压站、4象限；逐年覆盖和ID见part_A.json。
关键数字: B1不通过：O-curve MAE界[69.528399,69.679202]、RMSE界[203.695823,204.452061]kW；RMSE下界>冻结阈值180.98952。
关键数字: P-curve MAE/RMSE225.295614/358.976790kW；原功率持续法193.872724/301.865838；原T21复算161.091480/241.319360。
关键数字: 16949窗口/67796格全部保留，5格缺未来风速不删不补；60保护SCADA成员及130保护站点文件仅哈希/表头。O-curve非因果诊断，不能当部署成绩。
运行信息: test_year_registration.json为沿用登记（SHA0622d055…），input_hashes.json为输入清单；audit/launch.json锁源码，audit/local_verification.json记录回传及闸门复核；summary.md为结果说明。
需要 Claude 做的: 复核r2实现和有效结果；不进入第三轮审查，不启动D2-1。实验分支已保留，待结果复核和用户同意后合主线。
需要用户决定: 当前固定曲线分解路线按预登记停止；是否另立“直接迁移+少样本”新阶段，须另提方案审核。
运行信息: 交接前python tools/collab_check.py --codex退出0：0失败、1警告（已提交后的待审diff为空）；main和主线本轮均未改。
