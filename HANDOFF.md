# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-02 第 19 次（D2-0 结题 + P1 首轮） -->
轮次: R1/2 | 阶段: P1 | 分支: exp/P1 | 基线提交: 98d39bf
请求: 审方案（P1 STAGE_PLAN）+ 实现并运行数据准备（不训练）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/prepare_penmanshiel_v35.py, utils/penmanshiel_v35.py, tests/test_prepare_penmanshiel_v35.py, output/penmanshiel_hourly/prep_v35/**
审查文件: docs/review/P1-R1-codex.md
本轮决议: D2-0 的 R2 一节没有新编号；我用本地副本独立复核了 A1、A2、A3 和 B1，结论与你一致（见 DECISIONS.md 的 D2-0 一节）

需要你做的:
1. **D2-0 结题**：用户 2026-11-02 同意合并。请先在 exp/D2-0 上提交本轮的 DECISIONS.md 和 HANDOFF.md，再用 `--no-ff` 把 exp/D2-0 合回主线 `codex/self-evolving-wiki` 并推送，exp/D2-0 保留。STAGE_PLAN 先不提交，带到 exp/P1。
2. **P1**：从主线拉 `exp/P1`，审 STAGE_PLAN。没有阻断，就由你实现 `prepare_penmanshiel_v35.py`（另加可选的 utils 文件和测试）。只做参数化，复用 Kelmarsh v31/v32 的冻结函数；改了哪些 Kelmarsh 专用常量，要逐个列在审查文件里。服务器上先跑测试，再跑 1 个准备实例；2022 年起的内容只记哈希。
3. **结果**：把元数据拷到 `output/penmanshiel_hourly/prep_v35/`，核对 SHA256。审计的 launch.json 里写好 `source_code`。跑 `--codex` 之后，按 v2 覆盖你那一节，写明：Q1–Q4 各自是否通过；起始日期；三段的窗口数；每年的 `upstream_missing` 占比。

需要用户决定: 无（"第二风场复现"方向和合并 D2-0 已由用户 2026-11-02 选定）

说明:
- 用户选的方向是：在 Penmanshiel 上复现已确认的上游特征增益，再做 Kelmarsh → Penmanshiel 迁移。P1 只准备数据；P2（开发折上的复现与迁移）和 P3（2022 年一次性测试）的细节暂缓，本轮不用审。
- 如果准备过程中出现解析类的确定缺陷，你可以在允许改动的文件里直接修，修完后再跑 1 次，最多 2 个实例；两次都失败就停下交回。
- 边界：不训练，不下载，不改 Kelmarsh 的任何冻结代码或数据；Penmanshiel 和站点 2022 年起的内容一律不解析。

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
