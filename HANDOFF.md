# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-03 第 20 次（P1 结题 + P2 首轮） -->
轮次: R1/2 | 阶段: P2 | 分支: exp/P2 | 基线提交: e61e084
请求: 审方案（P2 STAGE_PLAN）+ 实现并运行训练（24 次拟合）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md tools/collab_check.py
允许改动: tools/collab_check.py, scripts/train_penmanshiel_p2_v35.py, tests/test_train_penmanshiel_p2_v35.py, output/penmanshiel_hourly/p2_v35/**
审查文件: docs/review/P2-R1-codex.md
本轮决议: P1 一节采纳 3 条（R1-S1、R1-B1、R1-P1），暂缓 1 条（R1-S2，转为 P2 的风险 1）；P1 的结果我已用本地副本复核

需要你做的:
1. **P1 结题**：请先在 exp/P1 上提交 DECISIONS.md、HANDOFF.md 和 tools/collab_check.py。collab_check 的改动是 R1-P1：check_frozen 两种 SHA 都认，自测 23/23。之后用户同意合并的话（见下方用户决定），用 `--no-ff` 把 exp/P1 合回主线并推送，exp/P1 保留。STAGE_PLAN 先不提交，带到 exp/P2。
2. **P2**：从主线拉 `exp/P2`，审 STAGE_PLAN（复现 R、零样本 Z1 和 Z2、少样本 F1 和 F3，配方沿用 v32）。没有阻断，就由你实现 `train_penmanshiel_p2_v35.py` 和测试，复用 v28、v32 的冻结函数。在服务器上先跑测试和 smoke，再跑 1 个真实实例。零样本一律用 Kelmarsh 拟合年的 scaler 和填补值。
3. **结果**：拷到 `output/penmanshiel_hourly/p2_v35/`，不拷权重和预测数组；核对 SHA256；launch.json 写 `source_code`。跑 `--codex` 之后，按 v2 覆盖你那一节，写明：R、Z1、Z2、F1、F3 各自的判定；各臂 2021 年的 MAE/RMSE（3 个 seed 的平均）；增益的 CI 和 Holm p。

需要用户决定: P1 合回主线（用户转话时会一并说明是否同意）

说明:
- 实现中遇到确定缺陷，可在允许改动的文件里直接修，修完后再跑 1 次，最多 2 个真实实例；两次都失败就停下交回。
- 只有 R 决定是否进入 P3；Z1、Z2、F1、F3 都是正负照记的开发证据。不追加臂，不调超参数，不改特征或站网。
- 边界：不碰 Penmanshiel 2022 年及以后的数据；K-* 只用 Kelmarsh 2016–2020 年训练。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-11-03，P1数据准备验收通过，未训练 -->
轮次: R1/2 | 阶段: P1 | 分支: exp/P1 | 提交号: 实现7f92979，修复c51baaf，结果ad48564；原始日志补入本次提交
审查结论: 通过，评级B，0条剩余阻断；docs/review/P1-R1-codex.md。P1只证明准备条件满足，不证明预测或迁移增益。
已直接修复: R1-B1（c51baaf）运行审计将动态模块相对文件名误识别为仓库源码；只修新增文件的路径解析，按第19次解析缺陷补跑授权实施。
运行信息: D2-0结题5d361cf，已用--no-ff合回主线c1a4ff9并推送，原分支保留；P1从该主线接续，main未动。
运行信息: 本地纯逻辑及服务器原环境各9/9测试通过；第1实例230898在读数据前失败，原日志/提交保留在结果attempt1/；第2实例231246（监督231245）退出0，357.875秒，无第3实例。
运行信息: 第2实例2026-11-03 02:31:52–02:37:54 UTC；隔离worktree /home/XT/nuist/pythoncode/TimeDARTFirst_P1_7f92979（运行HEAD c51baaf）。
运行信息: python -u scripts/prepare_penmanshiel_v35.py --data-root /home/XT/nuist/pythoncode/TimeDARTFirst --out /home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Penmanshiel/prep_v35_c51baaf_r2；完整命令见audit/*.process.completed.json。
关键数字: Q1通过：14台真实ID（无03）、21通道同序；起点2016-08-01，逐时664944行，上游47496行。
关键数字: Q2通过：train2016–19 / validation2020 / development_eval2021的ALL窗口64771 / 35603 / 38147（各≥5000）。
关键数字: Q3通过：2016–2021 upstream_missing分别9.5861% / 7.5228% / 13.4475% / 10.7078% / 10.3028% / 11.9863%，各≤20%。
关键数字: Q4通过：标签修改/截断/插补0格；原严格聚合、整行选择及21通道组装后标签一致；仅上游输入按训练年均值填补。
关键数字: 120输入文件大小/哈希符合D2登记；16项运行源码前后不变并与提交对象逐字节核对；回传清单内16文件209990字节SHA一致，清单自身SHA也已独立核对。
运行信息: 数据留服务器datasets/wind_public_v35/resampled_hourly_scada/penmanshiel_multiyear/；两CSV已复算SHA256，见audit/server_dataset_verification.json。本地只存output/penmanshiel_hourly/prep_v35/元数据。
运行信息: audit/launch.json有source_code；audit/verify_metadata.py独立复核闸门、源码、回传哈希，退出0；首次路径缺陷及本地CRLF/Git字节核验差异均有记录。未训练/下载，2022+仅验字节，不读表头。
需要 Claude 做的: 复核P1实现、R1-B1修复及结果，并决议R1-P1检查器修复提案；确认后另写P2方案。P1未合主线，不启动P2/P3。
需要用户决定: --codex退出1：检查器sha_lf误报历史CRLF文件（原始字节与Git对象均一致）；tools/collab_check.py超出本轮允许改动，已提R1-P1待Claude批准，未擅改；P1合并仍须复核和用户同意。
