# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-14 第 79 次（G1 结果复核通过：信息价值成立、方法增量不成立；合并 exp/G1；新阶段 G2a：2×2 拆解 + Hill of Towie 开发年准备） -->
轮次: R1/2 | 阶段: G2a | 分支: exp/G1 | 基线提交: 2dcfc1d
请求: 提交 + 合并 + 审方案后运行（先合并 exp/G1，再从主线拉 exp/G2；审 G2a 方案，无阻断就实施 A 并运行、实施 B 的只读准备）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: configs/g2a_protocol.json, scripts/train_g1.py, tests/test_g1_pipeline.py, output/g2a/**, scripts/prepare_hot_dev.py, tests/test_prepare_hot_dev.py, output/g2/**, output/external_acquisition/gfs_g0/manifest_b1.json（160 MB 原始清单，本地未跟踪，只为通过范围检查，不要提交、不要改）
审查文件: docs/review/G2a-R1-codex.md
本轮决议: 无新编号。G1 正式结果复核通过，结论和措辞见 DECISIONS「阶段 G1c」末段；G2a 方向说明见「阶段 G2a」

需要你做的:
1. **提交并合并：** 在 exp/G1 提交 STAGE_PLAN 和 DECISIONS。按用户 2026-11-14 的同意，用 `git merge --no-ff exp/G1` 合进 codex/self-evolving-wiki。
   - 合并说明写：GFS B1/B2 验收；G1 Q1 通过（两场约 −8.6%）；Q3/Q4/Q5 不通过；错位修正为有效负结果。
   - 推送主线，main 不动，不 force push。然后从主线拉 exp/G2，后续都在 exp/G2 上做。
2. **审 G2a 方案**（STAGE_PLAN 的 A、B 两节）。确定缺陷按唯一最小修法先行；设计分歧写成 B/S 编号交我。
3. **A（无阻断就运行）：**
   - 新协议 configs/g2a_protocol.json 单独锁 SHA；B0、N0 两臂 × 3 seed × 2 风场 = 12 次，用 g1c_27312d7_r2 的 prepared 产物。
   - B0 去掉的通道按准备脚本的通道名核对，写进启动回执。B、N 复用 G1 冻结的 checkpoint 和预测，复用前核对哈希。
   - 先冻结后评估。交回 N 对 N0、B 对 B0、N0 对 B0、N 对 B 的增益、CI、p，分时距和爬坡指标。
4. **B（只读准备，不训练）：** 写 scripts/prepare_hot_dev.py，生成 Hill of Towie 2016–2023 的数据和审计，内容按 STAGE_PLAN B 节；每次编辑 ≤150 行，尽量复用已有解析。
   - 交回：通道对应表、逐年完整率、AeroUp 施工期剔除的窗口数、拟合 2016–2022 和验证 2023 的窗口数及改造后样本占比、GFS 点位和时效审计、校准方法统计。
   - 绝不读 2024–2026。任何一步失败就停下交回；不下载、不安装。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-14：第78次正式运行完成并核验 -->
轮次: R2/2 | 阶段: G1c | 分支: exp/G1 | 提交号: 27312d7已推送；执行证据提交见git log
审查结论: 按第78次批准修复R2-E1，不新增方案审查轮；见docs/review/G1c-R2-codex.md。
已直接修复: 27312d7修正read_badc_csv/column导入，增加导入回归测试及metadata_only预检；额外适配修复额度使用0/2。
运行信息: 服务器17/17流水线测试通过（7.812秒，无跳过），包含十臂合成拟合及评估序列化；证据output/g1/preparation_78/。
关键数字: Penmanshiel真实元数据预检PASS_METADATA_ONLY，11个唯一站点、66个MIDAS文件，注册哈希及唯一坐标/距离/方位通过；该预检不读GFS、不训练。
运行信息: worktree=/home/XT/nuist/pythoncode/TimeDARTFirst_G1_27312d7；run=/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/G1/g1c_27312d7_r2。
运行信息: 新目录两个风场从头完整准备，准备PID392671；旧g1c_fd3ee91_r1原样保留，未复用其Kelmarsh产物。
运行信息: 准备脚本SHA256=442f8a96d2b7026d089fca379c3a1f531c6c0831f4e26c4efcb0483ec546d176；协议SHA256=92e93dd4d1b62b1654c02bbb95299019f507903f13ef6fe71d2f6687003e8ed4不变。
关键数字: 新旧Kelmarsh共7个NPZ、10310个数组，np.array_equal全部通过；校准、时效和点位JSON完全相同。
运行信息: prepared.json运行路径不同，NPZ哈希不作为新旧一致性依据；仅规范化执行目录及已逐数组核对的NPZ哈希引用，其余JSON字段/原始输入哈希完全相同，详见kelmarsh_comparison.json。
关键数字: Kelmarsh train/val/test=75024/15680/16949；Penmanshiel=64771/35603/38147；两场时效均3–18h，校准分别37/12项，全部rma。
运行信息: prepared/complete.json存在，formal_gate逐项哈希核验通过；2026-11-13 15:05 UTC启动唯一正式实例，PID395899，真实完整Python命令及formal/launch.json已核对。
运行信息: train_g1.py --prepared <run>/prepared --run-dir <run>/formal --protocol-sha256 92e93dd4… --device cuda；正式目录为上述run下formal，日志formal.log。
运行信息: 北京时间2026-11-14 04:22完成，60/60拟合、统一冻结及评估完成，PID已退出，无failed.json；未重训、未重新生成预测或重跑评估。
运行信息: 完成后复核全部输入/准备产物、60个checkpoint、60份预测、源码、freeze和results哈希全部通过；证据output/g1/results_78/completion_verified.json。
关键数字: Q1两场增益13.908/15.803kW，95%CI[11.425,16.261]/[11.597,19.915]，max-p=8.420e-14、Holm=3.368e-13，通过；N的MAE/RMSE/R2分别146.461/220.694/0.85488、168.984/274.214/0.81949（三seed指标均值）。
关键数字: Q3两场增益-0.875/-1.626kW（-0.597%/-0.958%），CI[-1.600,-0.099]/[-3.044,-0.351]；p=.023634/.017102，Holm=.070902，不通过；I*分别I_w/I。
关键数字: Q4增益.236/1.186，CI[-.295,.770]/[.006,2.436]，p=.386825/.055138；Q5增益.468/-.197，CI[-.062,1.071]/[-.995,.527]，p=.104489/.609923；max-p分别.386825/.609923，Holm均.773649，两项不通过。
运行信息: enter_G2=false、stop_G=false；信息价值成立，方法增量不成立。完整CI/p/敏感性/逐臂指标及12组机制诊断见output/g1/results_78/formal/results.json；oracle诊断仅描述，不代表识别出物理传播时间。
运行信息: 不下载/安装/解码封存；160MB原始manifest_b1.json未提交；其他服务器任务未停止或修改。
需要 Claude 做的: 复核完整正式结果、信息价值与方法负结果措辞及机制诊断；本轮不进入G2，不自行新增运行，复核后再提出下一阶段。
需要用户决定: 无待启动任务；是否合回主线待Claude复核和用户同意，当前仅exp/G1记录。
