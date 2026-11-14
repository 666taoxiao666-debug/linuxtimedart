# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-13 第 78 次（R2-E1：导入最小修复已批准；预检后从头完整准备，门禁都过就跑 60 次拟合） -->
轮次: R2/2 | 阶段: G1c | 分支: exp/G1 | 基线提交: cf9c9db
请求: 修复后运行（执行失败的恢复，不是新的方案审查轮）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/prepare_g1_batches.py, tests/test_g1_pipeline.py, scripts/train_g1.py, output/g1/**, output/gfs_g0/**, output/external_acquisition/gfs_g0/**
审查文件: docs/review/G1c-R2-codex.md（续写一节"第 78 次执行"）
本轮决议: 采纳 1（R2-E1，最小导入修复）；R1-B1/B2 的处理复核通过。恢复范围见 DECISIONS G1c 末段

需要你做的:
1. **修复：** scripts/prepare_g1_batches.py:314 改为 `from scripts.screen_kelmarsh_upstream_observations import read_badc_csv, column`，加一个导入回归测试，提交。configs/g1_protocol.json 不动，PROTOCOL_SHA256 不变；新的准备脚本 SHA 写进启动回执。
2. **真实适配预检**（只读、不训练、不读 GFS）：跑 Penmanshiel 的 load_development 元数据部分，核对注册表哈希、11 个站坐标唯一、MIDAS 文件登记齐全、distance_km/bearing_deg 都在。
   - 再遇到同类导入或适配的确定缺陷：修法唯一、且不改数据/年份/阈值/臂/模型的，可以按最小修法先行直接修，最多 2 处，写进审查文件；超过 2 处或修法不唯一就停下交回。
3. **从头完整准备：** 新建运行目录，两个风场都重新准备，不复用失败那次的 Kelmarsh；失败现场 g1c_fd3ee91_r1 原样保留。
   - 新旧 Kelmarsh 产物逐数组（np.array_equal）和 JSON 比较，不比 npz 文件哈希（zip 含写入时间）；不一致就停下交回。
4. **门禁都过就直接跑 60 次拟合**（授权同第 76、77 次，正式拟合目前 0/60）：先冻结后评估，结果拷回 output/g1/。交回 Q1/Q3/Q4/Q5 两场的增益、CI、p、max-p、Holm 结论，I* 选了谁，两场的校准方法统计，机制诊断。任何一步失败就停下交回。不下载、不安装；manifest_b1.json 原文件不提交。

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
