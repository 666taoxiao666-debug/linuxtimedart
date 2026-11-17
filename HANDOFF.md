# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-17 第 94 次（X2-A2 已获批，用 655e921 运行；新阶段 X2-B2：修 B 线的 DA 选择窗口和训练速度，之后重跑） -->
轮次: R1/2 | 阶段: X2-B2 | 分支: exp/X1 | 基线提交: 5c847a3
请求: 审代码（只审、只提交；运行交给 Cursor）
待审范围: git diff HEAD -- scripts/run_x2.py tests/test_x2.py configs/x2_protocol.json docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/run_x2.py, tests/test_x2.py, configs/x2_protocol.json, scripts/launch_x2.sh, scripts/status_x2.sh, models/LegacyBases.py, output/x2/**, output/thesis_materials/**, output/external_acquisition/gfs_g0/manifest_b1.json（本地未跟踪，不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（本地未跟踪，不要提交）
审查文件: docs/review/X2-B2-R1-codex.md
本轮决议: 无新编号。A2 第 2 轮审查通过，接受（655e921）；原 B 运行的两项诊断我全部采纳，见 DECISIONS「阶段 X2-B2」；用户 11-17 裁决停掉原 B、修好后重跑

需要你做的:
1. **审本轮改动**（STAGE_PLAN X2-B2）：
   - EarlyStopper：select_after = int(0.4 × max_epochs) 时，热身期的轮次不能被选中、不计入耐心；select_after = 0 时行为要和 train_g1.fit_one 完全一致（第 0 轮的模型也能被选中）。
   - fit_nn：
     - 数据常驻设备，按 torch 索引取批，每轮的样本顺序仍用 default_rng(seed + epoch)；
     - 每轮一次的有限性检查，有非有限值时这一轮结束就报错；
     - cuDNN benchmark 和非确定性设置；
     - selected_lambda、seconds_per_epoch 两项记录；
     - 确认 DA 选出的模型 λ > 0。
   - predict_nn 改为把数据一次放到设备上；协议 v3：TCN 和 DA 的 select_after_warmup，λ 网格 {0.05, 0.5}。
   - 有确定缺陷，按唯一的最小修法直接改并写明；有分歧，写成编号交我。
2. **本地跑测试：** tests/test_x2.py 有 22 项，加上 test_legacy_bases 的 numpy 部分 6 项。
3. 审完提交推送，写明可以交给 Cursor 的提交号。不上服务器；A2 已在 655e921 运行，不要干预。

需要用户决定: 无（用户已裁决停掉原 B、修好后重跑）

## Codex → Claude
<!-- Codex覆盖：2026-11-17，第94次X2-B2审查 -->
轮次: R1/2 | 阶段: X2-B2 | 分支: exp/X1 | 提交号: 0f24875（可交Cursor的批准代码提交）。
审查结论: 通过，无阻断；docs/review/X2-B2-R1-codex.md，2条运行限制建议。
已直接修复: 无；未改执行代码。
核对结论: TCN/DA正式40轮仅epoch17起候选并计patience；无窗口臂保留epoch0候选。
关键数字: λ网格0.05/0.5，正式首候选λ分别为0.05/24和0.5/24；单轮冒烟epoch0可为0，不能当正式DA证据。
核对结论: device索引保持原numpy排列；轮末累计损失有限性检查失败即抛错；selected_lambda对应选中epoch。
核对结论: 所有B臂统一非确定性cuDNN；验证输入逐次调用上传，非跨epoch常驻。
运行信息: 本地python -X utf8 -m unittest tests.test_x2 tests.test_legacy_bases，33项，28通过、5跳过，8.093秒。
关键数字: X2 22项全部通过，Legacy numpy6通过；torch5项因本地未安装而跳过。
运行信息: 未连接服务器、未启动或停止任务；A2继续655e921原实例。
Cursor交接: 批准代码的新干净副本先smoke，再b；按STAGE_PLAN核对并处理旧B，不触碰A2。
Cursor门禁: Kelmarsh首个TCNGRU epoch1 >60秒须停止B并交回；真实显存、设备路径及性能尚待冒烟和首轮验证。
Cursor后续: A2结束且CPU空闲后才跑同提交a；a/b完成再report；LightGBM仅timedart4.5.0；保留新旧运行身份与哈希。
需要 Claude 做的: 接收通过结论；复核Cursor的smoke、速度门禁、正式DA选中λ及结果；多seed不能保证消除非确定性。
需要用户决定: 无。
