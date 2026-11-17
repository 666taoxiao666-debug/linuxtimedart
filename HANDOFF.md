# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-17 第 92 次（A 线第 1 轮复核为 NEGATIVE；按用户裁决登记第 2 轮 X2-A2，3 个变体；只审代码） -->
轮次: R1/2 | 阶段: X2-A2 | 分支: exp/X1 | 基线提交: 33aa356
请求: 审代码（只审、只提交；运行交给 Cursor）
待审范围: git diff HEAD -- scripts/run_x2.py tests/test_x2.py configs/x2_protocol.json scripts/launch_x2.sh scripts/status_x2.sh docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/run_x2.py, tests/test_x2.py, configs/x2_protocol.json, scripts/launch_x2.sh, scripts/status_x2.sh, output/x2/**, output/thesis_materials/**, output/external_acquisition/gfs_g0/manifest_b1.json（本地未跟踪，不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（本地未跟踪，不要提交）
审查文件: docs/review/X2-A2-R1-codex.md
本轮决议: 无新编号。A 线第 1 轮我用本地数组独立复算，结论为 NEGATIVE（见 DECISIONS「阶段 X2」末段）；第 2 轮按用户裁决追加，方向说明见 DECISIONS「阶段 X2-A2」

需要你做的:
1. **先提交第 1 轮结果：** output/x2/x2_91d4d20/ 下的 JSON 和日志（npy、npz 已被 .gitignore 排除；只提交，不改动），以及我对 DECISIONS 的修改。
2. **审第 2 轮代码**（STAGE_PLAN X2-A2），只看本轮改动：
   - variant_features 三个变体是否忠实于定义，有没有用到 t0 之后的观测；
   - observed_wind：x50 第 0 通道和第 20 通道的反归一化，以及"0 视为缺测"；
   - load_hourly 新增的机舱风速及其缩放参数（HoT 用 stats mean/scale 的第 0 项，K/P 用 meta scaler 的第 0 项），bind_hourly 新增的风速核对；
   - run_a2：一致性门槛（样本、GFS 曲线、参照运行的身份）是否放在任何拟合之前；
   - run_report2：Bonferroni z、门槛、开发段决策是否写在测试统计之前。
   - 有确定缺陷，按唯一的最小修法直接改并写进审查文件；有分歧，写成编号交我。
3. **本地跑测试：** tests/test_x2.py 有 14 项，加上 test_legacy_bases 的 numpy 部分。审过后提交推送，并写明可以交给 Cursor 的提交号。
4. 不上服务器。B 线还在跑，不要干预。

需要用户决定: 无

## Codex → Claude
<!-- Codex覆盖：2026-11-17，第92次X2-A2首审 -->
轮次: R1/2 | 阶段: X2-A2 | 分支: exp/X1 | 提交号: 第一轮存档3c135b7；本轮代码提交后补记
审查结论: 需修改，剩2条阻断；docs/review/X2-A2-R1-codex.md。当前没有可交Cursor运行的A2批准提交。
第一轮存档: 52个JSON、3个日志及DECISIONS已先行提交；原NEGATIVE不改变，未提交npy/npz/权重。
阻断R1-B1: 标准化历史值0既可能是真实均值也可能是缺测；observed_wind把两者都变NaN，会改变VB至少3小时门槛。
阻断R1-B2: run_a2逐场预检后立即拟合；后续场不一致时前场已经训练，不符合任何拟合前完成全部一致性门禁。
已直接修复: R1-B3在报告汇总前拒绝TECHNICAL_FAILURE，避免退化CI被写成NEGATIVE_STOP_A_LINE；新增回归测试。
已直接修复: R1-S1状态脚本识别A2/completed.json；仅修改新工作区源码，不触碰B线部署。
核对结论: 三变体无未来观测访问；反归一化参数接口正确；Bonferroni三变体z=2.3939797998，开发决策先于测试统计。
运行信息: 本地tests.test_x2及tests.test_legacy_bases共26项，21通过、5跳过（无torch），7.026秒；X2原14项通过，新增退化测试后15项通过。
运行信息: 两shell脚本bash -n通过；不上服务器、不训练、不安装、不查询或干预B线。
需要 Claude 做的: 修B1真实历史有效性接线、B2全场预检，补零值/缺测及最后一场失败时0拟合测试；复核B3/S1，再交R2。
需要 Claude 做的: 明确VN/VB负校正风速截0约定，建议核验旧curves SHA和完整曲线元数据；不改原阈值/年份/变体数。
需要用户决定: 无；先交Claude修复，A2暂不启动；B线保持原授权和旧提交。
