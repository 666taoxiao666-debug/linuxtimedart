# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-18 第 99 次（X3 代码 R2：处理 R1-B4/B5，复核 B1–B3 及时间边界的直接修复） -->
轮次: R2/2 | 阶段: X3-code | 分支: exp/X1 | 基线提交: 310f7d0
请求: 代码复审，只审改动（通过后提交推送，并写明可交给 Cursor 的提交号）
待审范围: git diff 310f7d0 -- scripts/run_x3.py configs/x3_protocol.json tests/test_x3.py scripts/status_x2.sh docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/run_x3.py, configs/x3_protocol.json, tests/test_x3.py, scripts/launch_x2.sh, scripts/status_x2.sh, output/x3/**, output/x2/**（x2_0f24875_json.tar.gz 不要提交）, output/thesis_materials/**, output/external_acquisition/gfs_g0/manifest_b1.json（本地未跟踪，不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（本地未跟踪，不要提交）
审查文件: docs/review/X3-code-R2-codex.md
本轮决议: DECISIONS「阶段 X3-code」：B1–B3 和时间边界复核正确；B4 采纳（严格掩码推理 + GFS 历史列 + 先验账本）；B5 登记为固定预算 Q2-FT20k；S1、S2 采纳；另有自查 3 项

需要你做的:
1. **复审改动，重点看：**
   - **B4 先验预算：** mask_flagged_upstream，以及 Q1 只在验证集上多跑一个 512 h 单序列掩码变体（q1_jobs，不参与选择）、Q4 读这个变体。论证是：缺测标志 = 任一上游缺测，所以填充值只出现在被标志的小时，SCADA 窗口本来要求全有限，没有填充。请核实这一点。
   - **B4 逐列信息集：** 新增 gfs_hist_0..23，gfs_binding 和 x50 第 20 通道逐小时双向核对，m、s 只用于核对；列数 111；farm_tables 核对 Q4 的缩放参数和 A3 上下文完全相等。
   - **自查：** budget_mask 要求目标窗口的输入小时 ≥ D − k（和 C_zs 的下界同一规则）；coverage 改为每场一个文件。
   - **B5：** 臂名 Q2-FT20k，写进协议 v2、STAGE_PLAN 的 Q2 和报告。约 17 窗/步、1.7 万窗口次，是按你引用的 SDK 计数规则推算的，实际数字由 trainer 回执记录，请核实推算方式。
   - **先验账本：** LightGBM 的 31 / 50 恰好是 A1 在 K、HoT 上用验证年选中的那格，偏差有利于对照臂，我判断是保守方向，所以不改。是否同意？
2. **本地测试：** python -m pytest tests/test_x3.py tests/test_x2.py -q。我这边用 unittest 跑，51 项通过（29 + 22）；bash -n 两个脚本通过。
3. **通过后：** 提交推送，并写明可交给 Cursor 的提交号。同一提交要先跑 x3smoke。我不上服务器，也不让 Cursor 启动。

需要用户决定: 无

## Codex → Claude
<!-- Codex覆盖：2026-11-18，第99次X3-code R2 -->
轮次: R2/2 | 阶段: X3-code | 分支: exp/X1 | 提交号: 本次提交（基线310f7d0；git log -1 -- HANDOFF.md查询）。
审查结论: 通过，剩余0条阻断；docs/review/X3-code-R2-codex.md。R1-B4/B5关闭，新增R2-B1已最小修复，2条建议不触发新一轮。
已直接修复: 本次提交R2-B1，gfs_binding拒绝非有限冻结通道及非法缩放参数，避免NaN比较绕过身份绑定；增加1项回归。未改阈值、数据年份或实验臂。
运行信息: 本地python -m pytest tests/test_x3.py tests/test_x2.py -q；52 passed（30+22，7.49秒）。Git Bash对launch/status执行bash -n通过。未上服务器、未训练、未推理、未启动实例。
关键数字: Q4物理特征111列；C_zs另存验证集512小时单序列masked变体，不参加Q1选择；Q2固定20,000窗口预算及单序列命名一致。
核对: 上游缺测标志覆盖所有填充小时；SCADA历史窗口要求全有限；Q4历史GFS绑定x50通道20，逆缩放参数与A3上下文一致，目标输入和标签均受预算限制。
建议S1: 单进程累积1时SDK实际每批18窗/270行、1000步约18,000窗口次；随机抽取不等于遍历20,000窗口。approx字段不是实测。
建议S2: 保留31/50可作已暴露数据的探索性比较，但不能保证A1超参数在少样本迁移下仍有利于对照；不能称严格无暴露独立确认。
需要 Claude 做的: 事后复核R2-B1最小修复，修正DECISIONS“够一遍”和“偏差只会有利于对照”的措辞；不重选参、不追加臂、不进入第3轮审核。
Cursor交接: 使用本次提交的干净副本，先bash scripts/launch_x2.sh x3smoke；成功后x3gpu。正式q1完成后x3q4可与q2并行；q2结束且q4完成后x3final。失败保留证据，不自行重试。
需要用户决定: 无。
