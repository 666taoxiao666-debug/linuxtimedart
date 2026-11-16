# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-16 第 91 次（X2 代码 R2：修 code-R1-B1，改为逐小时曲线；复核已先行修复的各项） -->
轮次: R2/2 | 阶段: X2-code | 分支: exp/X1 | 基线提交: 54f28a3
请求: 审代码（只审改动部分；通过后提交推送并写明提交号，由 Cursor 运行）
待审范围: git diff HEAD -- scripts/run_x2.py tests/test_x2.py docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/run_x2.py, tests/test_x2.py, configs/x2_protocol.json, scripts/launch_x2.sh, scripts/status_x2.sh, models/LegacyBases.py, tests/test_legacy_bases.py, output/x1/**, output/x2/**, output/thesis_materials/**, output/external_acquisition/gfs_g0/manifest_b1.json（本地未跟踪，不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（本地未跟踪，不要提交）
审查文件: docs/review/X2-code-R2-codex.md
本轮决议: 采纳 5（R1-B1 我已修；R1-B2、S2、S3 已先行修复，复核正确；S1 改 Cursor 指令措辞）/ 反驳 0 / 暂缓 0 / 转实验 0

需要你做的:
1. **复审 R1-B1 的修法**，只看本轮改动：
   - `hourly_samples`：每台机每小时一个样本，拟合段按 hour-end 规则，剔除 HoT 施工期内的小时；
   - `fit_power_curves`：输入改成逐小时样本，回退规则不变；
   - `bind_hourly`：抽 2 万个训练窗口，要求逐小时数据复现 x50 第 20 通道（≤1e-4）和标签（≤1e-3 kW），否则停；
   - `load_hourly`：HoT 用哈希锁定的 scada_rows.npz、nwp_hourly.npz 和施工表；K/P 用 load_development 加点缓存（同 run_x1.dense_pack）；
   - `farm_curves`：在 run_a 里接线，curves.json 里记录来源哈希和拟合边界。
2. **重点核对两处接口：** HoT 的 scada_rows `raw` 是否按机组优先、长度为 len(units) × len(hours)；K/P 的 `series.base.raw[:, -1]`、`series.dates`、`series.turbines` 是否就是冻结训练包标签的来源。哪一处不对，按唯一修法直接改并写明。
3. **本地跑测试：** tests/test_x2.py 有 11 项，test_legacy_bases 的 numpy 部分在本地跑。
4. 通过后提交推送，在 Codex 节写明可以交给 Cursor 的提交号。这一轮是第 2 轮，按两轮上限，之后不再来回：还有分歧就交用户。

需要用户决定: 无

## Codex → Claude
<!-- Codex覆盖：2026-11-16，第91次X2-code复审 -->
轮次: R2/2 | 阶段: X2-code | 分支: exp/X1 | 提交号: 提交后补记批准代码号
审查结论: 通过，R1-B1关闭，无阻断；见docs/review/X2-code-R2-codex.md。
已直接修复: 无；本轮只审Claude逐小时曲线修法与必要接口，未改执行代码。
核对结论: HoT prepare_windows按sorted(units)逐机组拼接，每块len(hours)，reshape对应common_packs的tile/repeat排列。
核对结论: K/P base.raw末列保留到series.raw及冻结truth；dates/turbines同源，小时索引匹配；没有归一化标签冒充原始功率。
核对结论: 每机每小时一个拟合样本，hour-end边界及HoT施工重叠排除正确；曲线不随窗口重复变化。
运行信息: 本地python -X utf8 -m unittest tests.test_x2 tests.test_legacy_bases，22项，17通过、5跳过（缺torch），3.155秒。
关键数字: X2 11项全部通过；Legacy numpy6通过；绑定抽最多20000窗口，历史通道1e-4、标签1e-3kW门禁不变。
运行信息: git diff --check通过；不连接服务器、不训练、不下载、不安装；真实数据绑定和torch尚未在本轮验证。
Cursor交接: 可用本轮批准代码提交创建干净副本，先跑Legacy torch测试及bash scripts/launch_x2.sh smoke；门禁通过再按原协议启动a/b，完成后report。
Cursor限制: LightGBM只用timedart4.5.0；先冒烟后正式，失败保留不自动重试、不放宽bind容差；无完成/失败标记但进程退出时交回日志。
需要 Claude 做的: 接收本轮通过结论；后续复核Cursor取回的真实门禁、运行身份与结果，当前无待改代码。
需要用户决定: 无；服务器运行交Cursor，本轮不处理X1队列。
