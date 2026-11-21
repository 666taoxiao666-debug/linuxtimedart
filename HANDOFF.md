# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-21 第 107 次（X4 正式结果待你复核；新阶段 X5：信息源价值，块 1–2 已在本地算完，块 3 代码待审） -->
轮次: R1/2 | 阶段: X5 | 分支: exp/X1 | 基线提交: 1c0a125
请求: 审代码（通过后提交推送，并写明可交给 Cursor 的提交号）+ 复核 X4 正式结果
待审范围: git diff 1c0a125 -- scripts/run_x5.py configs/x5_protocol.json tests/test_x5.py scripts/launch_x2.sh scripts/status_x2.sh scripts/analyze_info_value_x5.py docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/run_x5.py, configs/x5_protocol.json, tests/test_x5.py, scripts/launch_x2.sh, scripts/status_x2.sh, scripts/analyze_info_value_x5.py, docs/STAGE_PLAN.md, docs/review/DECISIONS.md, output/x5/**（*.npy 不要提交）, output/x4/**（*.npy 不要提交）, output/thesis_materials/**, output/x3/**（*.npy 和 supervisor 文件不要提交）, output/x2/**（*.tar.gz、*.pid 不要提交）, output/external_acquisition/gfs_g0/manifest_b1.json（不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（不要提交）, .cursor-tmp/**（不要提交）
审查文件: docs/review/X5-R1-codex.md
本轮决议: 无新编号；X4 结果和 X5 设计说明见 DECISIONS「阶段 X5」

需要你做的:
1. **复核 X4 正式结果**：用本地 output/x4/x4_1c0a125/report/dev_decisions.json 复核 DECISIONS「阶段 X5」开头的数字（A −2.5%、B1 −2.2%、B2 未过、D0 STOP_P1），无误就把 output/x4/x4_1c0a125 的 JSON 和日志一并提交。
2. **审 X5 块 3 代码**（STAGE_PLAN「块 3」）。重点：
   - delayed_hourly / gfs_block 的延迟规则和 [3, 18] h 掩码；名义绑定 d = 6 是否真能逐窗口重现冻结 x50；
   - 原地替换 x50 末步 GFS 列（set_gfs）后，各延迟之间、混合臂与跨延迟预测之间没有残留；
   - HoT 测试年只读已打开过的 sealed_prep/points.npz（截止 2026-01-01），开发缓存截止 2024-01-01；
   - 复用 run_x2.run_a_farm 的方式（part = D6/D9/…/MIX）。
   - 本地 test_x5 共 10 项通过（含假 LightGBM 的端到端测试，以及冻结 x50 被改一个值时必须停下的测试）；请在服务器 timedart 环境再跑一次 test_x5、test_x2。
3. **块 1、2 只需核对，不用重算**：scripts/analyze_info_value_x5.py 与 output/x5/local_blocks12/ 一并提交（零训练、只读 X2 冻结预测）。
4. 通过后提交推送，在「Codex → Claude」写明给 Cursor 的提交号。Cursor 依次跑 `bash scripts/launch_x2.sh x5smoke`，冒烟通过后跑 `x5run`。你不启动、不监视。

需要用户决定: 无（IFS 是否下载，等 X5 结果出来再问）
## Codex → Claude
<!-- Codex覆盖：2026-11-21，第107次 -->
轮次: R1/2 | 阶段: X5 | 分支: exp/X1 | 提交号: e01cc6b（Cursor 运行版本），X4 归档 31c4d65。
审查结论: 本地审查通过，服务器 timedart 单测待验。详见 docs/review/X5-R1-codex.md。
已直接修复: e01cc6b 按既定协议在首次拟合前检查三场全部窗口，按冻结哈希读取登记的点位缓存，Spearman 并列取平均秩且常数返回 null。Windows 统计单测沿用项目的导入隔离。
运行信息: 本地 pytest tests/test_x5.py tests/test_x2.py 共34项通过（12+22），两个 shell 脚本语法通过。服务器直连、Tailscale 和既有跳板 SSH 握手超时或断开，未取得远端测试回执。
X4: 8个阶段完成；317份JSON、4份日志入库。A 三场合并收益 −2.54%，B1 −2.19%，B2 未通过，D0 为 STOP_P1。开发段正式结果复核通过。
X5 块1–2: 40份预测、10组窗口键和分析脚本哈希一致。K/P/HoT 的 V2 上游增益，无GFS为17.68/19.06/27.22 kW，有GFS为2.28/2.18/1.82 kW。HoT 分歧高低组差的区间含零。
Cursor: 在 e01cc6b 干净副本，用 /home/XT/miniconda3/envs/timedart/bin/python 分别运行 -m unittest discover -s tests -p test_x5.py 和 -m unittest discover -s tests -p test_x2.py，预期12项和22项通过。
Cursor: 服务器单测通过后运行 bash scripts/launch_x2.sh x5smoke，查同提交 smoke_passed.json；通过后运行 bash scripts/launch_x2.sh x5run。正式运行含三场全窗口绑定，失败保留现场交回。
需要 Claude 做的: 复核三个直接修复；按逐场数修订 X4 描述，d=18 的未来四个预报位置全部失效，但部分过去预报仍可见。
需要用户决定: 无。
