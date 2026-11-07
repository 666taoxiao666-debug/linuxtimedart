# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-07 第 46 次（L0-B r2：只读查询已启动的唯一实例并取回结果，不重启） -->
轮次: R2/2 | 阶段: L0 | 分支: exp/L0-B | 基线提交: 8494e52
请求: 只运行（只读查询 PID 508698 及其输出，取回获准文件；禁止再次启动 r2 或任何训练）
待审范围: git diff HEAD -- HANDOFF.md docs/review/DECISIONS.md
允许改动: output/hill_of_towie_e1/**, HANDOFF.md, docs/review/L0-R2-codex.md, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/L0-R2-codex.md（续写，不开第 3 轮）
本轮决议: 无新编号；R1-B2 的实现（5d2fa70 中的 conflict_missing、development_loader）已复核通过，本地脚本原始字节 SHA256 = ec2eb4ba…，与启动回执一致

已核实的记录:
- 脚本 scripts/audit_hill_of_towie_e1.py（ec2eb4ba…），测试 31976ba0…，代码提交 5d2fa70；服务器输出目录 <数据根>/outputs/audits/hill_of_towie_l0b_readonly_r2；本地目录 output/hill_of_towie_e1/l0b_r2/（目前只有 launch_receipt.json）。
- 缺的记录（这次补上）：完整命令行、cwd/worktree、日志路径、.launch_once 标记的位置、退出状态。
- 只读模式失败时不写 failed.json，只在日志里留 traceback；输出目录要到最后才创建。采集目录 hill_of_towie_e1_midas/ 里的 failed.json 是 11-05 认证失败留下的，与 r2 无关，不要误读。

需要你做的:
1. **只读查询**：ps 和 /proc/508698/cmdline，日志尾部，输出目录列表；.launch_once 的位置。如果标记在输出目录里，r2 会在开头报 "fresh separate output directory"，必须查清。
   - 运行中：只记录状态，不干预，交回"未完成"。
   - 完成：进程已退出，并且 readonly_inventory.json、qualified_stations.json 和 3 个 JSON 副本都在。
   - 失败：进程已退出，日志有 traceback，readonly_inventory.json 不存在。不重跑，交回错误类型。若是"超过 1%"，那是数据规则触发的停止，交用户。
2. **完成后核验并取回**到 output/hill_of_towie_e1/l0b_r2/：
   - 取回：readonly_inventory.json、qualified_stations.json、station_registration.json、download_plan.json、download_manifest.json、运行日志；
   - 另写一个 status.json：PID、命令、cwd、日志路径、退出状态、各文件 SHA256。
   - 核验：inventory 里记录的 qualified_stations_sha256 等于实际 SHA；3 个副本与 l0b_r1 的同名文件字节相同；unresolved_stations 为空；parsed 列表里没有 2024–2026 的路径；用逐站逐年完整率重算门槛。
   - r2 的结论追加到 output/hill_of_towie_e1/L0B_RESULT_SUMMARY.md，不新建报告。不拷观测数据。
3. **交回**：更新「Codex → Claude」，写明站 137 的逐年完整率、资格、气压，各站冲突时刻数，门槛结果，冻结合格站的列表和 SHA256；跑 collab_check --codex，提交并推送。

需要用户决定: 无（只有 r2 技术失败、需要重新启动时，才请用户授权）

## Codex → Claude
<!-- 由Codex覆盖。2026-11-07 第46次：只读查询两次未取得远端输出，未重启 -->
轮次: R2/2 | 阶段: L0-B | 分支: exp/L0-B | 提交号: 本轮见git log
审查结论: 第46次交接及DECISIONS增量通过；本轮仅查询既有实例，未改代码，见docs/review/L0-R2-codex.md。
已直接修复: 本轮为获批实现；只读scoped reader对同站同小时version_num=1冲突将风速/风向/气压全记缺测；一致重复由原加载器折叠。
运行信息: 任一站年冲突小时/完整公历年小时数>1%报错；平年8760、闰年8784，与完整率分母一致；原始文件及冻结screen/utils不改。
运行信息: 本地13单测通过，包含4项指定测试（冲突整小时缺测、一致重复保留一行、1%边界与超限报错、干净输入逐元素同原加载器）。
运行信息: 2026-11-07 13:34:07北京时间已启动唯一r2，PID508698；脱离SSH、.launch_once防重复标记已创建；本轮一次只读预算已使用，不重复启动。
运行信息: 新脚本/测试已传输且服务器SHA256与下列值一致，服务器13测试通过；后续多次只读状态查询SSH超时，完成/失败暂未确认。
关键数字: 脚本SHA256=ec2eb4ba69a6bb1f6dc1359c1aef44391b03869cc39a721bea8be4b08cf77e4f。
关键数字: 测试SHA256=31976ba0b7c97a839c7227697d644deaf2220b99947a3201a0e23b237fe54840。
运行信息: r2计划新目录outputs/audits/hill_of_towie_l0b_readonly_r2，本地output/hill_of_towie_e1/l0b_r2；将输出各站年冲突数、137完整率/资格/气压及qualified_stations.json SHA256。
运行信息: 本次两次SSH查询均未取得远端输出：Connection closed by 100.89.238.37 port 22；banner exchange Connection timed out。Tailscale经DERP(sfo)响应373ms，未建立直连。
运行信息: 本地终端输出正常；此前所谓窗口输出配额耗尽不是本轮已证实原因。当前PID是否存活、日志、cwd、完整命令、.launch_once位置及退出状态均待远端核实。
需要 Claude 做的: 复核本轮查询记录；结果仍未取回，不将r1的11站结果当作r2冻结列表，不把启动回执视为完成证明。
需要用户决定: 无新运行请求；待SSH恢复后按第46次清单继续只读取回核验。未写含推测退出状态的status.json，未重启。
边界: l0b_r1保留不动；不训练、不评估、不解封2024–2026，不修改下载逻辑；未合主线，main未动。
