# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

长期执行分工（用户于 2026-11-24 指定）：本项目原交给 Cursor 的服务器调用统一交给反重力（Antigravity），包括环境核验、已批准的下载、运行、状态查询和结果取回。交接写明已审提交、可执行步骤、回执位置和停止条件。历史路径里的 Cursor 字样按原名称使用。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-24 第 112 次（wind-journal-baselines 代码 R2：复核 B1–B3，实施 B4） -->
轮次: R2/2 | 阶段: wind-journal-baselines-code | 分支: exp/X1 | 基线提交: db32df0（本轮和 R1 的改动都在工作区，未提交）
请求: 审代码后运行（R2 只审 B4 的改动；通过后提交，交反重力先跑冒烟）
待审范围: git diff db32df0 -- docs/STAGE_PLAN.md docs/review/DECISIONS.md；新文件只看 B4 部分：scripts/run_wjb.py 的 mark_started、remaining_lower_bound、check_time_limit、run_pool、tsg_fit_job（started.json、wall_seconds）、mode_tsg（kind = 候选名）；configs/wjb_protocol.json 的 tsg.time_limit_scope、time_limit_rule；tests/test_wjb.py 的 TSGTimeLimit
允许改动: scripts/run_wjb.py, configs/wjb_protocol.json, tests/test_wjb.py, models/vendor_cabin/**, models/vendor_tsg/**, scripts/launch_x2.sh, scripts/status_x2.sh, docs/STAGE_PLAN.md, docs/review/DECISIONS.md, docs/review/wind-journal-baselines-code-R1-codex.md, docs/review/wind-journal-baselines-code-R2-codex.md, output/wjb/**（*.npy、*.npz、*.pkl、*.pt、*.pid 不要提交）, output/x5/**（*.npy、*.npz、*.pid 不要提交）, output/thesis_materials/**, output/x4/**（*.npy 不要提交）, output/x3/**（*.npy、supervisor 文件不要提交）, output/x2/**（*.tar.gz、*.pid 不要提交）, output/external_acquisition/gfs_g0/manifest_b1.json（不要提交）, output/g2/hot_dev/audit_metadata_r5.tar.gz（不要提交）, .cursor-tmp/**（不要提交）
审查文件: docs/review/wind-journal-baselines-code-R2-codex.md
本轮决议: 采纳 4（R1-B1–B3 复核通过、R1-B4 由 Claude 实施）/ 反驳 0 / 暂缓 0 / 转实验 0，见 DECISIONS「阶段 wind-journal-baselines-code」末尾的表

B4 口径: 三场合计 15 h（本组上限 5 h × 3；三场任务相同，等价于每场 5 h），见 STAGE_PLAN 第 98–101 行。
B4 规则: 每完成一棵树，判"已用时间 + 剩余用时下限 > 15 h"。
- 剩余下限 = max(剩余工作量 / 进程数, 最长的剩余树)。
- 每棵未完成的树按同一候选已完成树的平均用时计，扣掉它已跑的时间；还没有完成树的候选计 0。
- 判定用的是下限，只有乐观估计也超时才停。
B4 测试: 你的反例（8.4 h 的批次）全程不停；96 × 1.5 h 在 1.5 h 时停；先快后慢的 18 h 批次在第一棵慢树完成后停；两种派发顺序的 12 h 批次都不停；run_pool 用线程池原样接线测试。
本地: test_wjb、test_x5、test_x2 共 81 项，79 项通过，2 项因无 torch 跳过；工作区 blob：run_wjb.py f4abab06、test_wjb.py 1d5383e5、wjb_protocol.json f5ff45b5。
需要你做的:
1. 审 B4 的实现和测试，并在 timedart 环境重跑 tests.test_wjb、tests.test_x5、tests.test_x2（含 torch 的 2 项）。
2. 通过后提交推送（R1 的 B1–B3 修复和本轮 B4 一起），交反重力：核对审定提交，在服务器跑同样三组单测，再 `bash scripts/launch_x2.sh wjbsmoke`。
3. smoke_report 为 COMPLETED 后，同时起 wjbcpu 和 wjbgpu，两条都结束后起 wjbfinal。任何 FAILED 就停下交回；TSG 写出 technical_limit.json 属于登记过的停止，照常起 wjbfinal。环境变量 X2_THREADS、WJB_WORKERS、WJB_WORKER_THREADS 按服务器空闲核数设置。
需要用户决定: 无

## Codex → Claude
<!-- 第 112 次交接：服务器冒烟结果复核，2026-11-24 -->
轮次: R2/2 结果 | 阶段: wind-journal-baselines-code | 分支: exp/X1 | 提交号: 1fe7ab6ff37a1e21cbd8d1895d3fc09140f4686f（服务器审定运行版本）。
审查结论: 三场期刊对照的 wjbsmoke 已验收。KRR、LightGBM-hist、TSG、Cabin、report 五步的完成回执、脚本、协议、作者 blob 和 registry 均与审定版本及登记摘要一致，日志以 [WJB] chain done 结束。
已直接修复: 本轮仅归档冒烟回执；B1–B4 修复及审查见运行提交与 docs/review/wind-journal-baselines-code-R2-codex.md。
运行信息: 反重力报告服务器 timedart 的三组单测 81/81 通过、零跳过、7.592 s。运行 worktree 为 /home/XT/nuist/pythoncode/TimeDARTFirst_WJB_1fe7ab6。冒烟外层 PID 3603829 来自反重力回执，报告 PID 3604192 已核对 launch.json。
运行信息: 归档 output/wjb/wjb_1fe7ab6/ 下的 26 份 JSON、SUMMARY.md 和 wjbsmoke.log。逐步核对完成状态、所有 registry 文件及四份 v1_freeze 的 SHA256；逐预测数组留在服务器。
关键数字: Cabin 与 LightGBM-hist 的 18 组实际输入摘要全部相等。6 个 Cabin 候选最大单窗差为 8.94e-8（容差 1e-5），打开 sample softmax 的反向对照为 0.0052000。TSG 的三场两臂列映射和四种受控路由均通过。
关键数字: smoke_report/completed.json 的 SHA256 为 f43525b4d248549de72a781c436266b142962e5b328c33a7e20c981c682de129，results.json 为 7943bf2698ca24600e52496ceaab49796d0a04b5fba7899417297e7c6a216de2。
交反重力: 正式链沿用原 worktree 和运行提交 1fe7ab6。核对既有锁和进程，按当前空闲资源并行运行 bash scripts/launch_x2.sh wjbcpu 和 wjbgpu，各一次。两条结束后按协议起 wjbfinal；仅登记的 TSG TechnicalLimit 带 technical_limit.json 可继续汇总，其他失败停下交回。
需要 Claude 做的: 正式结果交回后再复核论文比较。服务器冒烟完成；截至交接仅有冒烟回执，正式启动待反重力确认。Codex 的反重力 CLI 缺少会话地址，桌面工具未获反重力访问权限。
需要用户决定: 无。
