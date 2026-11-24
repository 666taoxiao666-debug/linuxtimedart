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
<!-- 第 112 次交接，2026-11-24 -->
轮次: R2/2 | 阶段: wind-journal-baselines-code | 分支: exp/X1 | 提交号: 本条交接随审定代码同一提交；在该提交执行 git rev-parse HEAD 可取完整号。
审查结论: 通过，0 项阻断。Claude 已复核 B1–B3。B4 的实现符合登记的三场合计 15 h 估算规则。见 docs/review/wind-journal-baselines-code-R2-codex.md。
已直接修复: R1 的开发缓存边界、两模型实际输入一致性和 Cabin 单窗口验收纳入本次提交。R2 的 B4 由 Claude 实施。
运行信息: 本地 timedart 的 python -B -m unittest tests.test_wjb tests.test_x5 tests.test_x2 全部 81 项通过，零跳过，19.798 s；含真实 torch 的两项 Cabin 测试。
关键数字: 原反例 8.4 h 及两种顺序的 12 h 均跑完；18 h 的先快后慢样例在 5.5 h 触发；恰好 15 h 继续。跨场或时距的同候选耗时差异仍可能导致提前停止。回执解释为按登记估计超时。
交反重力: 在服务器干净 worktree 核对本次完整提交和本轮三个 Git blob，timedart 跑上述三组单测，通过后仅启动一次 bash scripts/launch_x2.sh wjbsmoke，按空闲资源设置线程与进程数。
回执位置: 数据根的 outputs/logs/WJB/wjb_<提交短号>/wjbsmoke.log、wjbsmoke.pid、smoke_report/completed.json；失败取相应 smoke_<mode>/failed.json 和日志尾，原目录保留，停止交回。
需要 Claude 做的: 待反重力交回服务器单测和冒烟报告，按完成标记核对；R2-S1 的说明可用于解释正式 TSG 的技术停止。
需要用户决定: 无。
