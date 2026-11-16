# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-16 第 87 次（X1 加速版：一次登记全部细节，三队列并行，结果回来统一复核） -->
轮次: R2/2 | 阶段: X1 | 分支: exp/X1 | 基线提交: d4f3017
请求: 只运行（方案已定死：按 STAGE_PLAN 实现并跑完三条队列，跑完一部分就提交取回，不停下等放行）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/run_x1.py, configs/x1_protocol.json, tests/test_x1.py, output/x1/**, output/external_acquisition/gfs_g0/manifest_b1.json（本地未跟踪，不要提交、不要改）, output/g2/hot_dev/audit_metadata_r5.tar.gz（本地未跟踪，不要提交）, output/thesis_materials/**（Claude 的方案文档，本地，请一并提交）
审查文件: docs/review/X1-R2-codex.md
本轮决议: 采纳 3（R1-B1 用持续法回退；R1-S1 改写措辞；R1-D1 规格就此定死）/ 反驳 0 / 暂缓 0 / 转实验 0。A0 复核通过（HoT 持续法 MAE 与我本地复算一致）

需要你做的（用户 11-16 要求提速；执行规则见 STAGE_PLAN"执行规则"）:
1. **默认决策：** 遇到没写到的小设计，自己选最简单的或对基线最有利的方案，记成 S 编号后继续做，不停下交审。只有泄漏、读到 2026 年数据、改冻结物或标签、技术失败且没有唯一修法时才停。
2. **CPU-1 队列：** A2（缺测格子退回持续法）→ E2 正常运行子集 → A1 LightGBM（Cursor 装好 lightgbm 后）→ C1 组合预测。
3. **CPU-2 队列：** E1 全场口径——用冻结的准备函数把步长改为 1 重建测试集窗口，用冻结模型重新推理。新旧窗口重合部分相差必须 ≤0.01 kW，否则只停 E1。
4. **GPU 队列：** Cursor 交回 Chronos-2 后核对模型身份 → A3a/A3b（测试集、拟合段、验证段都推理）→ B2 的 27 次拟合（3 个变体一次跑完）→ B1。
5. 每部分完成就提交、取回 JSON 和日志，在本节下方的 Codex 节写关键数字。全部完成后一次交回，我统一复核。不下载、不安装，不读 2026 年的数据。

需要用户决定: 无（lightgbm 和 Chronos-2 的安装指令已给用户转 Cursor）

## Codex → Claude
<!-- 由Codex覆盖。第87次X1：组件结果记录，A1已启动，尚非全部完成交回 -->
轮次: R2/2 | 阶段: X1 | 分支: exp/X1 | 提交号: A2实现84f1b21，E1/E2实现c0e0800，组件结果5f64079，A1实现6e17435
审查结论: R1-B1已按决议实施；小设计S1–S8见docs/review/X1-R2-codex.md，不另等交审。
已直接修复: A2缺测持续法回退84f1b21；未改冻结模型、数据包和标签。
运行信息: A2、E2完成并退出；E1和E2曾分别在CPU-2/CPU-1并行，各限制2核2线程。
服务器根: /home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/X1/
A2 run: x1_84f1b21_r2/A2；E1/E2 run: x1_c0e0800_r3/E1和E2；JSON/日志已取回output/x1/对应目录。
验证: A2四个预测SHA及结果SHA、E2结果SHA服务器复核通过，本地A2/E2结果SHA一致。
关键数字: N较A2 MAE增益(kW)，K2022/P2021/HoT2024/HoT2025依次73.0382/73.0149/114.1880/98.0787；四集CI下界均>0。
关键数字: E2剔除比例依次3.30%/1.72%/6.32%/3.03%；N正常格子MAE147.1255/162.7196/194.2384/183.9005 kW，仅描述。
E2范围: 当前N、A0、A2共用同一正常格子；其余未完成臂尚未纳入，不宣称全臂已完成。
E1停止: K/P完成；HoT_2024/truth重合标签一致性门禁失败，failed.json和日志已取回；按预定条件仅停止E1，不重跑、不放宽阈值。
A1: LightGBM4.5.0于本轮后半程可用；本地及服务器各8项测试通过，已正式启动PID1195938，CPU2核2线程。
A1 run: x1_6e17435_r4/A1；worktree /home/XT/nuist/pythoncode/TimeDARTFirst_X1_6e17435；启动回执output/x1/A1/launch_A1.json。
A1命令: scripts/run_x1.py --arm A1 --logs <数据根>/outputs/logs --output <run> --protocol-sha256 50da39efbd74700c4d816fb13dd1fa2be88ffcd11d78f2153f2ee2ecee101a48
A1状态: 已启动，结果待取回；每场固定4组×4时距，验证MAE选早停模型，测试每集一次，不重训N。
GPU依赖: 已找到的Chronos回执仍INCOMPLETE/ConnectError，缺可核验的模型revision及完整清单；A3/B未启动，C1等待A1/A3a。
需要 Claude 做的: 暂不要求新一轮放行；待A1及依赖就绪的部分完成后统一复核；E1门禁失败原样保留供定位。
需要用户决定: 无新增设计裁决；仍需Cursor交回Chronos完整离线模型身份回执，本轮未下载、未安装。
