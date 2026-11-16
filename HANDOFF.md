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
<!-- 由Codex覆盖。第86次X1：G2已合并，A0执行，A2缺测规则待决议 -->
轮次: R1/2 | 阶段: X1 | 分支: exp/X1 | 提交号: 文档86d3045，主线合并26c82c7，A0实现1d8479b
审查结论: A0通过；A2有1条阻断R1-B1，见docs/review/X1-R1-codex.md；A3/B按阶段暂缓。
已直接修复: 无；A2回退规则不唯一，未代改Claude方案。
合并: exp/G2已--no-ff合入codex/self-evolving-wiki并推送；从主线创建exp/X1，main未动。
运行信息: A0正式PID1166199，2026-11-16T00:25:15Z启动，四集COMPLETED、进程退出；CPU2核/2线程，无新增拟合。
服务器run: /home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/X1/x1_1d8479b_r1/A0
命令: scripts/run_x1.py --logs /home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs --output <run> --protocol-sha256 690397175db685e492c06cbaa85a72eb02e1cfe53dfa1177c2aceaf6b9cd0cc7
验证: 本地与服务器各3项单测通过；固定测试包SHA、父完成记录与N前三seed预测SHA均在执行入口核对。
结果: output/x1/A0/；服务器4个A0预测及results SHA复核通过，JSON/日志已取回。
关键数字: N较A0 MAE增益(kW)：Kelmarsh47.4118 CI[42.5247,51.7403]；Penmanshiel37.0850 CI[29.3309,44.3887]。
关键数字: HoT2024增益45.2094 CI[40.6728,49.8288]；HoT2025增益48.9252 CI[44.2101,53.7995]；四集判定通过。
关键数字: A2测试gF_mask[:,6:10]无效单元：Kelmarsh96、Penmanshiel624；不得将占位0当真实静风或删窗口。
依赖: timedart无lightgbm，A1未运行；Chronos-2等待Cursor回执，A3/B未运行；未下载或安装。
需要 Claude 做的: 决议R1-B1，规定A2拟合有效样本与缺测预测回退规则；复核A0实现及结果；后续依赖就绪再交回。
需要用户决定: 无；方案R1可由Claude在R2处理，无需用户重复授权。
