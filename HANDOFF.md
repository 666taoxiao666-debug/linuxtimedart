# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-03 第 22 次（P2 结题 + P3 首轮） -->
轮次: R1/2 | 阶段: P3 | 分支: exp/P3 | 基线提交: 850f4fd
请求: 审方案（P3 STAGE_PLAN，三折滚动）+ 冻结协议、解封 2022–2024 并运行
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: configs/penmanshiel_p3_protocol_v35.json, scripts/train_penmanshiel_p3_v35.py, scripts/prepare_penmanshiel_v35_p3.py, tests/test_train_penmanshiel_p3_v35.py, output/penmanshiel_hourly/p3_v35/**
审查文件: docs/review/P3-R1-codex.md
本轮决议: P2 的 R2 一节没有新编号；P2 结果我已独立复核，与你一致（见 DECISIONS.md 的 P2 一节）

需要你做的:
1. **P2 结题**：先在 exp/P2 上提交本轮的 DECISIONS.md 和 HANDOFF.md。用户同意的话（见下方），用 `--no-ff` 把 exp/P2 合回主线并推送，exp/P2 保留。STAGE_PLAN 先不提交，带到 exp/P3。
2. **P3**：从主线拉 `exp/P3`，审 STAGE_PLAN。没有阻断，就严格按"执行顺序"一步一步做：
   - 先写协议 JSON，用 `PROTOCOL_SHA256` 锁住，单独提交并推送；
   - 解封核对：2022–2024 的 SHA 与 D2-0 的登记一致，2016–2021 的每一行与 P1 产物完全相同，并按窗口门槛定好 T23、T24 的角色；
   - 完成 3 折共 72 次拟合，全部冻结，runner 要支持断点续跑；
   - 只评估一次。

   确定缺陷可以按 AGENTS 的提速规则处理。2022 的指标一旦算出，不补跑、不改任何东西。
3. **结果**：拷到 `output/penmanshiel_hourly/p3_v35/`，不拷权重和预测数组；核对 SHA256；launch.json 写 `source_code`。跑 `--codex` 之后，按 v2 覆盖你那一节，写明：T22、T23 各自的 R、F1、F3 是否获确认（含效应、CI、Holm p）；R 的总体表述；T24 和 Z1、Z2 的描述性结果；各折各臂的 MAE/RMSE/R²。

需要用户决定: 合并 P2，并启用 2022–2024 做 P3（用户转话时一并说明）

说明:
- 按用户 2026-11-03 的意思，多次实验合并一次跑：三折滚动（T22、T23 做确认，T24 只描述）。2022–2024 是 Penmanshiel 剩下的全部年份，只用这一次；解封前如果发现任何问题，停下交用户。
- 这是 P3 的第 1 轮。方案如果有阻断，而且修法不唯一，就先停在第 1 步之前交回，此时这些年份都还没有解封。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-11-04，P2合并完成；P3方案首审完成，年份范围待确认 -->
轮次: R1/2 | 阶段: P3 | 分支: exp/P3 | 提交号: P2复核2682f6c；主线合并0e2e939；P3审查见本次提交
审查结论: 方案通过，评级B，0条技术阻断、3条非阻断建议；docs/review/P3-R1-codex.md。方案通过不等于新runner通过测试。
已直接修复: 无；未改Claude方案或决议，只保存其原始P3方案与本轮审查记录。
运行信息: 已按用户2026-11-04批准，git merge --no-ff exp/P2合回codex/self-evolving-wiki并推送；exp/P2保留，main未动。
运行信息: exp/P3从合并主线0e2e939建立，P3 STAGE_PLAN从原工作区带入此分支，未混入主线P2结题提交。
运行信息: P3未冻结协议、未实现/部署runner、未解封数值、未训练；本轮只读现有登记元数据和待审diff。
关键数字: P2已由Claude独立复核，R/F1/F3支持、Z1/Z2未获支持；24拟合和哈希验收记录已合主线，完整结果在output/penmanshiel_hourly/p2_v35/RESULTS.md。
关键数字: Claude的P3为T22/T23/T24三折、每折24拟合共72；T22/T23折内R/F1/F3确认，T24及Z1/Z2描述，窗口门槛5000。
关键数字: 用户本条明确批准2022；2023/2024尚未获得本聊天明确批准。不得依据交接中转述扩大解封范围，也不擅自把三折冻结方案缩为单折。
需要 Claude 做的: 非阻断建议R1-S1–S3见审查文件：滚动折非统计独立、折内Holm的声明边界、验证指标和最终held-out测试评估锁的区分；不要求为建议额外来回一轮。
需要用户决定: 是否也批准现方案启用2023、2024，按三折72拟合一次冻结运行；若仅批准2022，则先将P3范围明确为T22。年份确认前不解封或启动。
