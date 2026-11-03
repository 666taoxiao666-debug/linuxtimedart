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
<!-- 由 Codex 覆盖。最后更新：2026-11-04，三年授权齐全，P3实现待服务器验收 -->
轮次: R1/2 | 阶段: P3 | 分支: exp/P3 | 提交号: 协议14ec00a；本次为实现提交
审查结论: 方案/实现静态审查通过，评级B，0条未解决技术阻断；docs/review/P3-R1-codex.md；服务器测试/smoke尚未验收。
已直接修复: 无冻结代码改动；按Claude明确授权新增协议、准备包装器、runner及测试；S1–S3只落实执行边界，STAGE_PLAN仍由Claude维护。
运行信息: P2已合主线0e2e939并推送；本分支从该主线建立，main未动。
运行信息: 用户明确同意2022后又同意2023/2024；协议已先提交推送14ec00a，SHA256(LF)=a99b7ca91910b159088991bdbc680ba95ed240410e02c414f3e68ada22fe8926。
运行信息: 静态编译/diff检查通过；接下来在独立服务器worktree测试及--smoke，通过后才真实解封/准备/训练；当前未访问真实测试数值。
关键数字: T22/T23/T24三折，8臂×3seed×3=72拟合；确认门槛5000；T24及Z1/Z2只描述。
关键数字: 全72拟合冻结后独占创建最终测试锁；标记出现后不重启、不补跑；之前最多两次进程尝试，完整拟合按哈希复用。
需要 Claude 做的: 事后复核实现及S1–S3表述；不更改已冻结协议。结果完成后另行交接。
需要用户决定: 无；按用户最新偏好，正式运行启动后每30分钟检查，只通知完成、失败或需处理事项，不逐拟合汇报。
