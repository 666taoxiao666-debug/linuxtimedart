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
<!-- 由 Codex 覆盖。最后更新：2026-11-03，P2结果回传及哈希核验完成 -->
轮次: R2/2 | 阶段: P2 | 分支: exp/P2 | 提交号: 实现e3ae7a6；启动aa37091；提速规则e3554d5；结果见本次提交
审查结论: 通过，评级B，0条剩余阻断；docs/review/P2-R2-codex.md；结果技术核验完成，等待Claude复核科学结论。
已直接修复: 无新增代码改动；未改冻结模块或STAGE_PLAN；本轮不援引提速例外。
运行信息: 服务器10/10合成测试通过，smoke八臂各1seed/1epoch完成（11.029秒）；唯一真实实例24/24拟合完成，用时1385.350秒；原PID243138退出，无重跑。
运行信息: worktree=/home/XT/nuist/pythoncode/TimeDARTFirst_P2_e3ae7a6；run=/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Penmanshiel/p2_v35_e3ae7a6_r1。
运行信息: python -u scripts/train_penmanshiel_p2_v35.py --device cuda:0 --data-root /home/XT/nuist/pythoncode/TimeDARTFirst --out <上述run>；完整命令见process.json。
运行信息: 网络恢复后完成46源码、184输入（1,117,273,762字节）、120训练产物SHA核验，24份trained与freeze一致；25评估数组哈希留档，保存数组复算报告通过。
运行信息: stdout关闭后刷新清单；83份元数据及清单回传output/penmanshiel_hourly/p2_v35/并逐文件核对大小/SHA256，未回传权重和数组。
结果说明: 各臂3seed平均MAE/RMSE/R²及全部CI/p见output/penmanshiel_hourly/p2_v35/RESULTS.md和result.json；2021 ALL=38147窗口，h1–4合并，仅开发证据。
关键数字: R支持：P-last10→P-upstream MAE 195.5209→184.3714，RMSE 319.8619→299.8978，R² .754390→.784086；MAE增益CI [8.5347,13.9389]，Holm p=4.8457e-15。
关键数字: Z1未获支持：MAE增益3.0885，CI [-3.4426,10.1698]，Holm p=.365129；Z2未获支持：MAE增益-5.2222，CI [-9.9290,-.1436]，Holm p=.069357。
关键数字: F1支持：Scratch→FT MAE 314.4308→202.9806，RMSE 415.7316→313.9544；MAE增益CI [104.9873,118.2838]，Holm p=2.7009e-245；三个seed均选epoch0，不能称微调本身有效。
关键数字: F3支持：Scratch→FT MAE 225.7788→189.0949，RMSE 336.7399→302.7313；MAE增益CI [33.6116,39.9990]，Holm p=9.2960e-110；最佳epoch 7/9/11。
关键数字: RMSE增益CI依次R [15.1965,24.7787]、Z1 [24.6444,44.1513]、Z2 [-1.1158,13.6349]、F1 [96.1005,110.9388]、F3 [29.0518,40.0922]；统计比较用seed平方误差先平均再开方。
需要 Claude 做的: 复核P2实现、结果及F1 epoch0解释；F1/F3仅完整配方归因；R满足提请P3条件，请按新流程准备下一阶段方案，不将P2写成独立确认。
需要用户决定: 当前无；Claude复核后再请用户决定合回主线。主线仍882f30c，main未动，P3未启动。
