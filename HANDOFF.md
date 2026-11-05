# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-05 第 29 次（N0 结题 + N1-PH 首轮） -->
轮次: R1/2 | 阶段: N1-PH | 分支: exp/N1-PH | 基线提交: 9f125d0
请求: 审方案（N1-PH STAGE_PLAN）+ 没有阻断就实现并运行
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: configs/n1ph_protocol.json, scripts/train_n1ph.py, tests/test_train_n1ph.py, output/n1ph/**
审查文件: docs/review/N1-PH-R1-codex.md
本轮决议: N0 一节没有新编号；N0 结果我已用 result.json 的 56 对原始记录独立复核，与你一致，判为有效负结果

需要你做的:
1. **N0 结题**：先在 exp/N0 上提交 DECISIONS.md 和 HANDOFF.md。用户同意的话，用 `--no-ff` 把 exp/N0 合回主线并推送，exp/N0 保留。STAGE_PLAN 先不提交，带到 exp/N1-PH。
2. **N1-PH**：从主线拉 `exp/N1-PH`，审 STAGE_PLAN。重点审三处：PH 通道有没有时间泄漏；零样本时是否只用 Kelmarsh 和 N0 的统计量；复用 P2、P3 数组时窗口是否一致。没有阻断，就按"执行顺序"实现并跑 1 个实例（≤1.5 h）：先冻结协议，再核对哈希，再完成 36 次拟合，最后只评估一次。确定缺陷可以按提速规则处理。
3. **交回**：「关键数字」写：
   - 主折上 ZP、ZQ、F1P、F3P 的判定（含效应、CI、Holm p）；
   - IP 的非劣结果；
   - 继续、部分、停止三者中的哪一个；
   - 描述折 T22、T23 的同方向数字；
   - 各臂 2021 年的 MAE/RMSE。

需要用户决定: 合并 N0（用户转话时一并说明）

说明:
- PH 的迁移优势来自 N0 的事后观察，所以 N1-PH 作为新假设在风场上重新登记。创新预期约 5–5.5，论文走 CCF-B 的应用赛道。
- 这是 N1-PH 的第 1 轮。方案如果有阻断，而且修法不唯一，就停在冻结协议之前交回。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-11-05，N0唯一正式实例完成 -->
轮次: R2/2 | 阶段: N0 | 分支: exp/N0 | 提交号: 运行0fba03f；结果825bb60（已推送）
审查结论: 通过，0条阻断；docs/review/N0-R2-codex.md；R1-B1/S1/S2均已处理。
已直接修复: 获批范围内新建N0脚本及测试；常量/全缺测列标准差置1，防止无意义放大；未改旧源码或判定规则。
运行信息: 服务器timedart环境11项测试通过（含合成端到端）；本地语法检查通过，本地无torch，不安装或更改环境。
运行信息: scripts/pilot_station_operator_n0.py --run；run=outputs/logs/StationOperator/n0_0fba03f_r1；345.668秒，5次拟合完成；未重启补跑。
运行信息: 核对282输入、309二进制产物哈希及304本站预测数组；迁移按56对保存记录复核算术；源码未变，元数据回传哈希差异0。
关键数字: 本站O对H改善中位数−0.044020 m/s；正值0/8。
关键数字: 迁移D中位数−0.008796，正值3/8；G_tr(O/H)中位数+0.041309/+0.040839 m/s。
关键数字: PH相对O本站MAE差−0.040598；迁移G_tr(O)−G_tr(PH)=−0.034679 m/s，均偏向PH。
关键数字: 继续=false，停止=true，选简=false；Penmanshiel仅描述：−0.028868 m/s，正值0/11。
关键数字: 规模7/14/28站本站改善−0.062941/−0.056165/−0.044020，均未通过门槛。
需要 Claude 做的: 复核output/station_operator_n0/RESULT_SUMMARY.md及实现，记录有效负结果；按原规则停止当前路线，N1未运行。
需要用户决定: Claude复核后再决定合回主线；此前GitHub连接/认证故障已恢复，代码及结果已推送exp/N0。
状态: 无新增N0定时任务，运行已结束；不读风场标签或2022+站点文件，不下载、不改main；结果留exp/N0。
