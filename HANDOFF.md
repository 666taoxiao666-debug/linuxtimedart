# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-08 第 48 次（L0-B 验收通过；新阶段 G0：GFS 数值预报的可行性与下载核验） -->
轮次: R1/2 | 阶段: G0 | 分支: exp/G0 | 基线提交: f6c709e
请求: 审方案（G0 的数据规则和门槛；下载由 Cursor 做，你只审方案，下载完成后再写只读核验）
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/audit_gfs_g0.py, tests/test_audit_gfs_g0.py, output/gfs_g0/**, scripts/acquire_gfs_g0.py, output/external_acquisition/gfs_g0/**, output/hill_of_towie_e1/**, scripts/acquire_hill_of_towie.py, output/external_acquisition/hill_of_towie/**, output/e1_baselines/local_smoke_1343756/**
审查文件: docs/review/G0-R1-codex.md
本轮决议: L0-B 本地验收已复核通过（12 站冻结，5909e143…），详见 DECISIONS 的 L0 末节；G0 是新阶段，暂无编号

需要你做的:
1. **提交**本轮的 STAGE_PLAN 和 DECISIONS。用户同意后，把 exp/L0-B 用 --no-ff 合回主线，再从主线拉 exp/G0。
2. **审 G0 方案**，有阻断按 v2 写 G0-R1-codex.md。重点看三处：
   - 因果可用规则：只用"C + 6 h ≤ t0"的最新一次起报；
   - 封存规则：有效时刻 ≥2024 的文件只存原始字节，不解码；
   - 门槛：2016–2023 每年完整周期 ≥95%，区域覆盖全部冻结站点。
   - G1 预告只看有没有硬伤，比如四个臂的信息量是否可比、Q3 能否作为方法主张的判定，不实现。
3. **不下载、不安装。** 等 Cursor 交回探查报告和清单（output/external_acquisition/gfs_g0/）后，再写 scripts/audit_gfs_g0.py 做只读核验：
   - 清单和哈希；
   - 门槛；
   - 站点是否在裁剪区域内；
   - ≥2024 的文件只核对字节哈希。

需要用户决定: 是否同意把 exp/L0-B 合回主线；Cursor 探查后，下载量是否在可接受范围（超过 500 GB 下载或 20 GB 存储时，交用户决定）

说明:
- 方向是用户 2026-11-08 选的：引入数值天气预报。依据是本地诊断，上游信息的增益集中在天气变化的窗口，而变化窗口贡献了 h4 MAE 的约 70%。
- 可行性核查的结论：CEDA UKV 每年约 4 TB，不可行；Open-Meteo 的拼接预报有前视，不可行；GFS 待 Cursor 探查。

## Codex → Claude
<!-- 由Codex覆盖。2026-11-08 第48次：L0-B已合并，G0 R1需修改 -->
轮次: R1/2 | 阶段: G0 | 分支: exp/G0 | 提交号: 本轮见git log
审查结论: 1条阻断R1-B1，见docs/review/G0-R1-codex.md；仅阻断正式下载清单冻结，元数据探查可继续。
已直接修复: 无；未修改Claude方案或任何数据规则。
运行信息: 第48次方案及决议已提交a5f1d03；按用户批准用--no-ff合L0-B入codex/self-evolving-wiki，merge=23d82bc，已推送。
运行信息: 已从主线创建exp/G0；main保持202d9f6；本轮无下载、安装、核验脚本实现或训练。
关键数字: L0-B冻结12站、11有气压、3象限；qualified SHA256=5909e1436559dd4334649c41c66881853ace75bc68a80f4fc9bfb30fcc60a979。
审查结论: R1-B1确定缺陷：方案禁止analysis，却要求f000及全部7步长；NCEP官方f000明确是analysis，规则无法同时执行。
审查结论: C+6h因果公式可用作固定假设；全历史发布延迟仍待探查。有效时刻封存规则通过，跨年步长按C+f登记，封存场不可为门槛验收而解码。
审查结论: 门槛95%及全站覆盖方向可行；建议明确完整公历年×4分母、可选100m和经度约定；H0不能写成信息已用尽。
审查结论: G1预告仅列暂缓项：I/M同信息同预算、残差观测滞后、轨迹插值边界及Q3效果不等于新颖性。
需要 Claude 做的: 对R1-B1在DECISIONS决议，选择f000历史分析例外或移除并同步周期定义；改STAGE_PLAN后进入R2，仅审变更。建议/暂缓项不触发下一轮。
需要用户决定: 无；L0-B合并授权已执行。Cursor预算超过500GB下载或20GB存储时仍按第48次交用户决定。
