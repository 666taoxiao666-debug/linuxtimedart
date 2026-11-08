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
<!-- 由Codex覆盖。2026-11-08 第47次收尾：LOCAL_VERIFIED -->
轮次: R2/2 | 阶段: L0-B | 分支: exp/L0-B | 提交号: 本轮见git log
审查结论: 本地验收通过；第47次完成，见docs/review/L0-R2-codex.md及L0B_RESULT_SUMMARY.md。
已直接修复: 无；只取回既有元数据、验收及交接，未改运行代码。
运行信息: 仅传inventory、qualified、run.log，内存压缩流27802字节；三个副本从r1复用；6文件SHA256均等于远端登记，副本字节相同。
运行信息: PID508698原实例已完成；原命令/cwd/退出码不可恢复的出处限制保留，按Claude第47次接受；无重跑、无远端写入。
运行信息: 本地目录output/hill_of_towie_e1/l0b_r2；status=LOCAL_VERIFIED；源码SHA256与启动回执和inventory一致。
关键数字: 重算62站逐年资格/气压标记及门槛一致；12合格、11有气压、象限[1,2,3]，passed=true，unresolved为空。
关键数字: 冻结ID=44,48,67,105,113,132,137,150,161,177,212,235。
关键数字: qualified_stations SHA256=5909e1436559dd4334649c41c66881853ace75bc68a80f4fc9bfb30fcc60a979。
关键数字: 137合格/有气压；2016–2023风速%=99.9772,99.9543,100,99.9886,100,99.9543,99.7260,99.7945；气压逐年值见报告/JSON。
关键数字: 仅137的2016年2个冲突小时，其余站年全0；未超1%。512 MIDAS文件及风机字节核验记录保留。
运行信息: inventory只存解析数414，没有逐文件parsed列表；414非封存清单均为2016–2023，哈希一致源码强制年份/白名单，表头也仅这些年；此证据边界如实记录。
需要 Claude 做的: 复核本地验收、站137和冻结站点集合，以及parsed路径列表缺失的审计限制；决定后续方案，未经用户批准不合主线。
需要用户决定: 无；等待Claude复核。未训练/评估/读取封存观测，main未动。
