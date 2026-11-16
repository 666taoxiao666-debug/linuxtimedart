# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-16 第 88 次（X1 中期复核；新阶段 X2 只审方案：LightGBM 底座消融 + ANI 冻结验证） -->
轮次: R1/2 | 阶段: X2 | 分支: exp/X1 | 基线提交: 34936cf
请求: 审方案（X2 本轮不跑）；X1 剩下的部分照原授权继续执行
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md
允许改动: scripts/run_x1.py, configs/x1_protocol.json, configs/x2_protocol.json, tests/test_x1.py, output/x1/**, output/x2/**, output/thesis_materials/**, output/external_acquisition/gfs_g0/manifest_b1.json（本地未跟踪，不要提交、不要改）, output/g2/hot_dev/audit_metadata_r5.tar.gz（本地未跟踪，不要提交）
审查文件: docs/review/X2-R1-codex.md
本轮决议: 采纳 1（X1 R2-E1：按登记的 0.01 kW 阈值判等价，重跑 HoT 的 E1）。A1 负结果接受（N 不优于 A1），换底座有开发数据支持；X2 的方向说明见 DECISIONS

需要你做的:
1. **审 X2 方案**（STAGE_PLAN 全文）。重点审三处：
   - 拿 LUG 当"完整 A1"参照是否公平（同一切分、同一网格预算、同样的 seed）；
   - ANI 和 PCL 是不是只对已有输入做确定性变换，有没有泄漏，尤其是功率曲线只用拟合段、g_hist 从逐小时数组取；
   - 门槛和"接近通过时"的规则写得是否清楚。
   - 有确定缺陷就按唯一的最小修法写进审查文件，交我决议。X2 本轮不实施、不运行。
2. **X1 继续：**
   - 按 R2-E1 重跑 HoT 的 E1；
   - Cursor 交回 Chronos-2 的新回执后，跑 GPU 队列（A3、B）和 C1；
   - 执行规则不变：小设计自己定，记成 S 编号。
3. 不下载、不安装，不读 2026 年的数据。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-16：用户追加两个旧底座，只读审计及比较提案；X2全方案尚未复审 -->
轮次: R2/2 | 阶段: X1 | 分支: exp/X1 | 提交号: A2实现84f1b21，E1/E2实现c0e0800，组件结果5f64079，A1实现6e17435
审查结论: R1-B1已按决议实施；小设计S1–S8见docs/review/X1-R2-codex.md，不另等交审。
已直接修复: A2缺测持续法回退84f1b21；未改冻结模型、数据包和标签。
运行信息: A2、E2完成并退出；E1和E2曾分别在CPU-2/CPU-1并行，各限制2核2线程。
服务器根: /home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/X1/
A2 run: x1_84f1b21_r2/A2；E1/E2 run: x1_c0e0800_r3/E1和E2；JSON/日志已取回output/x1/对应目录。
验证: A2四个预测SHA及结果SHA、E2结果SHA服务器复核通过，本地A2/E2结果SHA一致。
关键数字: N较A2 MAE增益(kW)，K2022/P2021/HoT2024/HoT2025依次73.0382/73.0154/114.1882/98.0794；四集CI下界均>0。
关键数字: E2剔除比例依次3.30%/1.72%/6.32%/3.03%；N正常格子MAE147.1255/162.7196/194.2384/183.9005 kW，仅描述。
E2范围: output/x1/E2_saved/已补B、B0、N0、A1，七臂共用首轮正常mask；只读已有预测，不重推理，结果SHA本地一致。
E1停止: K/P完成；一次获准诊断发现HoT2024重合truth有1858格差异，最大0.000244140625kW；前20键/值见output/x1/E1/diagnosis.json。
E1诊断: 机组顺序一致，差异为float32量级；疑似CSV往返精度差，但尚无唯一且已验证修法。未改标签、未放宽门禁、未重跑HoT，需Claude复核诊断后决定。
A1: LightGBM4.5.0于本轮后半程可用；本地及服务器各8项测试通过，已正式启动PID1195938，CPU2核2线程。
A1 run: x1_6e17435_r4/A1；worktree /home/XT/nuist/pythoncode/TimeDARTFirst_X1_6e17435；启动回执output/x1/A1/launch_A1.json。
A1命令: scripts/run_x1.py --arm A1 --logs <数据根>/outputs/logs --output <run> --protocol-sha256 50da39efbd74700c4d816fb13dd1fa2be88ffcd11d78f2153f2ee2ecee101a48
A1状态: 四集COMPLETED，进程退出；48次固定拟合完成，四测试预测、3验证预测及12选中模型SHA服务器复核，JSON/日志取回output/x1/A1/。
关键数字: N较A1 MAE增益K/P/HoT24/HoT25=-1.2065/-7.9743/-6.3662/-6.3345kW；CI依次[-2.4516,0.2159]/[-10.8587,-5.6121]/[-8.2454,-4.6381]/[-7.9445,-4.8081]。
结论: N不优于A1，P及HoT两年LightGBM显著更好；有效负结果原样保留，不调参重跑。
GPU依赖: 旧Chronos回执INCOMPLETE/ConnectError；另有/home/XT/models/chronos2-29ec3766离线权重，但缺对应Cursor完整验收清单；A3/B未启动，C1等待A1/A3a。
补充范围: 用户要求加入两个旧工程；本轮只读审计，没有训练、安装或执行旧代码，未处理其他运行任务。
补充提案: output/thesis_materials/底座候选审计与比较提案_20261116.md；这是候选审计，不是X2整体R1通过结论。
候选: 纯TCN-GRU、加域对抗消融、CNN/BiLSTM/LightGBM融合；与N及完整输入LightGBM公平比较，具体预算待登记。
旧结果: B集成R²=.76416，LightGBM=.74634；90.16%是归一化RMSE派生指标，不是R²；另一结果表融合不优于CNN。
审计: B先全量标准化再切分，且预测入口可用同一评估真值拟合融合权重；不能原样运行。A旧入口也全量归一化，v2已仅训练段fit。
审计: A的域梯度实际系数−λ²；主打印R²只算第1步。旧成绩与当前1–4h结果不可直接比较，未确认历史高分对应的运行身份。
需要 Claude 做的: 审核补充提案P1–P3，登记独立底座筛选阶段、预算、融合时间切分及域损失公式；保留X2本轮只审不跑限制，不直接扩充冻结实验。
后续状态: HANDOFF88已批准HoT E1的0.01kW等价判据，本轮未执行；A3/B/C1仍待新回执。以上原X1数字为既有记录，不代表本轮重新查服务器。
需要用户决定: 无；下一步由Claude把候选比较纳入正式执行方案。本轮不提交其尚未审核的STAGE_PLAN/DECISIONS改动。
