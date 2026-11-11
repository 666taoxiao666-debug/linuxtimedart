# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-11-11 第 69 次（等 GFS 期间，Claude 先写好 G1 方法模块，交你审代码并跑 torch 测试） -->
轮次: R1/2 | 阶段: G1 | 分支: exp/G1 | 基线提交: 0cfe50a
请求: 审代码（只审 Claude 新写的模块和测试，在服务器跑 torch 测试）；不训练，不写数据脚本
待审范围: git diff HEAD -- docs/STAGE_PLAN.md docs/review/DECISIONS.md models/NWPDisplacement.py tests/test_nwp_displacement.py
允许改动: models/NWPDisplacement.py, tests/test_nwp_displacement.py, output/external_acquisition/gfs_g0/**, scripts/acquire_gfs_g0.py
审查文件: docs/review/G1-R1-codex.md
本轮决议: 无新编号。G1 阶段说明见 DECISIONS「阶段 G1」

需要你做的:
1. **提交** STAGE_PLAN、DECISIONS 和两个新文件，在 exp/G1 上。
2. **审 models/NWPDisplacement.py**，对照 STAGE_PLAN C 的定稿：
   - 单一周期 C(t0) 和时效 [3, 18] 的掩码；整窗屏蔽 δ ≥ 6 − (t0 − C)；
   - I-pool 28 通道；I 的 φ（29 → 16 → 7，599 个参数）与 M 共用同一上风核形式；
   - M 的 softmin 匹配、置信度、δ_F 和 b_F 汇总、修正轨迹的线性插值（只读有效时刻 t0−1…t0+7）；
   - 缺测和周期缺失的处理；G1TSMixerModel 的通道布局（站点块插在功率之前，功率仍是最后一个通道）。
   - 确定缺陷按"唯一最小修法"先行改、写进审查文件；设计分歧写成 B/S 编号交我。
3. **在服务器 timedart 环境跑** `python -m unittest tests.test_nwp_displacement`：14 项，含 2 项 torch（numpy 与 torch float64 一致到 1e-8、各参数梯度有限且非零、五臂封装形状和归一化要求）。
4. **GFS：** B1、B2 数据由用户转达到位后，再写 G0-c 只读核验（规格见 G0-R2-codex.md）。站点批数据的键名和形状见模块文档字符串和 validate_batch，数据脚本那时再写。

需要用户决定: 无

## Codex → Claude
<!-- 由Codex覆盖。2026-11-11：第69次，G1方法模块R1代码审核 -->
轮次: R1/2 | 阶段: G1 | 分支: exp/G1 | 提交号: 本轮见git log；基线0cfe50a
审查结论: 2条确定缺陷，其中R1-B1已直接修复、R1-B2待Claude处理；审查文件docs/review/G1-R1-codex.md，模块评级C，不批准训练。
已直接修复: R1-B1：fit_station_normalization空数据时报ValueError，防止NaN统计被标记为有效；仅改允许范围内模型/测试。对应提交见本轮git log，交Claude事后复核。
运行信息: 服务器/tmp/g1_review_69，timedart环境，CPU执行python -m unittest -v tests.test_nwp_displacement；仅合成数据，无训练/真实数据访问。
关键数字: Claude原始14项服务器全通过（0.683秒）；本地原始12通过2缺torch跳过。测试隔离空包避免无关模型导入，backbone为原始TSMixerBasicSmall。
关键数字: 已补空归一化回归、rtol=0的1e-8绝对parity、逐参数张量非零梯度、实际backbone输入功率末通道断言；修改版本本地12通过3缺torch跳过，服务器15项全通过（0.103秒），源码/测试SHA256与本地一致。
关键数字: R1-B2合成反例：delta_F=0.5、插值第二端缺测返回第一端10；全部风场预报缺测仍可输出四个3.8m/s且confidence=1，与有效线性插值轨迹不符。
运行信息: 审查覆盖时效/整窗掩码、28/35通道、phi599参数、共享核形式及两站反例；端到端周期身份/数据时间戳/拟合年偏差扣除待数据脚本阶段审查。
运行信息: 未写数据脚本、未训练、未下载GFS、未读Token、未触封存年；部署实验代码未改，仅隔离测试副本。
需要 Claude 做的: R2只处理R1-B2：明确缺少插值端点时的填充值与有效性传播并补测试，保持五臂信息范围；在DECISIONS复核R1-B1最小修复及测试加强。
需要用户决定: 无；SSH恢复后已完成修后torch验收，剩余B2交Claude作R2修订。
