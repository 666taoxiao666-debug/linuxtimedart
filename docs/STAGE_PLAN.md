# STAGE_PLAN — 只写下一个阶段

作者：Claude（Codex 只提意见，不改本文件）。上限 150 行：要加新内容时，先删旧内容。
内容只从现有代码和结果中提取；拿不准的地方标「待确认」；事先设定的阈值标「事先设定」。
存档：G2b 第 1、2 步原文见 `git show 428316d:docs/STAGE_PLAN.md`；G2a 方案见 `git show 5b02f93:docs/STAGE_PLAN.md`；G1c 协议见 `git show 2dcfc1d:docs/STAGE_PLAN.md`；复核见 DECISIONS「阶段 G2b」。

## 阶段名

G2b（2026-11-14 起）：Hill of Towie（HoT）独立确认。第 1、2 步已完成并核对通过（428316d）；本轮做第 3 步，分 3a、3b、3c，3c 前停下交回。
- 用户 2026-11-14 裁决：先完成 HoT 确认，再试大模型；确认用 2024、2025 两年。

## 目标

1. 在一个从没看过的风场上，检验两条已在开发数据上成立的结论：数值预报增益（N 对 B）和种子集成增益。
2. 保证封存年只评一次，而且评估时的数据准备与训练时完全相同。

## 假设与判定标准

**已完成（只作记录，不再改）**
- 第 1 步：18/21 台 S2 ≤ −5%，3 台 |S2| < 5%，按规则选情形 A。拟合段 2016-01-01 至 2023-06-30，共 420475 窗，改造后占 8.53%；验证段 2023-07-01 至 2023-12-31，共 30056 窗，改造后占 100%。所有臂都保留改造通道。
- 第 2 步：14 次拟合，冻结 SHA 981f3ae2…，协议 SHA da809d69…。验证 MAE 的种子均值（kW）：B0 221.92，B 202.78，N0 184.94，N 182.29（只描述，不是检验）。

**第 3 步总则**
- 冻结包不动：scripts/train_g2b.py 以及 selection_freeze.json 里列出的源码都不改（一改 frozen_gate 就会拒绝）。
- 修复和新增功能写进新脚本 `scripts/evaluate_g2b_sealed.py`，原因只有一个：train_g2b.py 的哈希已锁定。
  - 新脚本从 train_g2b 原样导入这些函数：frozen_gate、common_packs、frozen_hourly、decode_test_points、arm_pack、make_model、evaluate_sealed、joint_test_decisions、pooled_descriptive。
  - 只重写两处：sealed_midas（加冲突规则，见 G2b-C1）和 prepare_sealed（改顺序、拆成 3b 和 3c，见 G2b-C2、C3）。
- 新脚本只有一个准备函数，参数是年份、小时轴、切分、截止时间；3a 和 3b 调用的是同一个函数。

**3a 演练（只用开发年 2023，必须先通过；无需封存授权）**
- 用这个准备函数从原始文件重建验证段，参数如下：
  - 年份 (2023,)，小时轴 2023-01-01 01:00 至 2024-01-01（左闭右开），切分 2023-07-01 至 2024-01-01，截止 2024-01-01；
  - GFS 用 2023 开发 nc 组成的清单，经 decode_test_points 解码；MIDAS 读 2023 开发目录；SCADA 按开发年规则读 2023。
- 用 14 个冻结模型推理，只在这个开发验证段上算 MAE。
- 通过标准（事先设定）：
  - 窗口数 = 30056；
  - 14 个模型的 MAE 与各自 trained.json 的 validation_mae 相差都 ≤ 0.01 kW。
  - 任一条不符就停下交回，不进 3b。
- 理由：封存年只能评一次。演练在开发数据上证明封存准备路径与训练时的准备路径一致，GFS 插值、上游特征、归一化、窗口步长、改造通道一并验证。

**3b 封存输入准备（本轮授权；只准备，不推理，不算任何误差）**
- 顺序：
  1. 先确认输出目录是新的；
  2. 写尝试标记 test_preparation_started.json；
  3. 依次处理 GFS、MIDAS，最后读 SCADA。
- GFS：按下面的"已批准 GFS 输入清单"生成清单。任一年完整率 < 95% 就在读 SCADA 前停下。
- MIDAS：读 2024、2025 封存文件，身份按下载清单核对（大小、MD5、SHA）。冲突规则与开发年相同：调用 audit_hill_of_towie_e1.conflict_missing，冲突的站点时次三个字段置缺测；单站单年冲突 > 1% 就停。
- SCADA：读 2023–2025 三年，用开发年同一套 read_month、边界合并和施工排除规则；2026 年的文件不打开。
- 冻结的拟合段统计（填充值、均值和尺度、GFS 均值和尺度）只应用，不重估。
- 产出：
  - 2024、2025 两个数据包，各附 SHA；
  - 只出计数：窗口数、改造占比、GFS 完整周期数、MIDAS 冲突数、SCADA 屏蔽小时数、上游缺测行占比。
  - 不加载模型；不算功率的均值、误差或任何标签统计。
- 写完停下交回。如果是格式类失败（像 r4、r5 那样），我复核修法后可以重跑 3b，最多 2 次。此时还没有任何模型见过封存年。

**已批准 GFS 输入清单（3b 生成，其 SHA 作为 --gfs-sha256）**
- 周期范围：
  - 2023 年 C ≤ 2023-12-31 00Z：用现有开发 nc，SHA 取 B1 修正审计；
  - 边界周期 2023-12-31 06Z、12Z、18Z：开发 nc 只含 valid < 2024 的时效。把它和封存 GRIB 用同一函数裁剪出的其余时效沿 step 合并成完整文件；合并后若通不过 decode_test_points 的时效和有效时间核对，就记为不完整；
  - 2024-01-01 00Z 至 2025-12-31 00Z（C + 18 h < 2026-01-01）：从 sealed_raw 转换。sealed_raw/2026 不打开。
- 转换方法：
  - 每条消息按 manifest_b1.json 核对字节数和 SHA；
  - 按开发时的顺序拼接（时效顺序），用开发时同一个裁剪函数（acquire_gfs_g0.py 的 crop_grib_to_netcdf），记录该脚本的 SHA；
  - 输出到 datasets/external/gfs_g0/test/<年>/gfs_<周期>_cropped.nc。
- 结构一致性：经纬网格和变量名必须与 2023 开发 nc 完全相同，否则停。
- 完整周期的判据：6 个时效齐全、valid_time = C + 时效、13 个点的 u10、v10 都是有限值。不完整的周期 cycle_ok = 0，各臂相同，不回退到上一周期。
- 门槛（事先设定，同开发年 passes_95pct）：2024、2025 每年完整周期都 ≥ 95%。
- 不安装软件包，用服务器上现有的 GFS 解码环境；缺依赖就停下，交给 Cursor 安装。

**3c 一次性评估（我复核 3b 和新脚本后另行放行，只做一次）**
- 先写独占标记 test_evaluation_started.json，再核对 3b 数据包的 SHA，然后调用冻结的 evaluate_sealed 做推理和统计，完成后交回。
- **主检验族（事先设定，不变）：**
  - Q1h：N 前 3 个 seed 对 B（逐窗损失先在 seed 上平均）；
  - Qe：N 的 5 个 seed 集成（预测平均）对 N 单模型（5 个 seed 的损失平均）。
  - 每个问题取 p_Q = max(p_2024, p_2025)，再对 {Q1h, Qe} 做 Holm（α = 0.05）。
  - "通过" = Holm 通过，且两年增益都 > 0、CI 下界都 > 0。
  - 统计同 G1：块长 8、5000 次、seed 2024、NW 滞后 7；块长 56 作敏感性。
- **次要（事先登记，不进 Holm）：** N 对 N0 两年的增益和 CI；三场逆方差合并估计，只描述。
- **只描述：** B 对 B0、N0 对 B0、分时距、爬坡、准确率（1 − nRMSE，额定 2300 kW）、各臂平均偏差。
- 结论写法：Q1h 通过 = "数值预报增益在第三个风场独立确认"；不通过就如实写"未在第三个风场复现"，不调参，不重跑。

## 实验配置表

| 步 | 内容 | 规模 | 之后 |
|---|---|---|---|
| 3a | 2023 演练（开发年） | CPU 约 1 h + 推理几分钟 | 通过 → 3b；不通过 → 停 |
| 3b | 封存 GFS 转换、MIDAS、SCADA 准备 | CPU 约 2–4 h（待确认） | 停下交回 |
| 3c | 推理 + 统计（一次） | GPU 几分钟 | 交回 |

## 改动文件清单（= HANDOFF「允许改动」）

- Codex：
  - `scripts/evaluate_g2b_sealed.py`（新文件，原因见总则）；
  - `tests/test_g2b.py`（追加测试：冲突置缺测、边界周期合并、输出目录检查在标记之前、SCADA 在 GFS 和 MIDAS 之后读）；
  - `output/g2b/rehearsal/**`、`output/g2b/sealed_prep/**`。
- 服务器上的 datasets/external/gfs_g0/test/** 不进 git。

## 本阶段不做什么

- 不改冻结包、协议、配方、臂、seed，也不重训。
- 3c 放行前不加载任何模型去碰 2024、2025 的数据包；3b 不算任何误差或标签统计。
- 不读 2026 年的 SCADA、MIDAS，不解码 2026 年的 GFS。
- 不加大模型，不加新方法臂。

## 风险（≤5 条）

1. 3b 遇到新年份的格式问题（开发年出过 4 次）：在推理之前发现，复核后重跑，不影响一次性评估。
2. 2024、2025 的 GFS 完整率低于 95%：在读 SCADA 前停下，由我和用户决定。
3. 测试年全是改造后数据，而拟合段只有 8.5% 改造后样本，各臂绝对误差可能偏高。比较的是同条件下的相对差，用各臂平均偏差描述。
4. 3a 不通过：说明准备路径有差异，先查清楚再进 3b，不带病评估。

## 存档引用

- 冻结包：output/g2b/freeze/（SUMMARY.md、verification.json、formal/selection_freeze.json）。
- 改造诊断：output/g2b/retrofit_diagnostic/（scores.json、status_summary.json）。
- HoT 开发年准备：output/g2/hot_dev/results82/；2×2 结果：output/g2a/results81/；G1：output/g1/results_78/。
- GFS：output/gfs_g0/b1/（acceptance.json、audit_b1_corrected.json）；裁剪函数在原文件夹 scripts/acquire_gfs_g0.py（未入库）。
