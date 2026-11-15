# STAGE_PLAN — 只写下一个阶段

作者：Claude（Codex 只提意见，不改本文件）。上限 150 行：要加新内容时，先删旧内容。
内容只从现有代码和结果中提取；拿不准的地方标「待确认」；事先设定的阈值标「事先设定」。
存档：G2b 全文（诊断、冻结、3a 演练、3b 准备、原 3c 判定标准）见 `git show 317e344:docs/STAGE_PLAN.md`；G2a 见 `git show 5b02f93:docs/STAGE_PLAN.md`；复核见 DECISIONS「阶段 G2b」「阶段 G2c」。

## 阶段名

G2c（2026-11-15）：Hill of Towie（HoT）封存年 2024、2025 的一次性评估，即 G2b 的 3c。
- G2b 的 3a 演练和 3b 准备已复核通过（317e344）。
- 这是 HoT 封存年唯一一次推理。之后不再有独立测试年；用户已同意，后续大模型实验一律写成开发阶段的探索性结果。

## 目标

1. 用冻结的 14 个模型和复核过的两个数据包，对 2024、2025 做唯一一次推理和统计。
2. 结果取回后由我独立复算，再写结论。

## 假设与判定标准

**输入固定（事先设定，任一不符就停，不写评估标记）**
- 冻结：SHA 981f3ae27e6792f5da71c177af1c24a0e91c4dfcd880322773fb4f9fe34607f6；协议 SHA da809d691e93b7bb284fc7ce282f1c16a59ecd0fe5342f68ceab0e8530966a53。
- 数据包：
  - 2024：a4dfc88037a7b14fecccf6fbcd56e1ed65b9a3d9f996d43d0446e7a65c942420，59587 窗；
  - 2025：07f34631dece57d7fafb52cfe647c58e3745fddd58e2f0909dc8018462952c02，56857 窗。
- prepared_complete.json 的状态必须是 PREPARED_ONLY，入口 SHA 为 cbdc1b98…（换行规范化后），准备目录里没有 failed.json。

**c0 冒烟（只用开发数据，必须先通过）**
- 把冻结目录复制到一个临时目录，用 3a 的 2023 演练数据包（30056 窗）同时充当 "2024" 和 "2025" 两个键，用下面的 evaluate 模式完整跑一遍。
- 通过 = results.json 和 completed.json 都生成，14 × 2 份预测都是有限值，没有 failed.json。
- 冒烟只检验代码能否跑通，结果不报告、不解读。不通过就停下交回，不碰真正的冻结目录。

**c1 一次性评估**
- 顺序：
  1. 核对上面所有哈希；
  2. 确认输出目录是新的；
  3. 在真正的冻结目录里独占写入 test_evaluation_started.json；
  4. 加载两个数据包，并再核一次 SHA；
  5. 调用冻结的 evaluate_sealed（CPU，2 线程，和 3a 相同）；
  6. 写 results.json 和 completed.json。
- 失败就写 failed.json，停下交回，不重试、不补跑。预测是逐个存盘的；如果失败发生在统计阶段，由我决议后从存盘预测重算统计，不重新推理。
- **主检验族（与 G2b 3c 相同，不改）：**
  - Q1h：N 前 3 个 seed 对 B（逐窗损失先在 seed 上平均）；
  - Qe：N 的 5 个 seed 集成（预测平均）对 N 单模型（5 个 seed 的损失平均）。
  - 每个问题取 p_Q = max(p_2024, p_2025)，再对 {Q1h, Qe} 做 Holm（α = 0.05）。
  - "通过" = Holm 通过，且两年增益都 > 0、CI 下界都 > 0。
  - 统计：paired_loss_statistics，按起报时刻成簇，块长 8，5000 次，seed 2024，NW 滞后 7；块长 56 作敏感性。
- **次要（事先登记，不进 Holm）：** N 对 N0 两年的增益和 CI；Kelmarsh、Penmanshiel、HoT 两年的逆方差合并估计，只描述（HoT 两年共用模型，不声称独立）。
- **只描述：** B 对 B0、N0 对 B0、分时距 MAE、爬坡、准确率（1 − nRMSE，额定 2300 kW）、R²、各臂平均偏差。
- 结论写法：
  - Q1h 通过 = "数值预报增益在第三个风场独立确认"；不通过 = "未在第三个风场复现"。
  - Qe 同理。
  - 不调参，不重跑，不按结果另选子集。

**取回（c1 完成后）**
- 提交：results.json、completed.json、日志、两个标记文件、预测 SHA 清单。
- 另外每年导出一个小数组包，不提交，拷到本地 output/g2b/sealed_eval/arrays/（加 .gitignore）：truth、issue、turbines、times、anchor，以及 14 组预测。供我独立复算 MAE、增益和统计量。

## 实验配置表

| 步 | 内容 | 规模 | 之后 |
|---|---|---|---|
| c0 | 冒烟（冻结目录副本 + 2023 演练包） | CPU 约 10–30 分钟（待确认） | 通过 → c1；不通过 → 停 |
| c1 | 一次性推理 + 统计 | CPU 约 20–60 分钟（待确认） | 取回后交回 |

## 改动文件清单（= HANDOFF「允许改动」）

- `scripts/evaluate_g2b_sealed.py`：只追加 evaluate 模式（参数 --prepared、--freeze、--output、--smoke）和配套函数。已有函数一行不改，c1 不调用 rehearsal_gate（入口 SHA 会变，改用固定的数据包 SHA 核对）。
- `tests/test_g2b.py`：追加三项测试——数据包哈希不符时拒绝、且不写标记；标记已存在时拒绝；输出目录已存在时拒绝。
- `output/g2b/sealed_eval/**`、`output/g2b/smoke/**`。

## 本阶段不做什么

- 不重训、不调参，不改协议、数据包或任何判定标准。
- 不读 2026 年的数据；不做事后分组挑选。事先登记的结果全部报告，不管好坏。
- 不加大模型；不合主线（结果复核通过、用户同意后再合）。

## 风险（≤5 条）

1. c1 中途失败：标记已经写了，不能重跑推理；统计阶段的失败可以从存盘预测重算。
2. 测试年全是改造后数据，拟合段只有 8.5% 改造后样本，各臂绝对误差可能偏高。相对比较不受影响，用平均偏差描述。
3. 主检验不通过：如实报告；论文写"在两个风场确认，第三个风场未复现"，并给出可能原因（改造、地形），不追加实验去挽救。
4. 2025 年窗口比 2024 年少 4.6%（56857 对 59587）。这只影响样本量，各臂用的是同一批窗口。

## 存档引用

- 冻结包：output/g2b/freeze/；演练：output/g2b/rehearsal/server/；准备：output/g2b/sealed_prep/server/。
- 新入口：scripts/evaluate_g2b_sealed.py（bc7730e）；冻结入口：scripts/train_g2b.py（evaluate_sealed、joint_test_decisions、pooled_descriptive）。
- 对照：output/g2a/results81/（Kelmarsh、Penmanshiel 的 N 对 N0）；output/g1/results_78/。
