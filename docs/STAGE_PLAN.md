# STAGE_PLAN — 只写下一个阶段

作者：Claude（Codex 只提意见，不改本文件）。上限 150 行：要加新内容时，先删旧内容。
内容只从现有代码和结果中提取；拿不准的地方标「待确认」；事先设定的阈值标「事先设定」。
E1 的方案原文见 `git show ae3b8f2:docs/STAGE_PLAN.md`；结果和复核见 DECISIONS.md 的 E1 一节。

## 阶段名

L0：判断"大模型设计可学习上游编码器"值不值得做，同时补齐底座。三条线：
- H0 线：上游信息余量探针（不训练深度模型，不用大模型）。它决定后续 L1（大模型结构搜索）开不开。
- A2 线：iTransformer、TimeXer 的归一化补查（E1 复核发现的问题）。
- B 线：Hill of Towie 清点，从 E1 原样移入，等 Cursor 下载完成后再做。

背景（用户 2026-11-06 提议）：把 11 个手工上游特征改成可学习的参数化结构，结构由大模型提出。
我的判断（理由见 DECISIONS.md 的 L0 一节）：现在直接做偏牵强。可学习编码器（K1b 的 A、B2）、站网算子（N0）、站网预测器（N1-PH）都没赢过手工特征；新增气温、露点、云量的筛查 S 也没信号。
所以先用最便宜的办法测：站点数据里还有没有手工特征没用到的信息。有，才做 L1；没有，就不加。

**结果（2026-11-07）**：H0 = STOP（Kelmarsh 平均 H_MAE 0.31%），L1 不开；A2 完成，4 个组合都选 use_norm=false。复核见 DECISIONS.md 的 L0 一节。现在只剩 B 线，下一个方法阶段等用户和导师决定。

## 目标

1. **H0**：在两个风场上估计"站点原始观测相对 11 个手工特征的增量信息"上限。
   - 结论只有三种：GO（开 L1）、STOP（不做大模型编码器）、边缘（交用户）。
2. **A2**：确认 E1 的强基线没有被归一化方式拖累。
   - 现象：Kelmarsh 2024 上，iTransformer、TimeXer 的 h1 MAE 是 131.6，与小时持续法 131.9 持平，比 10 分钟持续法的 126.0 还差，TSMixer-last10 是 117.7。
   - 它们的上游增益只有约 4 kW，TSMixer 是 15–18 kW。
   - 怀疑与 TSLib 默认的 use_norm=true 有关：逐窗口标准化后，网络内部看不到功率和外生通道的绝对水平（输出会按目标窗口均值和尺度还原，外生水平不还原）。TSMixer 只用全局标准化。以 A2 实测为准（R1-S2）。
3. **B**：同 E1 的 B 线，只读清点和测试年登记。

## 假设与判定标准

**H0（事先设定；只用已打开过的年份，结果只算开发证据）**
- **基线预测**：每个风场固定一个冻结的 TSMixer-upstream（TSMixer_upstream_all 配方），3 个 seed 的预测取平均，只推理、不训练。
  - Kelmarsh 用 v31 的模型（拟合 2016–2021，验证 2022）。探针年是 2023（已有测试数组）和 2024（用冻结 checkpoint 和它自己的 scaler 补推理）。
  - Penmanshiel 用 P3 T22 折的 P-upstream（拟合 2016-08–2020，验证 2021）。探针年是 2022（已有）和 2023（补推理）。
  - checkpoint 缺失就停下报告，不换模型凑数。
- **目标**：残差 r = y − ŷ（kW），ALL 窗口，4 个 h 叠在一起。
- **特征集**：全部因果，站点观测只用 ≤ 起报时刻 − 1 h 的，与手工特征相同。
  - a（对照）：ŷ、h、风机编号、上一小时均值功率、最后 10 分钟功率、时刻和年内日的 sin/cos、11 个手工特征（起报时刻和 3 小时前两个时刻）、上游缺测标志。
  - b1（原始站点库）：a，加上每个冻结站点的
    - t0 = 起报 − 1 h 时的风速、风向 sin/cos、海平面气压；
    - t0 − 1、−2、−3、−5 h 的风速和气压；
    - 相对来流的几何量：顺风距离 d∥、横风距离 d⊥、d∥ / max(本地风速, 2)。来流方向的取法同手工特征。
    - 缺测保留 NaN。
  - b2（按来流分箱）：a，加上 3 个距离环（20–50 / 50–100 / 100–150 km）× 4 个来流扇区（上风 ±45°、左、右、下风）的 {平均风速、风速 3 h 变化、气压 3 h 变化、有效站数}，共 48 维。
- **学习器**：sklearn HistGradientBoostingRegressor，参数固定不调：
  - learning_rate 0.05，max_iter 300，max_leaf_nodes 31，min_samples_leaf 200，l2_regularization 1.0，early_stopping=False，random_state 2024。
  - MAE 用 absolute_error、RMSE 用 squared_error，分别拟合。
  - 不新装包；sklearn 不可用就停下报告。
- **交叉拟合**：按年份，在风场内两个方向各做一次：用 A 年拟合、在 B 年打分，再反过来。
- **余量定义**：H = [err(a) − min(err(b1), err(b2))] / err(基线) × 100%，按方向分别算 MAE 和 RMSE。
  - b1、b2 取较好者，故意偏乐观：这是一个"否决"测试，乐观估计下仍没有余量，才能放心停止。
- **判定（事先设定）**：
  - **GO**：两个风场都满足：两方向平均的 H_MAE ≥ 1.5% 且 H_RMSE ≥ 1.0%，并且 4 个方向的 H_MAE 都 > 0。之后才写 L1 正式方案。
  - **STOP**：任一风场两方向平均的 H_MAE < 0.5%。不做大模型编码器。论文措辞限定为"该固定探针未发现足够余量"，不说成信息论上限（R1-S1）。
  - **边缘**：其余情况，交用户决定。我的默认建议是不开 L1。
  - 阈值理由：已确认的上游增益是 7–10%。上限不到 1.5% 的编码器改进，撑不起方法论文；不到 0.5% 就无事可做。
- **只描述**：a 相对基线的改进（说明 TSMixer 自身还有多少系统误差）、分 h 的余量、上游缺测子集、b1 和 b2 的差别。

**A2（只描述，规则同 E1 A 线）**
- 模型：iTransformer、TimeXer × {last10, upstream}，新配置 use_norm=false，其余固定参数和训练配方与 E1 完全相同。
  - R1-B1：移植的 iTransformer 无条件归一化，所以 false 由 A2 脚本里的子类实现（只去掉归一化和反归一化，参数不变）；TimeXer 本身遵守开关。
- 选参：
  - 在 Kelmarsh 2023 验证集上用 search seed 2024 新做 8 次搜索拟合（4 个模型 × 通道组合 × d_model {64, 128}）。
  - 每个模型 × 通道组合，在 4 个候选里按验证 MAE 选：{use_norm true/false} × {64, 128}，其中 use_norm=true 的两次复用 E1 冻结的搜索记录。
  - 平局时选 use_norm=true，再选较小的 d_model。
- 正式拟合：只有选中 use_norm=false 的组合，才在三个设定 × 3 个 seed 上新训练；选中 true 的直接复用 E1 的正式拟合。
- 全部拟合冻结后只评估一次；统计同 E1，不做 Holm。
- 报告补两行参照：三个设定的 PersistenceHour 和 PersistenceLast10，分 h 列出，论文的基线表要带上它们。
- 不改其他超参数，不加新模型。

**B 线**：规则同 E1 B 线原文（测试年只按文件名登记、只记哈希；MIDAS 门槛：合格站 ≥8、有气压 ≥5、覆盖 ≥3 个象限）。
- 下载和安装都由 Cursor 做，Codex 只做只读核验。
- 下载没完成就跳过，不催、不代下。

**L1 草案（只有 H0 = GO 才展开；本阶段不实现，只请 Codex 预先看方向有没有硬伤）**
- **搜索对象**：可微的"上游编码程序"。输入站点张量 [B, 24, S, F] 和几何信息，通过固定接口声明 ≤16 个可学习标量，输出 ≤8 个通道，与 TSMixer 端到端训练。
- **对照组（预算相同：每组 40 个程序，每个 1 seed、短训练）**：
  - H：11 个手工特征；
  - P：专家给定的参数化族（软上风锥、距离衰减、平流延迟核），参数可学习；
  - R：在 P 的离散结构选项里随机搜索；
  - L：大模型进化搜索，只用验证 MAE 反馈；
  - L-blind：变量名匿名、不给物理说明的大模型。
- **选择与检验**：在 Kelmarsh 开发年（拟合 ≤2021，验证 2022）上选；外部检查时结构冻结、参数重训，看 Penmanshiel T22 折验证年 2021，每组 top-1 跑 3 个 seed；确认只在 Hill of Towie 封存的 2024、2025 上做一次。
- **主张规则**：
  - 只有 L 在外部检查上同时赢过 R 和 P，才写"大模型"的贡献；
  - 只是 P 赢过 H，只能写"可学习编码器"（K1b 已有类似的失败结果，预期 P ≈ H）；
  - 部署时的模型里没有大模型。
- **近邻**：LLM-SR（ICLR 2025）、LLM-FE（2503.14434）、ELATE（2508.14667）、EFE-Time（2607.01548）、CAAFE（NeurIPS 2023）；Neshat 等 2026（ECM 368:121974，大模型在线推断风向相关的动态图）。我们的区别：大模型只在离线阶段写可微程序，并做跨风场检验。

## 实验配置表

| 线 | 内容 | 规模 |
|---|---|---|
| H0 | 2 个风场 × 2 个方向 × 3 个特征集 × 2 种损失 = 24 次 GBM 拟合；另加 2 个探针年的冻结推理（各 3 seed） | CPU ≤30 分钟（待确认），1 个实例 |
| A2 | 搜索 8 次；正式拟合 ≤36 次（只给选中 use_norm=false 的组合） | ≤44 次拟合，约 2 小时（按 E1 速度估计，待确认），1 个实例 |
| B | 只读核验 | 不训练；取决于 Cursor 下载进度 |

- H0 和 A2 互不依赖，可以并行。
- H0 的判定写进结果文件之前，代码、阈值和特征清单都先冻结并登记哈希。

## 改动文件清单（= HANDOFF「允许改动」）

- H0：`scripts/probe_upstream_headroom_l0.py`、`tests/test_probe_upstream_headroom_l0.py`、`configs/l0_headroom_protocol.json`，结果放 `output/l0_headroom/**`。
  - 测试要覆盖：站点特征只用 ≤ 起报 − 1 h 的观测；交叉拟合时 A 年和 B 年的窗口不重叠；余量和判定的计算正确（合成数据）。
- A2：`scripts/train_e1_baselines_a2.py`（新文件，导入冻结的 train_e1_baselines.py 并只覆盖 use_norm；原文件已登记哈希，不能改）、`tests/test_train_e1_baselines_a2.py`、`configs/e1_baselines_protocol_a2.json`，结果放 `output/e1_baselines/a2/**`。
- B：`scripts/audit_hill_of_towie_e1.py`、`output/hill_of_towie_e1/**`、`output/external_acquisition/hill_of_towie/**`（只读核验和登记）。
- 不拷权重、数组和原始数据。

## 本阶段不做什么

- 不调用任何大模型，不写 L1 代码，不开新的方法实验。
- 不碰 Hill of Towie 的测试年内容，不在 Hill of Towie 上训练或评估。
- 不改已冻结的 E1、v31、v32、P3 文件和结果；需要改就开新版本。
- 不按 H0 的结果事后改阈值，不追加特征集或调学习器。
- A2 只改 use_norm 一项，不扩大搜索。

## 风险（≤5 条）

1. H0 是残差探针，不是端到端编码器：学习器能力有限时会低估余量。用 b1/b2 取优和按年交叉拟合缓解；STOP 的结论措辞写成"本探针未发现"。
2. b1 维数高（Kelmarsh 约 540 维），有效样本只有约 8.7k 个起报时刻，可能过拟合；所以加了低维的 b2 并取较好者。
3. v31 模型补推理 2024 时，scaler 和通道必须用它自己的，不能借 v32 的。
4. A2 若 use_norm=false 明显更好，E1 A 线表要换成 A2 的结果，论文里如实说明选择过程。
5. 我估计 H0 = GO 的概率只有两三成。不通过也有价值：它解释了此前 5 次方法尝试为什么都没赢过手工特征。

## 存档引用

- 用户提议和我的判断：DECISIONS.md 的 L0 一节。
- 已有的可学习编码器失败记录：DECISIONS.md 的 K1b 一节；手工特征的定义在 `scripts/screen_kelmarsh_upstream_observations.py:213`。
- E1 A 线结果：`output/e1_baselines/RESULT_SUMMARY.md`、`output/e1_baselines/eval1/result.json`。
- 主结论：Kelmarsh v31/v32（`configs/kelmarsh_upstream_protocol_v3{1,2}.json`）；Penmanshiel P3（`output/penmanshiel_hourly/p3_v35/r2/`）。
