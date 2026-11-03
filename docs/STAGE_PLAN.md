# STAGE_PLAN — 只写下一个阶段

作者：Claude（Codex 只提意见，不改本文件）。上限 150 行：要加新内容时，先删旧内容。
内容只从现有代码和结果中提取；拿不准的地方标「待确认」；事先设定的阈值标「事先设定」。
长方案 `docs/UPSTREAM_INNOVATION_PROPOSAL.md` 已改为存档，只按章节号引用，不再整份送审。

## 阶段名

v35 P1：Penmanshiel 开发年数据准备（不训练）。

背景：
- D2-0 的数据闸门（A1–A3）通过，曲线闸门（B1）不通过，所以固定曲线分解路线停止。
- 用户 2026-11-02 选定"第二风场复现"：在 Penmanshiel 上检验已确认的主贡献（v30–v32 的上游 MIDAS 特征）能否复现，并做 Kelmarsh → Penmanshiel 的迁移。
- P1 是这条线的第一步，只把数据准备好。

## 目标

1. 按 Kelmarsh v31/v32 完全相同的口径，生成 Penmanshiel 开发年（2016–2021）的逐时增强数据。上游 11 个特征改用 Penmanshiel 当地的站网计算。
2. 用和 Kelmarsh 同名、同顺序的 21 个通道组成序列，确认窗口数量和缺测率足够做后续实验。
3. 测试年 2022（以及受保护的 2023、2024）继续封存，不解析任何内容。

## 假设与判定标准

**口径（全部沿用现有冻结代码，只换风场参数）**

- SCADA 处理：
  - 复用 `scripts/prepare_kelmarsh_multiyear.py` 加 v31 的重复行规范化（`prepare_kelmarsh_multiyear_v31.py`）：严格 6×10 分钟均值、同一套 `utils/kelmarsh_enrichment.RULES` 运行状态规则。
  - 标签不改、不截断、不插补。
- 风场参数（来自 D2-0 的 part_A）：
  - 归档 `penmanshiel_16807304`；风机 1、2、4–15，共 14 台 MM82，额定 2050 kW；
  - 列名对照沿用 D2-0 已通过的 A2 映射。
- 起始时间：取所有 14 台风机都有完整月度数据的第一个日历月的 1 日（规则事先设定，具体日期待确认，由 Codex 报告）。
- 折与曲线年（与 Kelmarsh 的 D1 折同形）：
  - 拟合 2016（从起始时间开始）–2019，验证 2020，开发评估 2021；
  - 功率曲线只用拟合年 2016–2019 拟合，和 Kelmarsh 的 curve years 一致。
- 上游特征：
  - 站网：冻结为 D2-0 判定合格的 11 个站，src_id 为 150、161、177、235、982、987、1023、1074、16589、24125、30523。不重新选站。
  - 计算：沿用 `upstream_features` / `hourly_upstream` 的定义，滞后 1 h。
  - 缺测：沿用 v32 `KelmarshUpstreamSeries` 的做法，只在输入端用训练年均值填补，并加 `upstream_missing` 标志。
  - 站点文件只读开发年；2022 年起的文件只记哈希。
- 序列：21 个通道，名称和顺序与 Kelmarsh 相同；输入 24、预测 4、步长 3；scaler 只用拟合年。

**继续条件（四条都满足，阈值为事先设定）**

- **Q1 通道一致**：21 个通道全部生成，名称和顺序与 Kelmarsh 完全相同；14 台风机都在序列里。
- **Q2 窗口够用**：拟合、验证、开发评估三段的 ALL 窗口数各 ≥5000（Kelmarsh D1 折为 75024 / 15680 / 16949）。
- **Q3 上游覆盖**：每个开发年中，`upstream_missing` 的行占比 ≤20%。
- **Q4 标签未动**：标签被修改、截断或插补的格子数为 0（审计方式同 v31：只选已有的整行，不填格子）。

不满足就停下，写清是哪一条、差多少，交用户决定。

## 实验配置表

| 项目 | 内容 |
|---|---|
| 运行 | 1 个数据准备实例，不训练，不用 GPU，预算 ≤60 分钟（待确认，14 台风机 × 6 年） |
| 读取 | Penmanshiel 2016–2021 年 SCADA；11 个站 2016–2021 年的 MIDAS 文件 |
| 只记哈希 | Penmanshiel 2022–2024 年 SCADA；上述站点 2022 年起的文件 |
| 服务器产出 | 逐时增强 CSV 和上游特征表，放在 `datasets/` 下的新目录（v35），不覆盖 Kelmarsh 的任何数据 |
| 拷回本地 | 只拷元数据：准备记录（哈希、规则、曲线、起始时间）；每年质量表；序列审计（窗口数、各通道有限率、`upstream_missing` 占比）；Q1–Q4 结论。不拷数据本身 |

## 改动文件清单（= HANDOFF「允许改动」）

- `scripts/prepare_penmanshiel_v35.py`：新文件，Codex 写。只做参数化和调用现有函数，不复制、不改写 Kelmarsh 的冻结逻辑。
- `utils/penmanshiel_v35.py`：可选，Codex 写。放风场参数、站网清单和序列组装。
- `tests/test_prepare_penmanshiel_v35.py`：Codex 写。用合成数据覆盖四类情况：缺风机号、全空行、封存年拒读、通道顺序。
- `output/penmanshiel_hourly/prep_v35/**`：Codex 拷回的元数据。
- 只读不改：Kelmarsh 的全部准备脚本、`utils/kelmarsh_*`、`data_provider/kelmarsh_multiyear.py`、冻结协议，以及 D2-0 的审计文件。

## 本阶段不做什么

- 不训练任何模型，不做迁移。
- 不解析 2022 年及以后的任何 Penmanshiel 或站点内容。
- 不下载新数据。
- 不改 Kelmarsh 的任何冻结代码或数据。
- 不重新选站，不调特征定义，不改运行状态规则。

## 风险（≤5 条）

1. 最近的站在 63 km 外，而 Kelmarsh 是 27 km。`local_ws` 这类特征的意义会变弱，P2 可能复现不了上游增益。这正是要检验的问题，不在 P1 修正。
2. Penmanshiel 的导出格式可能有 Kelmarsh 没出现过的问题，例如重复行、时区、空行（D2-0 已遇到全空行）。只按 v31 的"选整行、不填格子"规则处理，处理不了就停。
3. 2016 年是投运年，起始月的规则可能让拟合段明显变短（待确认）。
4. 有一个站（982）没有气压。梯度特征需要至少 5 个有气压的站，目前有 10 个，够用。
5. Kelmarsh 的准备代码里可能写死了 Kelmarsh 专用的路径或常量。参数化时必须逐个列出改了哪些，不能悄悄改行为。

## 存档引用

- 已确认的主贡献：Kelmarsh v30–v32（2023 和 2024 两个测试年）；协议 `configs/kelmarsh_upstream_protocol_v32.json`。
- D2-0 结果：`output/kelmarsh_hourly/d2_feasibility_v34_r2/`，其中 part_A.json 有合格站表和列名映射，test_year_registration.json 是测试年登记。
- 负结果记录：K1（D3）、K1b（E2）、D2-0（B1 不通过），见 DECISIONS.md 各阶段。
- 后续方向（细节暂缓，到时再写）：
  - P2：在 Penmanshiel 开发折上做两件事。一是本场训练的 T9 与 T21 对比，看上游增益能否复现；二是 Kelmarsh 训练的 T21 零样本迁移，以及少样本微调。
  - P3：冻结设计后，在 2022 年做一次性测试。
