# 三场风电预测成绩清单

第 113 次交接，2026-11-24。下列 464 个条目对应已归档的评估集、模型和条件。同一模型在多个延迟条件的记录，或在另一份报告中的复用，不构成额外独立重复。正式期刊对照另待运行回执。

## 阅读口径

- A 表示可在对应风场、相同窗口及切分内配对；同时核对信息集和训练口径，才能解释模型差异。B 表示同年但输入或上下文有别，只能并列。C 为历史参照。
- 确认性指使用首次打开的测试包。HoT 两年共用冻结模型，N 对 B 的主检验按 N 的前三个种子计算。G2a 在已打开的 K/P 测试包上补做，单列为探索性。等级与证据性质分别标注。
- 训练栏 K、P、H 分别为本场 Kelmarsh（2016–2020）、Penmanshiel（2016–2019）、Hill of Towie（2016–2023 年 6 月，按冻结规则处理改造期）的拟合数据。V1/V2 是相邻的开发段，V2 日期逐行标注。测试集选择配置使用各原协议。
- L0 仅本场，LU 加上游站，LG 加 GFS，LUG 两者都加。24 h 指原始历史窗口，树模型按原协议取特征；完整 24 h 展平树模型单列于 WJB。Q2 微调每场固定抽取 20,000 个拟合窗口。X3 集成使用长上下文分量与短历史树模型，树模型对照集成沿用自身输入。
- 指标是机组级 1–4 h 汇总。MAE、RMSE 单位 kW，R² 无量纲。多种子使用各自指标的均值；明确标注的集成用融合预测计分。堆叠的种子栏表示一份聚合预测，成员各自有种子。零样本、持续法和确定性功率曲线记 0 个新增随机拟合种子。
- “待核”表示相应回执没有给出可直接抄录的字段。已有哈希记录可追溯汇总与原始预测登记，此次未逐窗重算所有预测。X5、X5b 和 X3 的部分回执仅给 MAE。
- X5 同延迟训练、固定 6 h 模型在另一条件推理（d6@）和混合延迟模型（mix@）分列。X5b 的 A/C 是原样 6/15 h，D 是新周期按旧周期覆盖截短，Cc 是旧周期按新周期覆盖限制。18 h 条件下四个未来时效都超出已取 f018。

## 首次测试及同测试包上的补充拆解

| 模型 | 来源 | 风场 | 评估 | 训练 | 输入 | 种子 | 等级 | 证据 | MAE | RMSE | R² |
|---|---|---|---|---|---|---:|---|---|---:|---:|---:|
| TSMixer B | G1 | Kelmarsh | 2022 | K | LU / 24 h | 3 | A | 确认性 | 160.37 | 242.91 | 0.8242 |
| TSMixer N | G1 | Kelmarsh | 2022 | K | LUG / 24 h | 3 | A | 确认性 | 146.46 | 220.69 | 0.8549 |
| TSMixer B | G1 | Penmanshiel | 2021 | P | LU / 24 h | 3 | A | 确认性 | 184.79 | 299.54 | 0.7846 |
| TSMixer N | G1 | Penmanshiel | 2021 | P | LUG / 24 h | 3 | A | 确认性 | 168.98 | 274.21 | 0.8195 |
| TSMixer B0 | G2a | Kelmarsh | 2022 | K | L0 / 24 h | 3 | A | 探索性 | 180.15 | 276.16 | 0.7728 |
| TSMixer N0 | G2a | Kelmarsh | 2022 | K | LG / 24 h | 3 | A | 探索性 | 147.05 | 221.75 | 0.8535 |
| TSMixer B0 | G2a | Penmanshiel | 2021 | P | L0 / 24 h | 3 | A | 探索性 | 195.87 | 319.68 | 0.7547 |
| TSMixer N0 | G2a | Penmanshiel | 2021 | P | LG / 24 h | 3 | A | 探索性 | 170.73 | 282.07 | 0.8090 |
| TSMixer B | G2c | HoT | 2024 | H | LU / 24 h | 3 | A | 确认性 | 215.67 | 351.21 | 0.7270 |
| TSMixer N0 | G2c | HoT | 2024 | H | LG / 24 h | 3 | A | 确认性 | 201.67 | 333.11 | 0.7544 |
| TSMixer B0 | G2c | HoT | 2024 | H | L0 / 24 h | 3 | A | 确认性 | 228.88 | 373.45 | 0.6913 |
| TSMixer N | G2c | HoT | 2024 | H | LUG / 24 h | 5 | A | 确认性 | 200.56 | 330.31 | 0.7585 |
| TSMixer N（前三种子） | G2c | HoT | 2024 | H | LUG / 24 h | 3 | A | 确认性 | 200.76 | 331.38 | 0.7570 |
| TSMixer N（五种子预测集成） | G2c | HoT | 2024 | H | LUG / 24 h | 5 | A | 确认性 | 197.28 | 326.04 | 0.7647 |
| TSMixer B | G2c | HoT | 2025 | H | LU / 24 h | 3 | A | 确认性 | 208.11 | 342.19 | 0.7326 |
| TSMixer N0 | G2c | HoT | 2025 | H | LG / 24 h | 3 | A | 确认性 | 196.05 | 326.99 | 0.7558 |
| TSMixer B0 | G2c | HoT | 2025 | H | L0 / 24 h | 3 | A | 确认性 | 225.25 | 370.85 | 0.6860 |
| TSMixer N | G2c | HoT | 2025 | H | LUG / 24 h | 5 | A | 确认性 | 193.03 | 320.94 | 0.7648 |
| TSMixer N（前三种子） | G2c | HoT | 2025 | H | LUG / 24 h | 3 | A | 确认性 | 193.06 | 321.38 | 0.7641 |
| TSMixer N（五种子预测集成） | G2c | HoT | 2025 | H | LUG / 24 h | 5 | A | 确认性 | 190.10 | 316.96 | 0.7706 |

## 与测试年匹配的外部基线

| 模型 | 来源 | 风场 | 评估 | 训练 | 输入 | 种子 | 等级 | 证据 | MAE | RMSE | R² |
|---|---|---|---|---|---|---:|---|---|---:|---:|---:|
| 持续法 | X1-A0 | Kelmarsh | 2022 | 无本场拟合 | 末次功率 | 0 | A | 探索性 | 193.87 | 301.87 | 0.7285 |
| 持续法 | X1-A0 | Penmanshiel | 2021 | 无本场拟合 | 末次功率 | 0 | A | 探索性 | 206.07 | 348.14 | 0.7091 |
| 持续法 | X1-A0 | HoT | 2024 | 无本场拟合 | 末次功率 | 0 | A | 探索性 | 245.97 | 409.62 | 0.6286 |
| 持续法 | X1-A0 | HoT | 2025 | 无本场拟合 | 末次功率 | 0 | A | 探索性 | 241.98 | 404.81 | 0.6258 |
| LightGBM A1 | X1-A1 | Kelmarsh | 2022 | K | LUG / 24 h | 1 | A | 探索性 | 145.25 | 222.06 | 0.8531 |
| LightGBM A1 | X1-A1 | Penmanshiel | 2021 | P | LUG / 24 h | 1 | A | 探索性 | 161.01 | 270.46 | 0.8244 |
| LightGBM A1 | X1-A1 | HoT | 2024 | H | LUG / 24 h | 1 | A | 探索性 | 194.40 | 330.07 | 0.7589 |
| LightGBM A1 | X1-A1 | HoT | 2025 | H | LUG / 24 h | 1 | A | 探索性 | 186.72 | 319.59 | 0.7668 |
| GFS 功率曲线 | X1-A2 | Kelmarsh | 2022 | K | GFS 风速＋逐机功率曲线 | 0 | A | 探索性 | 219.50 | 336.18 | 0.6633 |
| GFS 功率曲线 | X1-A2 | Penmanshiel | 2021 | P | GFS 风速＋逐机功率曲线 | 0 | A | 探索性 | 242.00 | 397.62 | 0.6205 |
| GFS 功率曲线 | X1-A2 | HoT | 2024 | H | GFS 风速＋逐机功率曲线 | 0 | A | 探索性 | 314.95 | 502.11 | 0.4420 |
| GFS 功率曲线 | X1-A2 | HoT | 2025 | H | GFS 风速＋逐机功率曲线 | 0 | A | 探索性 | 291.14 | 469.83 | 0.4959 |
| Chronos-2 零样本 | X1-A3 | Kelmarsh | 2022 | 无本场拟合 | 含 GFS；24 h | 0 | B | 探索性 | 190.07 | 292.44 | 0.7452 |
| Chronos-2 零样本 | X1-A3 | Kelmarsh | 2022 | 无本场拟合 | 含 GFS；168 h | 0 | B | 探索性 | 163.54 | 245.81 | 0.8200 |
| Chronos-2 零样本 | X1-A3 | Penmanshiel | 2021 | 无本场拟合 | 含 GFS；24 h | 0 | B | 探索性 | 205.94 | 339.01 | 0.7241 |
| Chronos-2 零样本 | X1-A3 | Penmanshiel | 2021 | 无本场拟合 | 含 GFS；168 h | 0 | B | 探索性 | 185.39 | 303.35 | 0.7791 |
| Chronos-2 零样本 | X1-A3 | HoT | 2024 | 无本场拟合 | 含 GFS；24 h | 0 | B | 探索性 | 248.13 | 403.30 | 0.6400 |
| Chronos-2 零样本 | X1-A3 | HoT | 2024 | 无本场拟合 | 含 GFS；168 h | 0 | B | 探索性 | 223.94 | 365.24 | 0.7048 |
| Chronos-2 零样本 | X1-A3 | HoT | 2025 | 无本场拟合 | 含 GFS；24 h | 0 | B | 探索性 | 241.79 | 393.97 | 0.6456 |
| Chronos-2 零样本 | X1-A3 | HoT | 2025 | 无本场拟合 | 含 GFS；168 h | 0 | B | 探索性 | 217.72 | 359.22 | 0.7054 |

## X2 开发段底座和信息臂

| 模型 | 来源 | 风场 | 评估 | 训练 | 输入 | 种子 | 等级 | 证据 | MAE | RMSE | R² |
|---|---|---|---|---|---|---:|---|---|---:|---:|---:|
| LightGBM L0 | X2-dev | Kelmarsh | V2 2021下半年 | K | L0 / 24 h | 1 | A | 开发期 | 163.00 | 259.22 | 0.7927 |
| LightGBM LU | X2-dev | Kelmarsh | V2 2021下半年 | K | LU / 24 h | 1 | A | 开发期 | 145.31 | 229.35 | 0.8377 |
| LightGBM LG | X2-dev | Kelmarsh | V2 2021下半年 | K | LG / 24 h | 1 | A | 开发期 | 134.38 | 208.02 | 0.8665 |
| LightGBM LUG | X2-dev | Kelmarsh | V2 2021下半年 | K | LUG / 24 h | 3 | A | 开发期 | 132.05 | 204.70 | 0.8707 |
| LightGBM LUG_PCL | X2-dev | Kelmarsh | V2 2021下半年 | K | LUG / 24 h | 3 | A | 开发期 | 131.70 | 204.34 | 0.8712 |
| LightGBM LUG_ANI | X2-dev | Kelmarsh | V2 2021下半年 | K | LUG / 24 h | 3 | A | 开发期 | 131.94 | 204.85 | 0.8705 |
| LightGBM L0 | X2-dev | Penmanshiel | V2 2020下半年 | P | L0 / 24 h | 1 | A | 开发期 | 224.21 | 352.97 | 0.7091 |
| LightGBM LU | X2-dev | Penmanshiel | V2 2020下半年 | P | LU / 24 h | 1 | A | 开发期 | 205.15 | 327.17 | 0.7501 |
| LightGBM LG | X2-dev | Penmanshiel | V2 2020下半年 | P | LG / 24 h | 1 | A | 开发期 | 178.63 | 280.99 | 0.8156 |
| LightGBM LUG | X2-dev | Penmanshiel | V2 2020下半年 | P | LUG / 24 h | 3 | A | 开发期 | 176.28 | 276.94 | 0.8209 |
| LightGBM LUG_PCL | X2-dev | Penmanshiel | V2 2020下半年 | P | LUG / 24 h | 3 | A | 开发期 | 176.04 | 276.84 | 0.8210 |
| LightGBM LUG_ANI | X2-dev | Penmanshiel | V2 2020下半年 | P | LUG / 24 h | 3 | A | 开发期 | 175.27 | 277.58 | 0.8201 |
| LightGBM L0 | X2-dev | HoT | V2 2023第4季度 | H | L0 / 24 h | 1 | A | 开发期 | 244.38 | 400.15 | 0.7462 |
| LightGBM LU | X2-dev | HoT | V2 2023第4季度 | H | LU / 24 h | 1 | A | 开发期 | 217.16 | 359.32 | 0.7954 |
| LightGBM LG | X2-dev | HoT | V2 2023第4季度 | H | LG / 24 h | 1 | A | 开发期 | 188.00 | 320.52 | 0.8372 |
| LightGBM LUG | X2-dev | HoT | V2 2023第4季度 | H | LUG / 24 h | 3 | A | 开发期 | 186.39 | 317.51 | 0.8402 |
| LightGBM LUG_PCL | X2-dev | HoT | V2 2023第4季度 | H | LUG / 24 h | 3 | A | 开发期 | 186.92 | 318.23 | 0.8395 |
| LightGBM LUG_ANI | X2-dev | HoT | V2 2023第4季度 | H | LUG / 24 h | 3 | A | 开发期 | 187.20 | 318.03 | 0.8397 |
| TSMixer N_v1 | X2-dev | Kelmarsh | V2 2021下半年 | K | LUG / 24 h | 3 | A | 开发期 | 133.40 | 待核 | 待核 |
| TSMixer N_v1 | X2-dev | Penmanshiel | V2 2020下半年 | P | LUG / 24 h | 3 | A | 开发期 | 182.98 | 待核 | 待核 |
| TSMixer N_v1 | X2-dev | HoT | V2 2023第4季度 | H | LUG / 24 h | 3 | A | 开发期 | 198.44 | 待核 | 待核 |
| TCN-GRU | X2-dev | Kelmarsh | V2 2021下半年 | K | LUG / 24 h | 3 | A | 开发期 | 140.39 | 待核 | 待核 |
| TCN-GRU | X2-dev | Penmanshiel | V2 2020下半年 | P | LUG / 24 h | 3 | A | 开发期 | 193.82 | 待核 | 待核 |
| TCN-GRU | X2-dev | HoT | V2 2023第4季度 | H | LUG / 24 h | 3 | A | 开发期 | 218.08 | 待核 | 待核 |
| TCN-GRU 域对抗 | X2-dev | Kelmarsh | V2 2021下半年 | K | LUG / 24 h | 3 | A | 开发期 | 139.97 | 待核 | 待核 |
| TCN-GRU 域对抗 | X2-dev | Penmanshiel | V2 2020下半年 | P | LUG / 24 h | 3 | A | 开发期 | 193.10 | 待核 | 待核 |
| TCN-GRU 域对抗 | X2-dev | HoT | V2 2023第4季度 | H | LUG / 24 h | 3 | A | 开发期 | 215.31 | 待核 | 待核 |
| CNN＋注意力 | X2-dev | Kelmarsh | V2 2021下半年 | K | LUG / 24 h | 3 | A | 开发期 | 139.37 | 待核 | 待核 |
| CNN＋注意力 | X2-dev | Penmanshiel | V2 2020下半年 | P | LUG / 24 h | 3 | A | 开发期 | 192.22 | 待核 | 待核 |
| CNN＋注意力 | X2-dev | HoT | V2 2023第4季度 | H | LUG / 24 h | 3 | A | 开发期 | 201.01 | 待核 | 待核 |
| BiLSTM＋注意力 | X2-dev | Kelmarsh | V2 2021下半年 | K | LUG / 24 h | 3 | A | 开发期 | 137.54 | 待核 | 待核 |
| BiLSTM＋注意力 | X2-dev | Penmanshiel | V2 2020下半年 | P | LUG / 24 h | 3 | A | 开发期 | 195.32 | 待核 | 待核 |
| BiLSTM＋注意力 | X2-dev | HoT | V2 2023第4季度 | H | LUG / 24 h | 3 | A | 开发期 | 199.94 | 待核 | 待核 |
| CNN＋BiLSTM＋LightGBM 堆叠 | X2-dev | Kelmarsh | V2 2021下半年 | K | LUG / 24 h | 1 | A | 开发期 | 132.31 | 待核 | 待核 |
| CNN＋BiLSTM＋LightGBM 堆叠 | X2-dev | Penmanshiel | V2 2020下半年 | P | LUG / 24 h | 1 | A | 开发期 | 184.35 | 待核 | 待核 |
| CNN＋BiLSTM＋LightGBM 堆叠 | X2-dev | HoT | V2 2023第4季度 | H | LUG / 24 h | 1 | A | 开发期 | 199.25 | 待核 | 待核 |
| LightGBM LUG_VN | X2A2-dev | Kelmarsh | V2 2021下半年 | K | LUG / 24 h | 3 | A | 开发期 | 131.62 | 203.89 | 0.8717 |
| LightGBM LUG_VR | X2A2-dev | Kelmarsh | V2 2021下半年 | K | LUG / 24 h | 3 | A | 开发期 | 132.07 | 204.91 | 0.8704 |
| LightGBM LUG_VB | X2A2-dev | Kelmarsh | V2 2021下半年 | K | LUG / 24 h | 3 | A | 开发期 | 131.19 | 202.33 | 0.8737 |
| LightGBM LUG_VN | X2A2-dev | Penmanshiel | V2 2020下半年 | P | LUG / 24 h | 3 | A | 开发期 | 177.37 | 282.77 | 0.8133 |
| LightGBM LUG_VR | X2A2-dev | Penmanshiel | V2 2020下半年 | P | LUG / 24 h | 3 | A | 开发期 | 175.49 | 279.14 | 0.8181 |
| LightGBM LUG_VB | X2A2-dev | Penmanshiel | V2 2020下半年 | P | LUG / 24 h | 3 | A | 开发期 | 176.60 | 278.32 | 0.8191 |
| LightGBM LUG_VN | X2A2-dev | HoT | V2 2023第4季度 | H | LUG / 24 h | 3 | A | 开发期 | 186.56 | 317.03 | 0.8407 |
| LightGBM LUG_VR | X2A2-dev | HoT | V2 2023第4季度 | H | LUG / 24 h | 3 | A | 开发期 | 188.07 | 318.97 | 0.8388 |
| LightGBM LUG_VB | X2A2-dev | HoT | V2 2023第4季度 | H | LUG / 24 h | 3 | A | 开发期 | 187.38 | 318.35 | 0.8394 |

## X2 测试年探索性比较

| 模型 | 来源 | 风场 | 评估 | 训练 | 输入 | 种子 | 等级 | 证据 | MAE | RMSE | R² |
|---|---|---|---|---|---|---:|---|---|---:|---:|---:|
| LightGBM L0 | X2 | Kelmarsh | 2022 | K | L0 / 24 h | 1 | A | 探索性 | 181.15 | 279.33 | 0.7675 |
| LightGBM LU | X2 | Kelmarsh | 2022 | K | LU / 24 h | 1 | A | 探索性 | 160.63 | 246.95 | 0.8183 |
| LightGBM LG | X2 | Kelmarsh | 2022 | K | LG / 24 h | 1 | A | 探索性 | 147.86 | 226.43 | 0.8472 |
| LightGBM LUG | X2 | Kelmarsh | 2022 | K | LUG / 24 h | 3 | A | 探索性 | 145.34 | 222.19 | 0.8529 |
| LightGBM LUG_PCL | X2 | Kelmarsh | 2022 | K | LUG / 24 h | 3 | A | 探索性 | 144.92 | 221.98 | 0.8532 |
| LightGBM LUG_ANI | X2 | Kelmarsh | 2022 | K | LUG / 24 h | 3 | A | 探索性 | 145.33 | 222.60 | 0.8524 |
| TSMixer N_v1 | X2 | Kelmarsh | 2022 | K | LUG / 24 h | 3 | A | 探索性 | 147.18 | 222.28 | 0.8528 |
| TCN-GRU | X2 | Kelmarsh | 2022 | K | LUG / 24 h | 3 | A | 探索性 | 153.50 | 235.49 | 0.8347 |
| TCN-GRU 域对抗 | X2 | Kelmarsh | 2022 | K | LUG / 24 h | 3 | A | 探索性 | 152.77 | 233.84 | 0.8371 |
| CNN＋注意力 | X2 | Kelmarsh | 2022 | K | LUG / 24 h | 3 | A | 探索性 | 153.30 | 229.24 | 0.8434 |
| BiLSTM＋注意力 | X2 | Kelmarsh | 2022 | K | LUG / 24 h | 3 | A | 探索性 | 150.76 | 227.23 | 0.8461 |
| CNN＋BiLSTM＋LightGBM 堆叠 | X2 | Kelmarsh | 2022 | K | LUG / 24 h | 1 | A | 探索性 | 144.66 | 216.64 | 0.8602 |
| LightGBM L0 | X2 | Penmanshiel | 2021 | P | L0 / 24 h | 1 | A | 探索性 | 190.31 | 318.91 | 0.7558 |
| LightGBM LU | X2 | Penmanshiel | 2021 | P | LU / 24 h | 1 | A | 探索性 | 176.47 | 296.82 | 0.7885 |
| LightGBM LG | X2 | Penmanshiel | 2021 | P | LG / 24 h | 1 | A | 探索性 | 163.72 | 275.41 | 0.8179 |
| LightGBM LUG | X2 | Penmanshiel | 2021 | P | LUG / 24 h | 3 | A | 探索性 | 160.84 | 270.24 | 0.8247 |
| LightGBM LUG_PCL | X2 | Penmanshiel | 2021 | P | LUG / 24 h | 3 | A | 探索性 | 161.02 | 271.50 | 0.8230 |
| LightGBM LUG_ANI | X2 | Penmanshiel | 2021 | P | LUG / 24 h | 3 | A | 探索性 | 161.45 | 272.83 | 0.8213 |
| TSMixer N_v1 | X2 | Penmanshiel | 2021 | P | LUG / 24 h | 3 | A | 探索性 | 168.98 | 274.21 | 0.8195 |
| TCN-GRU | X2 | Penmanshiel | 2021 | P | LUG / 24 h | 3 | A | 探索性 | 178.93 | 299.22 | 0.7851 |
| TCN-GRU 域对抗 | X2 | Penmanshiel | 2021 | P | LUG / 24 h | 3 | A | 探索性 | 177.76 | 295.37 | 0.7906 |
| CNN＋注意力 | X2 | Penmanshiel | 2021 | P | LUG / 24 h | 3 | A | 探索性 | 179.16 | 296.00 | 0.7897 |
| BiLSTM＋注意力 | X2 | Penmanshiel | 2021 | P | LUG / 24 h | 3 | A | 探索性 | 182.97 | 300.65 | 0.7829 |
| CNN＋BiLSTM＋LightGBM 堆叠 | X2 | Penmanshiel | 2021 | P | LUG / 24 h | 1 | A | 探索性 | 172.68 | 267.03 | 0.8288 |
| LightGBM L0 | X2 | HoT | 2024 | H | L0 / 24 h | 1 | A | 探索性 | 229.15 | 378.72 | 0.6826 |
| LightGBM LU | X2 | HoT | 2024 | H | LU / 24 h | 1 | A | 探索性 | 212.17 | 350.04 | 0.7288 |
| LightGBM LG | X2 | HoT | 2024 | H | LG / 24 h | 1 | A | 探索性 | 197.58 | 335.18 | 0.7514 |
| LightGBM LUG | X2 | HoT | 2024 | H | LUG / 24 h | 3 | A | 探索性 | 194.10 | 328.57 | 0.7611 |
| LightGBM LUG_PCL | X2 | HoT | 2024 | H | LUG / 24 h | 3 | A | 探索性 | 194.21 | 328.89 | 0.7606 |
| LightGBM LUG_ANI | X2 | HoT | 2024 | H | LUG / 24 h | 3 | A | 探索性 | 194.45 | 329.11 | 0.7603 |
| TSMixer N_v1 | X2 | HoT | 2024 | H | LUG / 24 h | 3 | A | 探索性 | 199.40 | 328.26 | 0.7615 |
| TCN-GRU | X2 | HoT | 2024 | H | LUG / 24 h | 3 | A | 探索性 | 222.89 | 364.14 | 0.7065 |
| TCN-GRU 域对抗 | X2 | HoT | 2024 | H | LUG / 24 h | 3 | A | 探索性 | 220.94 | 364.64 | 0.7057 |
| CNN＋注意力 | X2 | HoT | 2024 | H | LUG / 24 h | 3 | A | 探索性 | 202.42 | 335.40 | 0.7510 |
| BiLSTM＋注意力 | X2 | HoT | 2024 | H | LUG / 24 h | 3 | A | 探索性 | 204.79 | 339.26 | 0.7452 |
| CNN＋BiLSTM＋LightGBM 堆叠 | X2 | HoT | 2024 | H | LUG / 24 h | 1 | A | 探索性 | 203.94 | 321.90 | 0.7707 |
| LightGBM L0 | X2 | HoT | 2025 | H | L0 / 24 h | 1 | A | 探索性 | 225.65 | 376.73 | 0.6759 |
| LightGBM LU | X2 | HoT | 2025 | H | LU / 24 h | 1 | A | 探索性 | 203.89 | 341.56 | 0.7336 |
| LightGBM LG | X2 | HoT | 2025 | H | LG / 24 h | 1 | A | 探索性 | 193.39 | 330.14 | 0.7511 |
| LightGBM LUG | X2 | HoT | 2025 | H | LUG / 24 h | 3 | A | 探索性 | 186.73 | 318.55 | 0.7683 |
| LightGBM LUG_PCL | X2 | HoT | 2025 | H | LUG / 24 h | 3 | A | 探索性 | 186.56 | 318.28 | 0.7687 |
| LightGBM LUG_ANI | X2 | HoT | 2025 | H | LUG / 24 h | 3 | A | 探索性 | 186.96 | 319.25 | 0.7673 |
| TSMixer N_v1 | X2 | HoT | 2025 | H | LUG / 24 h | 3 | A | 探索性 | 192.86 | 320.32 | 0.7657 |
| TCN-GRU | X2 | HoT | 2025 | H | LUG / 24 h | 3 | A | 探索性 | 213.03 | 353.11 | 0.7153 |
| TCN-GRU 域对抗 | X2 | HoT | 2025 | H | LUG / 24 h | 3 | A | 探索性 | 210.97 | 350.52 | 0.7194 |
| CNN＋注意力 | X2 | HoT | 2025 | H | LUG / 24 h | 3 | A | 探索性 | 194.78 | 325.33 | 0.7583 |
| BiLSTM＋注意力 | X2 | HoT | 2025 | H | LUG / 24 h | 3 | A | 探索性 | 196.07 | 327.02 | 0.7558 |
| CNN＋BiLSTM＋LightGBM 堆叠 | X2 | HoT | 2025 | H | LUG / 24 h | 1 | A | 探索性 | 197.06 | 312.15 | 0.7775 |
| LightGBM LUG_VN | X2A2 | Kelmarsh | 2022 | K | LUG / 24 h | 3 | A | 探索性 | 144.69 | 221.67 | 0.8536 |
| LightGBM LUG_VR | X2A2 | Kelmarsh | 2022 | K | LUG / 24 h | 3 | A | 探索性 | 144.85 | 221.99 | 0.8532 |
| LightGBM LUG_VB | X2A2 | Kelmarsh | 2022 | K | LUG / 24 h | 3 | A | 探索性 | 144.26 | 220.02 | 0.8558 |
| LightGBM LUG_VN | X2A2 | Penmanshiel | 2021 | P | LUG / 24 h | 3 | A | 探索性 | 163.38 | 278.17 | 0.8142 |
| LightGBM LUG_VR | X2A2 | Penmanshiel | 2021 | P | LUG / 24 h | 3 | A | 探索性 | 160.79 | 271.43 | 0.8231 |
| LightGBM LUG_VB | X2A2 | Penmanshiel | 2021 | P | LUG / 24 h | 3 | A | 探索性 | 161.88 | 272.79 | 0.8214 |
| LightGBM LUG_VN | X2A2 | HoT | 2024 | H | LUG / 24 h | 3 | A | 探索性 | 194.50 | 329.29 | 0.7600 |
| LightGBM LUG_VR | X2A2 | HoT | 2024 | H | LUG / 24 h | 3 | A | 探索性 | 195.10 | 329.62 | 0.7595 |
| LightGBM LUG_VB | X2A2 | HoT | 2024 | H | LUG / 24 h | 3 | A | 探索性 | 194.00 | 329.19 | 0.7602 |
| LightGBM LUG_VN | X2A2 | HoT | 2025 | H | LUG / 24 h | 3 | A | 探索性 | 186.93 | 318.85 | 0.7679 |
| LightGBM LUG_VR | X2A2 | HoT | 2025 | H | LUG / 24 h | 3 | A | 探索性 | 187.11 | 318.92 | 0.7677 |
| LightGBM LUG_VB | X2A2 | HoT | 2025 | H | LUG / 24 h | 3 | A | 探索性 | 186.76 | 319.00 | 0.7676 |

## Chronos-2 长上下文、适配及集成

| 模型 | 来源 | 风场 | 评估 | 训练 | 输入 | 种子 | 等级 | 证据 | MAE | RMSE | R² |
|---|---|---|---|---|---|---:|---|---|---:|---:|---:|
| Chronos-2 零样本 | X3 | Kelmarsh | 2022 | 无本场拟合 | 含 GFS；1024 h（长上下文分量） | 0 | B | 探索性 | 155.65 | 待核 | 待核 |
| Chronos-2 微调 | X3 | Kelmarsh | 2022 | K 抽取2万 | 含 GFS；1024 h（长上下文分量） | 待核 | B | 探索性 | 147.06 | 待核 | 待核 |
| 集成 | X3 | Kelmarsh | 2022 | K | 含 GFS；1024 h（长上下文分量） | 待核 | B | 探索性 | 141.39 | 待核 | 待核 |
| 对照集成 | X3 | Kelmarsh | 2022 | K | LightGBM 对照集成，冻结树模型特征 | 待核 | B | 探索性 | 142.50 | 待核 | 待核 |
| Chronos-2 零样本 | X3 | Penmanshiel | 2021 | 无本场拟合 | 含 GFS；1024 h（长上下文分量） | 0 | B | 探索性 | 175.30 | 待核 | 待核 |
| Chronos-2 微调 | X3 | Penmanshiel | 2021 | P 抽取2万 | 含 GFS；1024 h（长上下文分量） | 待核 | B | 探索性 | 165.39 | 待核 | 待核 |
| 集成 | X3 | Penmanshiel | 2021 | P | 含 GFS；1024 h（长上下文分量） | 待核 | B | 探索性 | 157.26 | 待核 | 待核 |
| 对照集成 | X3 | Penmanshiel | 2021 | P | LightGBM 对照集成，冻结树模型特征 | 待核 | B | 探索性 | 159.59 | 待核 | 待核 |
| Chronos-2 零样本 | X3 | HoT | 2024 | 无本场拟合 | 含 GFS；1024 h（长上下文分量） | 0 | B | 探索性 | 214.99 | 待核 | 待核 |
| Chronos-2 微调 | X3 | HoT | 2024 | H 抽取2万 | 含 GFS；1024 h（长上下文分量） | 待核 | B | 探索性 | 206.25 | 待核 | 待核 |
| 集成 | X3 | HoT | 2024 | H | 含 GFS；1024 h（长上下文分量） | 待核 | B | 探索性 | 193.27 | 待核 | 待核 |
| 对照集成 | X3 | HoT | 2024 | H | LightGBM 对照集成，冻结树模型特征 | 待核 | B | 探索性 | 192.75 | 待核 | 待核 |
| Chronos-2 零样本 | X3 | HoT | 2025 | 无本场拟合 | 含 GFS；1024 h（长上下文分量） | 0 | B | 探索性 | 208.15 | 待核 | 待核 |
| Chronos-2 微调 | X3 | HoT | 2025 | H 抽取2万 | 含 GFS；1024 h（长上下文分量） | 待核 | B | 探索性 | 197.48 | 待核 | 待核 |
| 集成 | X3 | HoT | 2025 | H | 含 GFS；1024 h（长上下文分量） | 待核 | B | 探索性 | 185.90 | 待核 | 待核 |
| 对照集成 | X3 | HoT | 2025 | H | LightGBM 对照集成，冻结树模型特征 | 待核 | B | 探索性 | 185.88 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | Kelmarsh | V2 2021下半年 | 无本场拟合 | 含 GFS；168 h / single | 0 | B | 开发期 | 148.78 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | Kelmarsh | V2 2021下半年 | 无本场拟合 | 含 GFS；512 h / single | 0 | B | 开发期 | 143.19 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | Kelmarsh | V2 2021下半年 | 无本场拟合 | 含 GFS；1024 h / single | 0 | B | 开发期 | 141.07 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | Kelmarsh | V2 2021下半年 | 无本场拟合 | 含 GFS；168 h / group | 0 | B | 开发期 | 148.66 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | Kelmarsh | V2 2021下半年 | 无本场拟合 | 含 GFS；512 h / group | 0 | B | 开发期 | 143.17 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | Kelmarsh | V2 2021下半年 | 无本场拟合 | 含 GFS；1024 h / group | 0 | B | 开发期 | 141.08 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | Kelmarsh | V2 2021下半年 | 无本场拟合 | 含 GFS；LUG | 0 | B | 开发期 | 131.55 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | Penmanshiel | V2 2020下半年 | 无本场拟合 | 含 GFS；168 h / single | 0 | B | 开发期 | 203.89 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | Penmanshiel | V2 2020下半年 | 无本场拟合 | 含 GFS；512 h / single | 0 | B | 开发期 | 203.86 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | Penmanshiel | V2 2020下半年 | 无本场拟合 | 含 GFS；1024 h / single | 0 | B | 开发期 | 198.86 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | Penmanshiel | V2 2020下半年 | 无本场拟合 | 含 GFS；168 h / group | 0 | B | 开发期 | 203.56 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | Penmanshiel | V2 2020下半年 | 无本场拟合 | 含 GFS；512 h / group | 0 | B | 开发期 | 203.41 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | Penmanshiel | V2 2020下半年 | 无本场拟合 | 含 GFS；1024 h / group | 0 | B | 开发期 | 198.62 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | Penmanshiel | V2 2020下半年 | 无本场拟合 | 含 GFS；LUG | 0 | B | 开发期 | 175.62 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | HoT | V2 2023第4季度 | 无本场拟合 | 含 GFS；168 h / single | 0 | B | 开发期 | 217.52 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | HoT | V2 2023第4季度 | 无本场拟合 | 含 GFS；512 h / single | 0 | B | 开发期 | 211.80 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | HoT | V2 2023第4季度 | 无本场拟合 | 含 GFS；1024 h / single | 0 | B | 开发期 | 206.67 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | HoT | V2 2023第4季度 | 无本场拟合 | 含 GFS；168 h / group | 0 | B | 开发期 | 217.21 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | HoT | V2 2023第4季度 | 无本场拟合 | 含 GFS；512 h / group | 0 | B | 开发期 | 211.69 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | HoT | V2 2023第4季度 | 无本场拟合 | 含 GFS；1024 h / group | 0 | B | 开发期 | 206.63 | 待核 | 待核 |
| Chronos-2 零样本 | X3-dev | HoT | V2 2023第4季度 | 无本场拟合 | 含 GFS；LUG | 0 | B | 开发期 | 185.74 | 待核 | 待核 |
| Chronos-2 FT_B | X4 | Kelmarsh | 2022 | K | 含 GFS；Chronos 分量 1024 h；FT_B 用同场 2 万窗 | 待核 | B | 探索性 | 145.53 | 待核 | 待核 |
| 集成 | X4 | Kelmarsh | 2022 | K | 含 GFS；Chronos 分量 1024 h；FT_B 用同场 2 万窗 | 待核 | B | 探索性 | 140.34 | 待核 | 待核 |
| Chronos-2 FT_B | X4 | Penmanshiel | 2021 | P | 含 GFS；Chronos 分量 1024 h；FT_B 用同场 2 万窗 | 待核 | B | 探索性 | 166.04 | 待核 | 待核 |
| 集成 | X4 | Penmanshiel | 2021 | P | 含 GFS；Chronos 分量 1024 h；FT_B 用同场 2 万窗 | 待核 | B | 探索性 | 157.34 | 待核 | 待核 |
| Chronos-2 FT_B | X4 | HoT | 2024 | H | 含 GFS；Chronos 分量 1024 h；FT_B 用同场 2 万窗 | 待核 | B | 探索性 | 203.93 | 待核 | 待核 |
| 集成 | X4 | HoT | 2024 | H | 含 GFS；Chronos 分量 1024 h；FT_B 用同场 2 万窗 | 待核 | B | 探索性 | 192.92 | 待核 | 待核 |
| Chronos-2 FT_B | X4 | HoT | 2025 | H | 含 GFS；Chronos 分量 1024 h；FT_B 用同场 2 万窗 | 待核 | B | 探索性 | 196.17 | 待核 | 待核 |
| 集成 | X4 | HoT | 2025 | H | 含 GFS；Chronos 分量 1024 h；FT_B 用同场 2 万窗 | 待核 | B | 探索性 | 185.56 | 待核 | 待核 |

## 历史模型

| 模型 | 来源 | 风场 | 评估 | 训练 | 输入 | 种子 | 等级 | 证据 | MAE | RMSE | R² |
|---|---|---|---|---|---|---:|---|---|---:|---:|---:|
| TSMixer-upstream | E1 | Kelmarsh | 2024 | K 2016–22 | 本场＋上游；24 h | 3 | C | 探索性 | 166.28 | 255.29 | 0.8148 |
| iTransformer_last10 | E1 | Kelmarsh | 2024 | K 2016–22 | 本场；24 h | 3 | C | 探索性 | 191.88 | 296.27 | 0.7506 |
| iTransformer_upstream | E1 | Kelmarsh | 2024 | K 2016–22 | 本场＋上游；24 h | 3 | C | 探索性 | 184.62 | 283.98 | 0.7708 |
| TimeXer_last10 | E1 | Kelmarsh | 2024 | K 2016–22 | 本场；24 h | 3 | C | 探索性 | 196.82 | 292.49 | 0.7569 |
| TimeXer_upstream | E1 | Kelmarsh | 2024 | K 2016–22 | 本场＋上游；24 h | 3 | C | 探索性 | 191.09 | 283.79 | 0.7711 |
| TSMixer-upstream | E1 | Penmanshiel | 2022 | P 2016–20 | 本场＋上游；24 h | 3 | C | 探索性 | 213.63 | 329.32 | 0.7534 |
| iTransformer_last10 | E1 | Penmanshiel | 2022 | P 2016–20 | 本场；24 h | 3 | C | 探索性 | 239.17 | 371.35 | 0.6864 |
| iTransformer_upstream | E1 | Penmanshiel | 2022 | P 2016–20 | 本场＋上游；24 h | 3 | C | 探索性 | 236.79 | 371.58 | 0.6859 |
| TimeXer_last10 | E1 | Penmanshiel | 2022 | P 2016–20 | 本场；24 h | 3 | C | 探索性 | 245.05 | 365.83 | 0.6957 |
| TimeXer_upstream | E1 | Penmanshiel | 2022 | P 2016–20 | 本场＋上游；24 h | 3 | C | 探索性 | 247.15 | 369.12 | 0.6902 |
| TSMixer-upstream | E1 | Penmanshiel | 2023 | P 2016–21 | 本场＋上游；24 h | 3 | C | 探索性 | 188.84 | 301.96 | 0.7725 |
| iTransformer_last10 | E1 | Penmanshiel | 2023 | P 2016–21 | 本场；24 h | 3 | C | 探索性 | 210.94 | 340.78 | 0.7103 |
| iTransformer_upstream | E1 | Penmanshiel | 2023 | P 2016–21 | 本场＋上游；24 h | 3 | C | 探索性 | 213.68 | 342.20 | 0.7078 |
| TimeXer_last10 | E1 | Penmanshiel | 2023 | P 2016–21 | 本场；24 h | 3 | C | 探索性 | 215.97 | 333.41 | 0.7227 |
| TimeXer_upstream | E1 | Penmanshiel | 2023 | P 2016–21 | 本场＋上游；24 h | 3 | C | 探索性 | 214.47 | 326.09 | 0.7347 |

## X5、X5b 开发段供应条件

| 模型 | 来源 | 风场 | 评估 | 训练 | 输入 | 种子 | 等级 | 证据 | MAE | RMSE | R² |
|---|---|---|---|---|---|---:|---|---|---:|---:|---:|
| LightGBM LU | X5 | Kelmarsh | V2 2021下半年 | K | LU / 24 h / d=6 h | 1 | A | 开发期 | 145.31 | 待核 | 待核 |
| LightGBM LG | X5 | Kelmarsh | V2 2021下半年 | K | LG / 24 h / d=6 h | 1 | A | 开发期 | 134.38 | 待核 | 待核 |
| LightGBM LUG | X5 | Kelmarsh | V2 2021下半年 | K | LUG / 24 h / d=6 h | 1 | A | 开发期 | 132.10 | 待核 | 待核 |
| LightGBM LG_mix@6 | X5 | Kelmarsh | V2 2021下半年 | K | LG / 24 h / d=6 h | 1 | A | 开发期 | 135.78 | 待核 | 待核 |
| LightGBM LUG_mix@6 | X5 | Kelmarsh | V2 2021下半年 | K | LUG / 24 h / d=6 h | 1 | A | 开发期 | 133.04 | 待核 | 待核 |
| LightGBM LG | X5 | Kelmarsh | V2 2021下半年 | K | LG / 24 h / d=9 h | 1 | A | 开发期 | 134.80 | 待核 | 待核 |
| LightGBM LUG | X5 | Kelmarsh | V2 2021下半年 | K | LUG / 24 h / d=9 h | 1 | A | 开发期 | 132.53 | 待核 | 待核 |
| LightGBM LG_d6@9 | X5 | Kelmarsh | V2 2021下半年 | K | LG / 24 h / d=9 h | 1 | A | 开发期 | 136.29 | 待核 | 待核 |
| LightGBM LUG_d6@9 | X5 | Kelmarsh | V2 2021下半年 | K | LUG / 24 h / d=9 h | 1 | A | 开发期 | 133.75 | 待核 | 待核 |
| LightGBM LG_mix@9 | X5 | Kelmarsh | V2 2021下半年 | K | LG / 24 h / d=9 h | 1 | A | 开发期 | 135.94 | 待核 | 待核 |
| LightGBM LUG_mix@9 | X5 | Kelmarsh | V2 2021下半年 | K | LUG / 24 h / d=9 h | 1 | A | 开发期 | 133.46 | 待核 | 待核 |
| LightGBM LG | X5 | Kelmarsh | V2 2021下半年 | K | LG / 24 h / d=12 h | 1 | A | 开发期 | 137.59 | 待核 | 待核 |
| LightGBM LUG | X5 | Kelmarsh | V2 2021下半年 | K | LUG / 24 h / d=12 h | 1 | A | 开发期 | 134.57 | 待核 | 待核 |
| LightGBM LG_d6@12 | X5 | Kelmarsh | V2 2021下半年 | K | LG / 24 h / d=12 h | 1 | A | 开发期 | 157.22 | 待核 | 待核 |
| LightGBM LUG_d6@12 | X5 | Kelmarsh | V2 2021下半年 | K | LUG / 24 h / d=12 h | 1 | A | 开发期 | 154.80 | 待核 | 待核 |
| LightGBM LG_mix@12 | X5 | Kelmarsh | V2 2021下半年 | K | LG / 24 h / d=12 h | 1 | A | 开发期 | 138.78 | 待核 | 待核 |
| LightGBM LUG_mix@12 | X5 | Kelmarsh | V2 2021下半年 | K | LUG / 24 h / d=12 h | 1 | A | 开发期 | 135.37 | 待核 | 待核 |
| LightGBM LG | X5 | Kelmarsh | V2 2021下半年 | K | LG / 24 h / d=15 h | 1 | A | 开发期 | 148.69 | 待核 | 待核 |
| LightGBM LUG | X5 | Kelmarsh | V2 2021下半年 | K | LUG / 24 h / d=15 h | 1 | A | 开发期 | 141.94 | 待核 | 待核 |
| LightGBM LG_d6@15 | X5 | Kelmarsh | V2 2021下半年 | K | LG / 24 h / d=15 h | 1 | A | 开发期 | 206.75 | 待核 | 待核 |
| LightGBM LUG_d6@15 | X5 | Kelmarsh | V2 2021下半年 | K | LUG / 24 h / d=15 h | 1 | A | 开发期 | 198.99 | 待核 | 待核 |
| LightGBM LG_mix@15 | X5 | Kelmarsh | V2 2021下半年 | K | LG / 24 h / d=15 h | 1 | A | 开发期 | 148.51 | 待核 | 待核 |
| LightGBM LUG_mix@15 | X5 | Kelmarsh | V2 2021下半年 | K | LUG / 24 h / d=15 h | 1 | A | 开发期 | 141.68 | 待核 | 待核 |
| LightGBM LG | X5 | Kelmarsh | V2 2021下半年 | K | LG / 24 h / d=18 h | 1 | A | 开发期 | 161.14 | 待核 | 待核 |
| LightGBM LUG | X5 | Kelmarsh | V2 2021下半年 | K | LUG / 24 h / d=18 h | 1 | A | 开发期 | 145.65 | 待核 | 待核 |
| LightGBM LG_d6@18 | X5 | Kelmarsh | V2 2021下半年 | K | LG / 24 h / d=18 h | 1 | A | 开发期 | 235.56 | 待核 | 待核 |
| LightGBM LUG_d6@18 | X5 | Kelmarsh | V2 2021下半年 | K | LUG / 24 h / d=18 h | 1 | A | 开发期 | 219.39 | 待核 | 待核 |
| LightGBM LG_mix@18 | X5 | Kelmarsh | V2 2021下半年 | K | LG / 24 h / d=18 h | 1 | A | 开发期 | 161.71 | 待核 | 待核 |
| LightGBM LUG_mix@18 | X5 | Kelmarsh | V2 2021下半年 | K | LUG / 24 h / d=18 h | 1 | A | 开发期 | 147.08 | 待核 | 待核 |
| LightGBM LU | X5 | Penmanshiel | V2 2020下半年 | P | LU / 24 h / d=6 h | 1 | A | 开发期 | 205.15 | 待核 | 待核 |
| LightGBM LG | X5 | Penmanshiel | V2 2020下半年 | P | LG / 24 h / d=6 h | 1 | A | 开发期 | 178.63 | 待核 | 待核 |
| LightGBM LUG | X5 | Penmanshiel | V2 2020下半年 | P | LUG / 24 h / d=6 h | 1 | A | 开发期 | 176.46 | 待核 | 待核 |
| LightGBM LG_mix@6 | X5 | Penmanshiel | V2 2020下半年 | P | LG / 24 h / d=6 h | 1 | A | 开发期 | 180.79 | 待核 | 待核 |
| LightGBM LUG_mix@6 | X5 | Penmanshiel | V2 2020下半年 | P | LUG / 24 h / d=6 h | 1 | A | 开发期 | 179.18 | 待核 | 待核 |
| LightGBM LG | X5 | Penmanshiel | V2 2020下半年 | P | LG / 24 h / d=9 h | 1 | A | 开发期 | 181.63 | 待核 | 待核 |
| LightGBM LUG | X5 | Penmanshiel | V2 2020下半年 | P | LUG / 24 h / d=9 h | 1 | A | 开发期 | 178.97 | 待核 | 待核 |
| LightGBM LG_d6@9 | X5 | Penmanshiel | V2 2020下半年 | P | LG / 24 h / d=9 h | 1 | A | 开发期 | 180.91 | 待核 | 待核 |
| LightGBM LUG_d6@9 | X5 | Penmanshiel | V2 2020下半年 | P | LUG / 24 h / d=9 h | 1 | A | 开发期 | 178.87 | 待核 | 待核 |
| LightGBM LG_mix@9 | X5 | Penmanshiel | V2 2020下半年 | P | LG / 24 h / d=9 h | 1 | A | 开发期 | 182.59 | 待核 | 待核 |
| LightGBM LUG_mix@9 | X5 | Penmanshiel | V2 2020下半年 | P | LUG / 24 h / d=9 h | 1 | A | 开发期 | 180.64 | 待核 | 待核 |
| LightGBM LG | X5 | Penmanshiel | V2 2020下半年 | P | LG / 24 h / d=12 h | 1 | A | 开发期 | 186.44 | 待核 | 待核 |
| LightGBM LUG | X5 | Penmanshiel | V2 2020下半年 | P | LUG / 24 h / d=12 h | 1 | A | 开发期 | 182.81 | 待核 | 待核 |
| LightGBM LG_d6@12 | X5 | Penmanshiel | V2 2020下半年 | P | LG / 24 h / d=12 h | 1 | A | 开发期 | 198.15 | 待核 | 待核 |
| LightGBM LUG_d6@12 | X5 | Penmanshiel | V2 2020下半年 | P | LUG / 24 h / d=12 h | 1 | A | 开发期 | 195.08 | 待核 | 待核 |
| LightGBM LG_mix@12 | X5 | Penmanshiel | V2 2020下半年 | P | LG / 24 h / d=12 h | 1 | A | 开发期 | 187.45 | 待核 | 待核 |
| LightGBM LUG_mix@12 | X5 | Penmanshiel | V2 2020下半年 | P | LUG / 24 h / d=12 h | 1 | A | 开发期 | 184.03 | 待核 | 待核 |
| LightGBM LG | X5 | Penmanshiel | V2 2020下半年 | P | LG / 24 h / d=15 h | 1 | A | 开发期 | 206.29 | 待核 | 待核 |
| LightGBM LUG | X5 | Penmanshiel | V2 2020下半年 | P | LUG / 24 h / d=15 h | 1 | A | 开发期 | 195.38 | 待核 | 待核 |
| LightGBM LG_d6@15 | X5 | Penmanshiel | V2 2020下半年 | P | LG / 24 h / d=15 h | 1 | A | 开发期 | 258.78 | 待核 | 待核 |
| LightGBM LUG_d6@15 | X5 | Penmanshiel | V2 2020下半年 | P | LUG / 24 h / d=15 h | 1 | A | 开发期 | 248.70 | 待核 | 待核 |
| LightGBM LG_mix@15 | X5 | Penmanshiel | V2 2020下半年 | P | LG / 24 h / d=15 h | 1 | A | 开发期 | 205.29 | 待核 | 待核 |
| LightGBM LUG_mix@15 | X5 | Penmanshiel | V2 2020下半年 | P | LUG / 24 h / d=15 h | 1 | A | 开发期 | 196.50 | 待核 | 待核 |
| LightGBM LG | X5 | Penmanshiel | V2 2020下半年 | P | LG / 24 h / d=18 h | 1 | A | 开发期 | 219.47 | 待核 | 待核 |
| LightGBM LUG | X5 | Penmanshiel | V2 2020下半年 | P | LUG / 24 h / d=18 h | 1 | A | 开发期 | 203.48 | 待核 | 待核 |
| LightGBM LG_d6@18 | X5 | Penmanshiel | V2 2020下半年 | P | LG / 24 h / d=18 h | 1 | A | 开发期 | 293.20 | 待核 | 待核 |
| LightGBM LUG_d6@18 | X5 | Penmanshiel | V2 2020下半年 | P | LUG / 24 h / d=18 h | 1 | A | 开发期 | 276.81 | 待核 | 待核 |
| LightGBM LG_mix@18 | X5 | Penmanshiel | V2 2020下半年 | P | LG / 24 h / d=18 h | 1 | A | 开发期 | 218.83 | 待核 | 待核 |
| LightGBM LUG_mix@18 | X5 | Penmanshiel | V2 2020下半年 | P | LUG / 24 h / d=18 h | 1 | A | 开发期 | 204.52 | 待核 | 待核 |
| LightGBM LU | X5 | HoT | V2 2023第4季度 | H | LU / 24 h / d=6 h | 1 | A | 开发期 | 217.16 | 待核 | 待核 |
| LightGBM LG | X5 | HoT | V2 2023第4季度 | H | LG / 24 h / d=6 h | 1 | A | 开发期 | 188.00 | 待核 | 待核 |
| LightGBM LUG | X5 | HoT | V2 2023第4季度 | H | LUG / 24 h / d=6 h | 1 | A | 开发期 | 186.18 | 待核 | 待核 |
| LightGBM LG_mix@6 | X5 | HoT | V2 2023第4季度 | H | LG / 24 h / d=6 h | 1 | A | 开发期 | 188.94 | 待核 | 待核 |
| LightGBM LUG_mix@6 | X5 | HoT | V2 2023第4季度 | H | LUG / 24 h / d=6 h | 1 | A | 开发期 | 185.94 | 待核 | 待核 |
| LightGBM LG | X5 | HoT | V2 2023第4季度 | H | LG / 24 h / d=9 h | 1 | A | 开发期 | 189.30 | 待核 | 待核 |
| LightGBM LUG | X5 | HoT | V2 2023第4季度 | H | LUG / 24 h / d=9 h | 1 | A | 开发期 | 186.72 | 待核 | 待核 |
| LightGBM LG_d6@9 | X5 | HoT | V2 2023第4季度 | H | LG / 24 h / d=9 h | 1 | A | 开发期 | 190.30 | 待核 | 待核 |
| LightGBM LUG_d6@9 | X5 | HoT | V2 2023第4季度 | H | LUG / 24 h / d=9 h | 1 | A | 开发期 | 189.12 | 待核 | 待核 |
| LightGBM LG_mix@9 | X5 | HoT | V2 2023第4季度 | H | LG / 24 h / d=9 h | 1 | A | 开发期 | 189.34 | 待核 | 待核 |
| LightGBM LUG_mix@9 | X5 | HoT | V2 2023第4季度 | H | LUG / 24 h / d=9 h | 1 | A | 开发期 | 186.59 | 待核 | 待核 |
| LightGBM LG | X5 | HoT | V2 2023第4季度 | H | LG / 24 h / d=12 h | 1 | A | 开发期 | 195.45 | 待核 | 待核 |
| LightGBM LUG | X5 | HoT | V2 2023第4季度 | H | LUG / 24 h / d=12 h | 1 | A | 开发期 | 191.96 | 待核 | 待核 |
| LightGBM LG_d6@12 | X5 | HoT | V2 2023第4季度 | H | LG / 24 h / d=12 h | 1 | A | 开发期 | 236.53 | 待核 | 待核 |
| LightGBM LUG_d6@12 | X5 | HoT | V2 2023第4季度 | H | LUG / 24 h / d=12 h | 1 | A | 开发期 | 231.26 | 待核 | 待核 |
| LightGBM LG_mix@12 | X5 | HoT | V2 2023第4季度 | H | LG / 24 h / d=12 h | 1 | A | 开发期 | 197.47 | 待核 | 待核 |
| LightGBM LUG_mix@12 | X5 | HoT | V2 2023第4季度 | H | LUG / 24 h / d=12 h | 1 | A | 开发期 | 193.52 | 待核 | 待核 |
| LightGBM LG | X5 | HoT | V2 2023第4季度 | H | LG / 24 h / d=15 h | 1 | A | 开发期 | 221.04 | 待核 | 待核 |
| LightGBM LUG | X5 | HoT | V2 2023第4季度 | H | LUG / 24 h / d=15 h | 1 | A | 开发期 | 207.58 | 待核 | 待核 |
| LightGBM LG_d6@15 | X5 | HoT | V2 2023第4季度 | H | LG / 24 h / d=15 h | 1 | A | 开发期 | 310.65 | 待核 | 待核 |
| LightGBM LUG_d6@15 | X5 | HoT | V2 2023第4季度 | H | LUG / 24 h / d=15 h | 1 | A | 开发期 | 287.71 | 待核 | 待核 |
| LightGBM LG_mix@15 | X5 | HoT | V2 2023第4季度 | H | LG / 24 h / d=15 h | 1 | A | 开发期 | 222.75 | 待核 | 待核 |
| LightGBM LUG_mix@15 | X5 | HoT | V2 2023第4季度 | H | LUG / 24 h / d=15 h | 1 | A | 开发期 | 208.13 | 待核 | 待核 |
| LightGBM LG | X5 | HoT | V2 2023第4季度 | H | LG / 24 h / d=18 h | 1 | A | 开发期 | 240.59 | 待核 | 待核 |
| LightGBM LUG | X5 | HoT | V2 2023第4季度 | H | LUG / 24 h / d=18 h | 1 | A | 开发期 | 216.21 | 待核 | 待核 |
| LightGBM LG_d6@18 | X5 | HoT | V2 2023第4季度 | H | LG / 24 h / d=18 h | 1 | A | 开发期 | 305.31 | 待核 | 待核 |
| LightGBM LUG_d6@18 | X5 | HoT | V2 2023第4季度 | H | LUG / 24 h / d=18 h | 1 | A | 开发期 | 280.19 | 待核 | 待核 |
| LightGBM LG_mix@18 | X5 | HoT | V2 2023第4季度 | H | LG / 24 h / d=18 h | 1 | A | 开发期 | 242.18 | 待核 | 待核 |
| LightGBM LUG_mix@18 | X5 | HoT | V2 2023第4季度 | H | LUG / 24 h / d=18 h | 1 | A | 开发期 | 216.06 | 待核 | 待核 |
| LightGBM LG | X5b | Kelmarsh | V2 2021下半年 | K | LG / 24 h / 状态 A | 1 | A | 开发期 | 134.38 | 待核 | 待核 |
| LightGBM LUG | X5b | Kelmarsh | V2 2021下半年 | K | LUG / 24 h / 状态 A | 1 | A | 开发期 | 132.10 | 待核 | 待核 |
| LightGBM LG | X5b | Kelmarsh | V2 2021下半年 | K | LG / 24 h / 状态 C | 1 | A | 开发期 | 148.69 | 待核 | 待核 |
| LightGBM LUG | X5b | Kelmarsh | V2 2021下半年 | K | LUG / 24 h / 状态 C | 1 | A | 开发期 | 141.94 | 待核 | 待核 |
| LightGBM LG | X5b | Kelmarsh | V2 2021下半年 | K | LG / 24 h / 状态 Cc | 1 | A | 开发期 | 148.40 | 待核 | 待核 |
| LightGBM LUG | X5b | Kelmarsh | V2 2021下半年 | K | LUG / 24 h / 状态 Cc | 1 | A | 开发期 | 141.97 | 待核 | 待核 |
| LightGBM LG | X5b | Kelmarsh | V2 2021下半年 | K | LG / 24 h / 状态 D | 1 | A | 开发期 | 147.57 | 待核 | 待核 |
| LightGBM LUG | X5b | Kelmarsh | V2 2021下半年 | K | LUG / 24 h / 状态 D | 1 | A | 开发期 | 141.54 | 待核 | 待核 |
| LightGBM LG | X5b | Penmanshiel | V2 2020下半年 | P | LG / 24 h / 状态 A | 1 | A | 开发期 | 178.63 | 待核 | 待核 |
| LightGBM LUG | X5b | Penmanshiel | V2 2020下半年 | P | LUG / 24 h / 状态 A | 1 | A | 开发期 | 176.46 | 待核 | 待核 |
| LightGBM LG | X5b | Penmanshiel | V2 2020下半年 | P | LG / 24 h / 状态 C | 1 | A | 开发期 | 206.29 | 待核 | 待核 |
| LightGBM LUG | X5b | Penmanshiel | V2 2020下半年 | P | LUG / 24 h / 状态 C | 1 | A | 开发期 | 195.38 | 待核 | 待核 |
| LightGBM LG | X5b | Penmanshiel | V2 2020下半年 | P | LG / 24 h / 状态 Cc | 1 | A | 开发期 | 206.15 | 待核 | 待核 |
| LightGBM LUG | X5b | Penmanshiel | V2 2020下半年 | P | LUG / 24 h / 状态 Cc | 1 | A | 开发期 | 196.76 | 待核 | 待核 |
| LightGBM LG | X5b | Penmanshiel | V2 2020下半年 | P | LG / 24 h / 状态 D | 1 | A | 开发期 | 201.01 | 待核 | 待核 |
| LightGBM LUG | X5b | Penmanshiel | V2 2020下半年 | P | LUG / 24 h / 状态 D | 1 | A | 开发期 | 193.08 | 待核 | 待核 |
| LightGBM LG | X5b | HoT | V2 2023第4季度 | H | LG / 24 h / 状态 A | 1 | A | 开发期 | 188.00 | 待核 | 待核 |
| LightGBM LUG | X5b | HoT | V2 2023第4季度 | H | LUG / 24 h / 状态 A | 1 | A | 开发期 | 186.18 | 待核 | 待核 |
| LightGBM LG | X5b | HoT | V2 2023第4季度 | H | LG / 24 h / 状态 C | 1 | A | 开发期 | 221.04 | 待核 | 待核 |
| LightGBM LUG | X5b | HoT | V2 2023第4季度 | H | LUG / 24 h / 状态 C | 1 | A | 开发期 | 207.58 | 待核 | 待核 |
| LightGBM LG | X5b | HoT | V2 2023第4季度 | H | LG / 24 h / 状态 Cc | 1 | A | 开发期 | 221.53 | 待核 | 待核 |
| LightGBM LUG | X5b | HoT | V2 2023第4季度 | H | LUG / 24 h / 状态 Cc | 1 | A | 开发期 | 207.27 | 待核 | 待核 |
| LightGBM LG | X5b | HoT | V2 2023第4季度 | H | LG / 24 h / 状态 D | 1 | A | 开发期 | 221.84 | 待核 | 待核 |
| LightGBM LUG | X5b | HoT | V2 2023第4季度 | H | LUG / 24 h / 状态 D | 1 | A | 开发期 | 207.22 | 待核 | 待核 |

## X5、X5b 已打开测试年供应条件

| 模型 | 来源 | 风场 | 评估 | 训练 | 输入 | 种子 | 等级 | 证据 | MAE | RMSE | R² |
|---|---|---|---|---|---|---:|---|---|---:|---:|---:|
| LightGBM LU | X5 | Kelmarsh | 2022 | K | LU / 24 h / d=6 h | 1 | A | 探索性 | 160.63 | 待核 | 待核 |
| LightGBM LG | X5 | Kelmarsh | 2022 | K | LG / 24 h / d=6 h | 1 | A | 探索性 | 147.86 | 待核 | 待核 |
| LightGBM LUG | X5 | Kelmarsh | 2022 | K | LUG / 24 h / d=6 h | 1 | A | 探索性 | 145.33 | 待核 | 待核 |
| LightGBM LG_mix@6 | X5 | Kelmarsh | 2022 | K | LG / 24 h / d=6 h | 1 | A | 探索性 | 150.01 | 待核 | 待核 |
| LightGBM LUG_mix@6 | X5 | Kelmarsh | 2022 | K | LUG / 24 h / d=6 h | 1 | A | 探索性 | 145.64 | 待核 | 待核 |
| LightGBM LG | X5 | Kelmarsh | 2022 | K | LG / 24 h / d=9 h | 1 | A | 探索性 | 148.94 | 待核 | 待核 |
| LightGBM LUG | X5 | Kelmarsh | 2022 | K | LUG / 24 h / d=9 h | 1 | A | 探索性 | 145.02 | 待核 | 待核 |
| LightGBM LG_d6@9 | X5 | Kelmarsh | 2022 | K | LG / 24 h / d=9 h | 1 | A | 探索性 | 149.33 | 待核 | 待核 |
| LightGBM LUG_d6@9 | X5 | Kelmarsh | 2022 | K | LUG / 24 h / d=9 h | 1 | A | 探索性 | 146.62 | 待核 | 待核 |
| LightGBM LG_mix@9 | X5 | Kelmarsh | 2022 | K | LG / 24 h / d=9 h | 1 | A | 探索性 | 150.25 | 待核 | 待核 |
| LightGBM LUG_mix@9 | X5 | Kelmarsh | 2022 | K | LUG / 24 h / d=9 h | 1 | A | 探索性 | 145.85 | 待核 | 待核 |
| LightGBM LG | X5 | Kelmarsh | 2022 | K | LG / 24 h / d=12 h | 1 | A | 探索性 | 151.17 | 待核 | 待核 |
| LightGBM LUG | X5 | Kelmarsh | 2022 | K | LUG / 24 h / d=12 h | 1 | A | 探索性 | 147.42 | 待核 | 待核 |
| LightGBM LG_d6@12 | X5 | Kelmarsh | 2022 | K | LG / 24 h / d=12 h | 1 | A | 探索性 | 168.75 | 待核 | 待核 |
| LightGBM LUG_d6@12 | X5 | Kelmarsh | 2022 | K | LUG / 24 h / d=12 h | 1 | A | 探索性 | 165.62 | 待核 | 待核 |
| LightGBM LG_mix@12 | X5 | Kelmarsh | 2022 | K | LG / 24 h / d=12 h | 1 | A | 探索性 | 152.24 | 待核 | 待核 |
| LightGBM LUG_mix@12 | X5 | Kelmarsh | 2022 | K | LUG / 24 h / d=12 h | 1 | A | 探索性 | 147.24 | 待核 | 待核 |
| LightGBM LG | X5 | Kelmarsh | 2022 | K | LG / 24 h / d=15 h | 1 | A | 探索性 | 161.90 | 待核 | 待核 |
| LightGBM LUG | X5 | Kelmarsh | 2022 | K | LUG / 24 h / d=15 h | 1 | A | 探索性 | 154.85 | 待核 | 待核 |
| LightGBM LG_d6@15 | X5 | Kelmarsh | 2022 | K | LG / 24 h / d=15 h | 1 | A | 探索性 | 219.89 | 待核 | 待核 |
| LightGBM LUG_d6@15 | X5 | Kelmarsh | 2022 | K | LUG / 24 h / d=15 h | 1 | A | 探索性 | 210.79 | 待核 | 待核 |
| LightGBM LG_mix@15 | X5 | Kelmarsh | 2022 | K | LG / 24 h / d=15 h | 1 | A | 探索性 | 163.46 | 待核 | 待核 |
| LightGBM LUG_mix@15 | X5 | Kelmarsh | 2022 | K | LUG / 24 h / d=15 h | 1 | A | 探索性 | 154.27 | 待核 | 待核 |
| LightGBM LG | X5 | Kelmarsh | 2022 | K | LG / 24 h / d=18 h | 1 | A | 探索性 | 175.98 | 待核 | 待核 |
| LightGBM LUG | X5 | Kelmarsh | 2022 | K | LUG / 24 h / d=18 h | 1 | A | 探索性 | 159.62 | 待核 | 待核 |
| LightGBM LG_d6@18 | X5 | Kelmarsh | 2022 | K | LG / 24 h / d=18 h | 1 | A | 探索性 | 251.44 | 待核 | 待核 |
| LightGBM LUG_d6@18 | X5 | Kelmarsh | 2022 | K | LUG / 24 h / d=18 h | 1 | A | 探索性 | 233.58 | 待核 | 待核 |
| LightGBM LG_mix@18 | X5 | Kelmarsh | 2022 | K | LG / 24 h / d=18 h | 1 | A | 探索性 | 177.65 | 待核 | 待核 |
| LightGBM LUG_mix@18 | X5 | Kelmarsh | 2022 | K | LUG / 24 h / d=18 h | 1 | A | 探索性 | 161.24 | 待核 | 待核 |
| LightGBM LU | X5 | Penmanshiel | 2021 | P | LU / 24 h / d=6 h | 1 | A | 探索性 | 176.47 | 待核 | 待核 |
| LightGBM LG | X5 | Penmanshiel | 2021 | P | LG / 24 h / d=6 h | 1 | A | 探索性 | 163.72 | 待核 | 待核 |
| LightGBM LUG | X5 | Penmanshiel | 2021 | P | LUG / 24 h / d=6 h | 1 | A | 探索性 | 161.02 | 待核 | 待核 |
| LightGBM LG_mix@6 | X5 | Penmanshiel | 2021 | P | LG / 24 h / d=6 h | 1 | A | 探索性 | 162.83 | 待核 | 待核 |
| LightGBM LUG_mix@6 | X5 | Penmanshiel | 2021 | P | LUG / 24 h / d=6 h | 1 | A | 探索性 | 160.67 | 待核 | 待核 |
| LightGBM LG | X5 | Penmanshiel | 2021 | P | LG / 24 h / d=9 h | 1 | A | 探索性 | 164.16 | 待核 | 待核 |
| LightGBM LUG | X5 | Penmanshiel | 2021 | P | LUG / 24 h / d=9 h | 1 | A | 探索性 | 160.96 | 待核 | 待核 |
| LightGBM LG_d6@9 | X5 | Penmanshiel | 2021 | P | LG / 24 h / d=9 h | 1 | A | 探索性 | 164.91 | 待核 | 待核 |
| LightGBM LUG_d6@9 | X5 | Penmanshiel | 2021 | P | LUG / 24 h / d=9 h | 1 | A | 探索性 | 162.01 | 待核 | 待核 |
| LightGBM LG_mix@9 | X5 | Penmanshiel | 2021 | P | LG / 24 h / d=9 h | 1 | A | 探索性 | 163.81 | 待核 | 待核 |
| LightGBM LUG_mix@9 | X5 | Penmanshiel | 2021 | P | LUG / 24 h / d=9 h | 1 | A | 探索性 | 161.71 | 待核 | 待核 |
| LightGBM LG | X5 | Penmanshiel | 2021 | P | LG / 24 h / d=12 h | 1 | A | 探索性 | 166.07 | 待核 | 待核 |
| LightGBM LUG | X5 | Penmanshiel | 2021 | P | LUG / 24 h / d=12 h | 1 | A | 探索性 | 163.39 | 待核 | 待核 |
| LightGBM LG_d6@12 | X5 | Penmanshiel | 2021 | P | LG / 24 h / d=12 h | 1 | A | 探索性 | 183.70 | 待核 | 待核 |
| LightGBM LUG_d6@12 | X5 | Penmanshiel | 2021 | P | LUG / 24 h / d=12 h | 1 | A | 探索性 | 180.58 | 待核 | 待核 |
| LightGBM LG_mix@12 | X5 | Penmanshiel | 2021 | P | LG / 24 h / d=12 h | 1 | A | 探索性 | 165.84 | 待核 | 待核 |
| LightGBM LUG_mix@12 | X5 | Penmanshiel | 2021 | P | LUG / 24 h / d=12 h | 1 | A | 探索性 | 163.54 | 待核 | 待核 |
| LightGBM LG | X5 | Penmanshiel | 2021 | P | LG / 24 h / d=15 h | 1 | A | 探索性 | 176.20 | 待核 | 待核 |
| LightGBM LUG | X5 | Penmanshiel | 2021 | P | LUG / 24 h / d=15 h | 1 | A | 探索性 | 169.98 | 待核 | 待核 |
| LightGBM LG_d6@15 | X5 | Penmanshiel | 2021 | P | LG / 24 h / d=15 h | 1 | A | 探索性 | 242.56 | 待核 | 待核 |
| LightGBM LUG_d6@15 | X5 | Penmanshiel | 2021 | P | LUG / 24 h / d=15 h | 1 | A | 探索性 | 232.23 | 待核 | 待核 |
| LightGBM LG_mix@15 | X5 | Penmanshiel | 2021 | P | LG / 24 h / d=15 h | 1 | A | 探索性 | 174.55 | 待核 | 待核 |
| LightGBM LUG_mix@15 | X5 | Penmanshiel | 2021 | P | LUG / 24 h / d=15 h | 1 | A | 探索性 | 169.08 | 待核 | 待核 |
| LightGBM LG | X5 | Penmanshiel | 2021 | P | LG / 24 h / d=18 h | 1 | A | 探索性 | 187.39 | 待核 | 待核 |
| LightGBM LUG | X5 | Penmanshiel | 2021 | P | LUG / 24 h / d=18 h | 1 | A | 探索性 | 176.04 | 待核 | 待核 |
| LightGBM LG_d6@18 | X5 | Penmanshiel | 2021 | P | LG / 24 h / d=18 h | 1 | A | 探索性 | 280.54 | 待核 | 待核 |
| LightGBM LUG_d6@18 | X5 | Penmanshiel | 2021 | P | LUG / 24 h / d=18 h | 1 | A | 探索性 | 262.49 | 待核 | 待核 |
| LightGBM LG_mix@18 | X5 | Penmanshiel | 2021 | P | LG / 24 h / d=18 h | 1 | A | 探索性 | 190.35 | 待核 | 待核 |
| LightGBM LUG_mix@18 | X5 | Penmanshiel | 2021 | P | LUG / 24 h / d=18 h | 1 | A | 探索性 | 179.16 | 待核 | 待核 |
| LightGBM LU | X5 | HoT | 2024 | H | LU / 24 h / d=6 h | 1 | A | 探索性 | 212.17 | 待核 | 待核 |
| LightGBM LG | X5 | HoT | 2024 | H | LG / 24 h / d=6 h | 1 | A | 探索性 | 197.58 | 待核 | 待核 |
| LightGBM LUG | X5 | HoT | 2024 | H | LUG / 24 h / d=6 h | 1 | A | 探索性 | 194.02 | 待核 | 待核 |
| LightGBM LG_mix@6 | X5 | HoT | 2024 | H | LG / 24 h / d=6 h | 1 | A | 探索性 | 198.16 | 待核 | 待核 |
| LightGBM LUG_mix@6 | X5 | HoT | 2024 | H | LUG / 24 h / d=6 h | 1 | A | 探索性 | 194.99 | 待核 | 待核 |
| LightGBM LG | X5 | HoT | 2024 | H | LG / 24 h / d=9 h | 1 | A | 探索性 | 198.78 | 待核 | 待核 |
| LightGBM LUG | X5 | HoT | 2024 | H | LUG / 24 h / d=9 h | 1 | A | 探索性 | 194.73 | 待核 | 待核 |
| LightGBM LG_d6@9 | X5 | HoT | 2024 | H | LG / 24 h / d=9 h | 1 | A | 探索性 | 199.72 | 待核 | 待核 |
| LightGBM LUG_d6@9 | X5 | HoT | 2024 | H | LUG / 24 h / d=9 h | 1 | A | 探索性 | 195.73 | 待核 | 待核 |
| LightGBM LG_mix@9 | X5 | HoT | 2024 | H | LG / 24 h / d=9 h | 1 | A | 探索性 | 199.22 | 待核 | 待核 |
| LightGBM LUG_mix@9 | X5 | HoT | 2024 | H | LUG / 24 h / d=9 h | 1 | A | 探索性 | 195.58 | 待核 | 待核 |
| LightGBM LG | X5 | HoT | 2024 | H | LG / 24 h / d=12 h | 1 | A | 探索性 | 204.25 | 待核 | 待核 |
| LightGBM LUG | X5 | HoT | 2024 | H | LUG / 24 h / d=12 h | 1 | A | 探索性 | 198.88 | 待核 | 待核 |
| LightGBM LG_d6@12 | X5 | HoT | 2024 | H | LG / 24 h / d=12 h | 1 | A | 探索性 | 224.30 | 待核 | 待核 |
| LightGBM LUG_d6@12 | X5 | HoT | 2024 | H | LUG / 24 h / d=12 h | 1 | A | 探索性 | 216.98 | 待核 | 待核 |
| LightGBM LG_mix@12 | X5 | HoT | 2024 | H | LG / 24 h / d=12 h | 1 | A | 探索性 | 205.48 | 待核 | 待核 |
| LightGBM LUG_mix@12 | X5 | HoT | 2024 | H | LUG / 24 h / d=12 h | 1 | A | 探索性 | 199.95 | 待核 | 待核 |
| LightGBM LG | X5 | HoT | 2024 | H | LG / 24 h / d=15 h | 1 | A | 探索性 | 216.13 | 待核 | 待核 |
| LightGBM LUG | X5 | HoT | 2024 | H | LUG / 24 h / d=15 h | 1 | A | 探索性 | 206.89 | 待核 | 待核 |
| LightGBM LG_d6@15 | X5 | HoT | 2024 | H | LG / 24 h / d=15 h | 1 | A | 探索性 | 260.01 | 待核 | 待核 |
| LightGBM LUG_d6@15 | X5 | HoT | 2024 | H | LUG / 24 h / d=15 h | 1 | A | 探索性 | 243.15 | 待核 | 待核 |
| LightGBM LG_mix@15 | X5 | HoT | 2024 | H | LG / 24 h / d=15 h | 1 | A | 探索性 | 217.27 | 待核 | 待核 |
| LightGBM LUG_mix@15 | X5 | HoT | 2024 | H | LUG / 24 h / d=15 h | 1 | A | 探索性 | 207.32 | 待核 | 待核 |
| LightGBM LG | X5 | HoT | 2024 | H | LG / 24 h / d=18 h | 1 | A | 探索性 | 227.21 | 待核 | 待核 |
| LightGBM LUG | X5 | HoT | 2024 | H | LUG / 24 h / d=18 h | 1 | A | 探索性 | 211.48 | 待核 | 待核 |
| LightGBM LG_d6@18 | X5 | HoT | 2024 | H | LG / 24 h / d=18 h | 1 | A | 探索性 | 260.14 | 待核 | 待核 |
| LightGBM LUG_d6@18 | X5 | HoT | 2024 | H | LUG / 24 h / d=18 h | 1 | A | 探索性 | 241.54 | 待核 | 待核 |
| LightGBM LG_mix@18 | X5 | HoT | 2024 | H | LG / 24 h / d=18 h | 1 | A | 探索性 | 227.45 | 待核 | 待核 |
| LightGBM LUG_mix@18 | X5 | HoT | 2024 | H | LUG / 24 h / d=18 h | 1 | A | 探索性 | 211.49 | 待核 | 待核 |
| LightGBM LU | X5 | HoT | 2025 | H | LU / 24 h / d=6 h | 1 | A | 探索性 | 203.89 | 待核 | 待核 |
| LightGBM LG | X5 | HoT | 2025 | H | LG / 24 h / d=6 h | 1 | A | 探索性 | 193.39 | 待核 | 待核 |
| LightGBM LUG | X5 | HoT | 2025 | H | LUG / 24 h / d=6 h | 1 | A | 探索性 | 186.69 | 待核 | 待核 |
| LightGBM LG_mix@6 | X5 | HoT | 2025 | H | LG / 24 h / d=6 h | 1 | A | 探索性 | 194.22 | 待核 | 待核 |
| LightGBM LUG_mix@6 | X5 | HoT | 2025 | H | LUG / 24 h / d=6 h | 1 | A | 探索性 | 187.61 | 待核 | 待核 |
| LightGBM LG | X5 | HoT | 2025 | H | LG / 24 h / d=9 h | 1 | A | 探索性 | 195.09 | 待核 | 待核 |
| LightGBM LUG | X5 | HoT | 2025 | H | LUG / 24 h / d=9 h | 1 | A | 探索性 | 187.98 | 待核 | 待核 |
| LightGBM LG_d6@9 | X5 | HoT | 2025 | H | LG / 24 h / d=9 h | 1 | A | 探索性 | 195.94 | 待核 | 待核 |
| LightGBM LUG_d6@9 | X5 | HoT | 2025 | H | LUG / 24 h / d=9 h | 1 | A | 探索性 | 189.18 | 待核 | 待核 |
| LightGBM LG_mix@9 | X5 | HoT | 2025 | H | LG / 24 h / d=9 h | 1 | A | 探索性 | 195.82 | 待核 | 待核 |
| LightGBM LUG_mix@9 | X5 | HoT | 2025 | H | LUG / 24 h / d=9 h | 1 | A | 探索性 | 188.81 | 待核 | 待核 |
| LightGBM LG | X5 | HoT | 2025 | H | LG / 24 h / d=12 h | 1 | A | 探索性 | 200.90 | 待核 | 待核 |
| LightGBM LUG | X5 | HoT | 2025 | H | LUG / 24 h / d=12 h | 1 | A | 探索性 | 192.11 | 待核 | 待核 |
| LightGBM LG_d6@12 | X5 | HoT | 2025 | H | LG / 24 h / d=12 h | 1 | A | 探索性 | 218.64 | 待核 | 待核 |
| LightGBM LUG_d6@12 | X5 | HoT | 2025 | H | LUG / 24 h / d=12 h | 1 | A | 探索性 | 209.09 | 待核 | 待核 |
| LightGBM LG_mix@12 | X5 | HoT | 2025 | H | LG / 24 h / d=12 h | 1 | A | 探索性 | 201.85 | 待核 | 待核 |
| LightGBM LUG_mix@12 | X5 | HoT | 2025 | H | LUG / 24 h / d=12 h | 1 | A | 探索性 | 192.66 | 待核 | 待核 |
| LightGBM LG | X5 | HoT | 2025 | H | LG / 24 h / d=15 h | 1 | A | 探索性 | 213.96 | 待核 | 待核 |
| LightGBM LUG | X5 | HoT | 2025 | H | LUG / 24 h / d=15 h | 1 | A | 探索性 | 200.14 | 待核 | 待核 |
| LightGBM LG_d6@15 | X5 | HoT | 2025 | H | LG / 24 h / d=15 h | 1 | A | 探索性 | 253.67 | 待核 | 待核 |
| LightGBM LUG_d6@15 | X5 | HoT | 2025 | H | LUG / 24 h / d=15 h | 1 | A | 探索性 | 235.68 | 待核 | 待核 |
| LightGBM LG_mix@15 | X5 | HoT | 2025 | H | LG / 24 h / d=15 h | 1 | A | 探索性 | 214.61 | 待核 | 待核 |
| LightGBM LUG_mix@15 | X5 | HoT | 2025 | H | LUG / 24 h / d=15 h | 1 | A | 探索性 | 199.87 | 待核 | 待核 |
| LightGBM LG | X5 | HoT | 2025 | H | LG / 24 h / d=18 h | 1 | A | 探索性 | 223.46 | 待核 | 待核 |
| LightGBM LUG | X5 | HoT | 2025 | H | LUG / 24 h / d=18 h | 1 | A | 探索性 | 203.59 | 待核 | 待核 |
| LightGBM LG_d6@18 | X5 | HoT | 2025 | H | LG / 24 h / d=18 h | 1 | A | 探索性 | 254.30 | 待核 | 待核 |
| LightGBM LUG_d6@18 | X5 | HoT | 2025 | H | LUG / 24 h / d=18 h | 1 | A | 探索性 | 234.49 | 待核 | 待核 |
| LightGBM LG_mix@18 | X5 | HoT | 2025 | H | LG / 24 h / d=18 h | 1 | A | 探索性 | 224.00 | 待核 | 待核 |
| LightGBM LUG_mix@18 | X5 | HoT | 2025 | H | LUG / 24 h / d=18 h | 1 | A | 探索性 | 203.57 | 待核 | 待核 |
| LightGBM LG | X5b | Kelmarsh | 2022 | K | LG / 24 h / 状态 A | 1 | A | 探索性 | 147.86 | 待核 | 待核 |
| LightGBM LUG | X5b | Kelmarsh | 2022 | K | LUG / 24 h / 状态 A | 1 | A | 探索性 | 145.33 | 待核 | 待核 |
| LightGBM LG | X5b | Kelmarsh | 2022 | K | LG / 24 h / 状态 C | 1 | A | 探索性 | 161.90 | 待核 | 待核 |
| LightGBM LUG | X5b | Kelmarsh | 2022 | K | LUG / 24 h / 状态 C | 1 | A | 探索性 | 154.85 | 待核 | 待核 |
| LightGBM LG | X5b | Kelmarsh | 2022 | K | LG / 24 h / 状态 Cc | 1 | A | 探索性 | 162.98 | 待核 | 待核 |
| LightGBM LUG | X5b | Kelmarsh | 2022 | K | LUG / 24 h / 状态 Cc | 1 | A | 探索性 | 155.15 | 待核 | 待核 |
| LightGBM LG | X5b | Kelmarsh | 2022 | K | LG / 24 h / 状态 D | 1 | A | 探索性 | 161.54 | 待核 | 待核 |
| LightGBM LUG | X5b | Kelmarsh | 2022 | K | LUG / 24 h / 状态 D | 1 | A | 探索性 | 154.24 | 待核 | 待核 |
| LightGBM LG | X5b | Penmanshiel | 2021 | P | LG / 24 h / 状态 A | 1 | A | 探索性 | 163.72 | 待核 | 待核 |
| LightGBM LUG | X5b | Penmanshiel | 2021 | P | LUG / 24 h / 状态 A | 1 | A | 探索性 | 161.02 | 待核 | 待核 |
| LightGBM LG | X5b | Penmanshiel | 2021 | P | LG / 24 h / 状态 C | 1 | A | 探索性 | 176.20 | 待核 | 待核 |
| LightGBM LUG | X5b | Penmanshiel | 2021 | P | LUG / 24 h / 状态 C | 1 | A | 探索性 | 169.98 | 待核 | 待核 |
| LightGBM LG | X5b | Penmanshiel | 2021 | P | LG / 24 h / 状态 Cc | 1 | A | 探索性 | 176.11 | 待核 | 待核 |
| LightGBM LUG | X5b | Penmanshiel | 2021 | P | LUG / 24 h / 状态 Cc | 1 | A | 探索性 | 170.70 | 待核 | 待核 |
| LightGBM LG | X5b | Penmanshiel | 2021 | P | LG / 24 h / 状态 D | 1 | A | 探索性 | 175.09 | 待核 | 待核 |
| LightGBM LUG | X5b | Penmanshiel | 2021 | P | LUG / 24 h / 状态 D | 1 | A | 探索性 | 169.44 | 待核 | 待核 |
| LightGBM LG | X5b | HoT | 2024 | H | LG / 24 h / 状态 A | 1 | A | 探索性 | 197.58 | 待核 | 待核 |
| LightGBM LUG | X5b | HoT | 2024 | H | LUG / 24 h / 状态 A | 1 | A | 探索性 | 194.02 | 待核 | 待核 |
| LightGBM LG | X5b | HoT | 2024 | H | LG / 24 h / 状态 C | 1 | A | 探索性 | 216.13 | 待核 | 待核 |
| LightGBM LUG | X5b | HoT | 2024 | H | LUG / 24 h / 状态 C | 1 | A | 探索性 | 206.89 | 待核 | 待核 |
| LightGBM LG | X5b | HoT | 2024 | H | LG / 24 h / 状态 Cc | 1 | A | 探索性 | 216.40 | 待核 | 待核 |
| LightGBM LUG | X5b | HoT | 2024 | H | LUG / 24 h / 状态 Cc | 1 | A | 探索性 | 207.04 | 待核 | 待核 |
| LightGBM LG | X5b | HoT | 2024 | H | LG / 24 h / 状态 D | 1 | A | 探索性 | 216.16 | 待核 | 待核 |
| LightGBM LUG | X5b | HoT | 2024 | H | LUG / 24 h / 状态 D | 1 | A | 探索性 | 206.46 | 待核 | 待核 |
| LightGBM LG | X5b | HoT | 2025 | H | LG / 24 h / 状态 A | 1 | A | 探索性 | 193.39 | 待核 | 待核 |
| LightGBM LUG | X5b | HoT | 2025 | H | LUG / 24 h / 状态 A | 1 | A | 探索性 | 186.69 | 待核 | 待核 |
| LightGBM LG | X5b | HoT | 2025 | H | LG / 24 h / 状态 C | 1 | A | 探索性 | 213.96 | 待核 | 待核 |
| LightGBM LUG | X5b | HoT | 2025 | H | LUG / 24 h / 状态 C | 1 | A | 探索性 | 200.14 | 待核 | 待核 |
| LightGBM LG | X5b | HoT | 2025 | H | LG / 24 h / 状态 Cc | 1 | A | 探索性 | 214.02 | 待核 | 待核 |
| LightGBM LUG | X5b | HoT | 2025 | H | LUG / 24 h / 状态 Cc | 1 | A | 探索性 | 200.09 | 待核 | 待核 |
| LightGBM LG | X5b | HoT | 2025 | H | LG / 24 h / 状态 D | 1 | A | 探索性 | 212.98 | 待核 | 待核 |
| LightGBM LUG | X5b | HoT | 2025 | H | LUG / 24 h / 状态 D | 1 | A | 探索性 | 199.50 | 待核 | 待核 |

## 尚待正式结果的记录

WJB 运行提交 `1fe7ab6`。核岭回归与同延迟 X5 树模型配对，Cabin 与同历史 LightGBM 配对，TSG 与混合延迟 X5 树模型配对。四个实现的正式成绩均待核，预定 A 级、开发期 V2。历史 v25–v28 的 Kelmarsh TSMixer 属 C 级，逐版本回执待核。

## 来源与校验

每行来源标识对应一份指标文件。下列 SHA256 按回执核对原始字节或 Git 的 LF 内容。E1 单独列出结果文件的 LF 摘要；其结果内嵌预测哈希清单，独立复核回执记录了 39 项数组哈希及三项指标重算。

- **G1** · 运行提交 `27312d7` · [指标](../g1/results_78/formal/results.json) · [核验回执](../g1/results_78/formal/completed.json) · `34767c3b2053ecb58b381723953ee80b1812b487091135aa592f2714125822cb`
- **G2a** · 运行提交 `e72e462` · [指标](../g2a/results81/results.json) · [核验回执](../g2a/results81/completed.json) · `6f917d7afd53b7870fce9cba8881913ae5f2f39b7b8c46a60196ae1e72db7187`
- **G2c** · 运行提交 `2eeb23f` · [指标](../g2b/sealed_eval/server/results.json) · [核验回执](../g2b/sealed_eval/server/completed.json) · `13c501fc5d827ec46991dddfba28b729bd0aa58369db35b31090e133402566f3`
- **X1-A0** · 运行提交 `1d8479b` · [指标](../x1/A0/results.json) · [核验回执](../x1/A0/completed.json) · `2e84e22ee3b7f83939605fe82c8eca7bee2e82f7bd70cf4b4e2c1a03613acb85`
- **X1-A1** · 运行提交 `6e17435` · [指标](../x1/A1/results.json) · [核验回执](../x1/A1/completed.json) · `9595a71144f0a379ba72c9c917b6809315a6132f30059b6ec2668b2a24f5f5e5`
- **X1-A2** · 运行提交 `84f1b21` · [指标](../x1/A2/results.json) · [核验回执](../x1/A2/completed.json) · `ea524f3681a51efaef07cc18c67336b0b9baed0aad4eae66ef584f0b1b90c580`
- **X1-A3** · 运行提交 `e4819b0` · [指标](../x1/x1_e4819b0_gpu_r1/A3/results.json) · [核验回执](../x1/x1_e4819b0_gpu_r1/A3/completed.json) · `7ecb0f0ad2edbbce1205a911bbe2dbbe70ba0c0a73b34f4e6c80f2ae7f23a1b9`
- **X2** · 运行提交 `0f24875` · [指标](../x2/x2_0f24875/report/results.json) · [核验回执](../x2/x2_0f24875/report/completed.json) · `f774d171a407716a1d01941a0640381d652282fc32b3e14df2a82c7479889a9a`
- **X2-dev** · 运行提交 `0f24875` · [指标](../x2/x2_0f24875/report/dev_decisions.json) · [核验回执](../x2/x2_0f24875/report/completed.json) · `e3ec382ad8f7c5bf2f1e23253fbcb9a2b7d6622a22cab8bb3dae0035c93e24d4`
- **X2A2** · 运行提交 `655e921` · [指标](../x2/x2_655e921/report2/results.json) · [核验回执](../x2/x2_655e921/report2/completed.json) · `01fc851ecda06a0363938c203b5a025d64c97b155b946eac57fa9d76c463d0c5`
- **X2A2-dev** · 运行提交 `655e921` · [指标](../x2/x2_655e921/report2/dev_decisions.json) · [核验回执](../x2/x2_655e921/report2/completed.json) · `15f8c6098ea7a2c0ee53d4e1e0150e8f62bea5d2e6d1f73f0565b4d28686a718`
- **X3** · 运行提交 `f8af1b6` · [指标](../x3/x3_f8af1b6/report/results.json) · [核验回执](../x3/x3_f8af1b6/report/completed.json) · `15db54f5f01be58268ae5b0c509b4d8c16874a8cc30ca2af6114a6b5c554b848`
- **X3-dev** · 运行提交 `f8af1b6` · [指标](../x3/x3_f8af1b6/report/dev_decisions.json) · [核验回执](../x3/x3_f8af1b6/report/completed.json) · `8199094b012ff624aab9fd11fa44d05220c038b8dec31f3729e99d60811b3917`
- **X5** · 运行提交 `e01cc6b` · [指标](../x5/x5_e01cc6b/run/X5/results.json) · [核验回执](../x5/x5_e01cc6b/run/completed.json) · `b30bc758334a4bdcc469ed74b707f0ab6827c3ccd1ddb128eb331930481dc78f`
- **E1** · 运行提交 `5944271` · [指标](../e1_baselines/a2/r1/result.json) · [核验回执](../e1_baselines/a2/r1/independent_verification.json) · `11f2ed2cda97100c6dc87a2067490ba9f67efd901f7959628f3c42485e183fff`
- **X4** · 运行提交 `1c0a125` · [指标](../x4/x4_1c0a125/report/results.json) · [核验回执](../x4/x4_1c0a125/report/completed.json) · `fd1113915de34082a8795e3b8fafb6804d9b0c01a60209a954436c07005eaa74`
- **X5b** · 运行提交 `b7018c0` · [指标](../x5/x5_b7018c0/mrun/X5/results.json) · [核验回执](../x5/x5_b7018c0/mrun/completed.json) · `fc9083ce5a1c5f6d4dd525a57147d48ae45df52ddce6f089b5d833fdc4b07b87`
