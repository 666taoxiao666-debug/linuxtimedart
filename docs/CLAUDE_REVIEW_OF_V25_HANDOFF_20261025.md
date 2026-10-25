# 对《给 Claude：最新训练结果、实际架构与下一轮修改建议》的逐项判断

日期：2026-10-25。对象：仓库根目录 `CLAUDE_V25_RESULTS_AND_NEXT_DESIGN_20261025.md`（GPT 生成）。
依据：本地仓库源码（HEAD `0c1eb15`，训练版本 `54a8d3a`）与本地已下载的 v25 运行目录
`output/kelmarsh_hourly/backbone_information_v25_review/backbone_information_v25_f0-1-2_grq4Ps/`。
数字核对方式：直接从三份 DLinear 重放 NPZ 与六份 TSMixer `selection_predictions.npz` 用 numpy 重算（不依赖登记的汇总文件）。

结论先行：**事实部分全部接受**（数字逐项复算一致，源码位置正确）；假设部分支持开展有限的锚定残差试验。2026-10-25 部署前复审纠正了 B1 的过强因果判断与 B3 的门槛解释；保留原定 epoch1 起选择及全部数值设置。

## A. 事实陈述

| # | 文档陈述 | 判断 | 证据位置 |
|---|---|---|---|
| A1 | v25 = TSMixerBasicSmall，G1=false、I1=true，6 fit / 110 epoch，无 A2/G2/outer | 接受 | `inner_gate_seed2024.json`；各 `trained.json` epochs_completed 13+19+18+20+20+20=110 |
| A2 | 协议 SHA e111a794…、训练代码指纹 1f102371…、数据 SHA 5c9c5a1f…、训练提交 54a8d3a、本地 HEAD 0c1eb15 | 接受 | LF 规范化重算协议哈希一致；`trained.json.contract.training_code_sha256`；`git rev-parse` |
| A3 | 宏平均表（Persistence/DLinear/Full/PowerOnly 的 MAE、MSE、RMSE、R²、Skill、±5%/±10% 命中率） | 接受，逐位一致 | numpy 重算：如 Full MAE 166.248724、RMSE 244.987834、R² 0.683783 |
| A4 | 逐 fold MAE/RMSE、最佳/完成 epoch（Full 8/13、13/18、15/20；PowerOnly 14/19、19/20、15/20） | 接受 | `runs/f*_s2024/*/trained.json` |
| A5 | 宏 MAE 差 5.323865、RMSE 差 5.359053；三折 Full 均劣于 DLinear | 接受 | 重算一致 |
| A6 | 逐步长 MAE/RMSE/R²；Full−DLinear MAE 差 +10.24/+4.20/+3.64/+3.22，首小时占 48% | 接受（48.07%） | 重算一致 |
| A7 | MBE：Full −26.948、DLinear −19.575；首小时 −19.670 vs −9.504 | 接受 | 重算一致；另：Full 的偏差主要来自 fold1（−53.4 kW） |
| A8 | Holm 必须一起看；原始 p 0.247/0.731/0.0248 | 接受；Holm 校正后为 0.494/0.731/0.074，**三折均不显著** | `paired_statistics.json.primary_family` |
| A9 | I1 改善 8.915967 kW；不是单通道因果 | 接受。逐折 Full−PowerOnly CI：f0 [−2.7,12.9]、f1 [3.2,18.8]、f2 [−0.6,18.8]，只有 f1 区间不含 0 | `paired_statistics.json.full_vs_power_only` |
| A10 | 数据：7 通道顺序、单风机 24h→4h 直接多步、无未来协变量、无风机 ID | 接受 | `data_provider/kelmarsh_hourly.py:14,75` |
| A11 | `valid_window_starts` 第 32 行：历史 7 通道有限、未来只要求功率、同风机连续 | 接受 | `kelmarsh_hourly.py:32-45` |
| A12 | 第 81 行 scaler 只拟合 outer 训练前缀前 60% | 接受 | `kelmarsh_hourly.py:80-81`（`base_fit_end = int(train*.6)`） |
| A13 | v25 控制器重建 90%/10% 内部队列；不要用 `subset()` 默认的 60/10 | 接受 | `utils/kelmarsh_backbone_information.py:551` `inner_datasets` vs `kelmarsh_hourly.py:97` |
| A14 | fit/selection 窗口 5285/696、6341/816、7397/657；issue time 116/136/110；UTC 日 15/18/17；f2 有间断 | 接受（f2 有 3 个缺口，最长 63 h） | `a0_result.json.data.*.selection_issue_coverage` |
| A15 | 训练配方（AdamW 1e-3、wd 1e-4、batch 64、20 epoch、patience 5、clip 1、cosine 20、0.8MAE+0.2MSE 标准化空间、内部 MAE 选 epoch 含 epoch0） | 接受 | v25 协议 `training`；控制器 `fit_arm` |
| A16 | TSMixer 结构与参数 1759×2+100=3618；LN 在 (24,7) 平面；两支路残差 | 接受 | `models/TSMixerBasicSmall.py:41,46,48,53,66,94`；参数逐项相加一致 |
| A17 | “只取功率槽不等于辅助通道未进入输出” | 接受 | 通道 MLP 写回残差流的第 6 槽 |
| A18 | DLinear：kernel25 端点复制填充、individual=0、只对功率求损失→只用历史功率、200 参数 | 接受 | `models/DLinear.py:7,44,52`；v25 控制器 `:301` |
| A19 | v25 与 TimeDART/Wiki 无父子关系；v21 是 TimeDART 中间表示锚定 | 接受 | `models/PowerAnchoredTimeDART.py` |
| A20 | 结论口径：v25 未替换成功、内部≠outer、I1≠单通道因果、R²≠准确率、旧证书不能迁移 | 接受 | — |

## B. 文档未写、但会改变下一轮解读的证据

**B1. TSMixer 的函数族能表示 DLinear，但这不能单独定位实际失败原因。**
TSMixer 的残差流原样携带标准化功率历史，输出头是 `Linear(24→4)`；只要两个 block 的增量为 0，输出就是功率历史的任意仿射映射。DLinear 的输出 `W_s(I−A)x + W_t A x + b` 也是这一类映射（A 为带端点填充的滑动平均矩阵）。因此“结构上绝对无法表示参照”不是充分解释。**TSMixerPowerOnly 比 DLinear 差 14.24 kW MAE**，大于 Full 相对 PowerOnly 的 8.92 kW 差，支持调查估计、优化、归纳偏置与选择稳定性；现有证据不能在随机初始化、dropout、早停或时间漂移之间唯一归因，也没有排除输入信息不足。锚定残差是一个可证伪的后续试验。

**B2. 先例：内部门槛通过 ≠ outer 改善。**
`docs/KELMARSH_SYSTEM_DIAGNOSIS_20261024.md:120`：v21 内部 seed2024 宏 MAE/RMSE 158.79/235.68，**优于**同段 DLinear 160.92/239.63，但 outer 上排序反转。另 `docs/KELMARSH_NUMERIC_ADAPTER_DIAGNOSIS.md:18`：早期零头残差设计在 fold1 三个 seed 都选回了 epoch0。v26 即使过 G1，也只是开发证据。

**B3. epoch0 不参与选择是诊断取舍，不是修复恒真 G1。**
锚定后 epoch0 就是 DLinear，纳入内部 MAE 选择会保证 MAE 不劣；但 G1 要求 MAE 和 RMSE **严格改善**并满足跨 fold 条件，打平仍然失败，因此原建议并不使 G1 恒真。v26 保留 epoch0 仅记录、从 epoch1 起选择的预声明方案，用于直接披露训练后残差的表现（包括退化）。它与 v25 的选择策略不同，不能声称两者选择偏差对称；锚点与残差都用过这段 selection，结论仍是开发证据。协议的说明文字已纠正，数值规则不变。

## C. 优化假设逐条

| 文档条目 | 判断 | 说明 |
|---|---|---|
| 优先级1：保留冻结 DLinear，只学辅助信息增量 | 接受（见 B1，证据比文档更强） | 已实现为 v26，单一机制 |
| “残差形式不能宣称为创新” | 接受 | 协议 `novelty_claim: none` |
| 优先级2：训练侧分步长/偏差/工况诊断，不能按 selection 修数值 | 接受 | 新增只读脚本 `scripts/diagnose_kelmarsh_fit_side_v25.py`：工况在看误差前冻结（额定功率物理分档；爬坡三分位只用 fit 窗口）；禁止把观察到的偏差加回预测（写入协议 forbidden） |
| 优先级3：不要先删 LN；LN/置零会改变统计 | 接受 | v26 残差支路结构完全不改；I1 解释文字注明置零也改变 LN 输入统计 |
| 优先级4：用冻结 checkpoint 做 fit 上 eval 推理，统一尺度后再谈过/欠拟合 | 接受 | 同上脚本；v26 每个 fit 结束也自动记录 `fit_eval_best` |
| 优先级5：数据/信息预算可能限制精度，但无“脏数据主因”证据 | 接受 | 本轮不清洗、不插值、不换数据、不加 NWP |
| 6.1：最后一层权重与偏置置零、不加门控；不要同时把头和门都置零 | 接受 | 仅 `residual.head` 置零，无门控/缩放 |
| 6.1：epoch0 参与选择、选回即零增益 | 有效备选；本轮选另一诊断策略（B3） | epoch0 记录但不可选；并非原门槛无效 |
| 6.2：三臂（冻结 DLinear 重放、AnchoredFull、AnchoredPowerOnly），首轮 seed2024×3fold×2 = 6 fit，≤20 epoch | 接受 | A2 条件追加的残差 seed 2025/2026 只训练两臂，**锚点不重训**，总上限 18 fit / 360 epoch（v25 为 24/480） |
| 6.2：fit 残差是训练内残差，不能叫 OOF、不能做证书 | 接受 | 协议 `residual_targets_are_in_sample` |
| 6.3：沿用 G1 + 预先声明首小时不劣于 DLinear | 接受 | G1/G2 都加入 lead-1 宏 MAE 与 RMSE 不劣于 DLinear（容差 1e-6 kW，只用于浮点打平） |
| 6.3：同时检查 Full 是否真正优于 PowerOnly | 接受，作为**归因规则** | I1/I2 不改变主候选，但只有 I1（及 A2 跑了时的 I2）成立，才允许说“辅助信息带来增量” |
| 7：十项测试要求 | 接受 | 对应测试见设计说明第 6 节 |
| 8：Wiki 边界 | 接受 | v26 不涉及 Wiki、不读旧证书 |

## D. 需要用户知道的一项取舍

G1 通过但 I1 不成立时，v26 仍按协议进入 A2（数值上可能有用，但不能归因于辅助通道）。如果你希望“辅助信息增量”是唯一目标、I1 不成立就停止，需要在**训练前**把 I1 并入 G1 并重新冻结协议（协议哈希会变）。当前版本选择前者，理由是它与 v25 的结构一致，且不丢弃可能的数值改进。
