# v27 训练侧前向分块诊断：设计说明、v26 证据链与修改清单

日期：2026-10-25。完整协议 ID：`kelmarsh_forward_block_residual_diagnostic_v27`（不要只写“v27”）。
状态：原稿已完成代码审查，用户已授权修正、部署及本轮训练；真实训练仍须先通过服务器 `--preflight-only`。本次运行前修订不改网络、训练配方、44 fit 预算和窗口队列，调整隔离实现、身份核验和不充分的结论规则。原稿协议 SHA 为 `702f4e537c737b1fef7165668da981a34911043aef0d03db6007ab0fef76a2ec`，从未执行。

执行补记：审查版已通过正式预检并完成44 fits/396 epochs，独立复核807/807通过；协议结论 STOP。详见 `docs/CLAUDE_FORWARD_BLOCK_RESIDUAL_HANDOFF_20261025.md`。本文保留运行前设计，不因负结果修改规则。

## 1. v26 运行身份核对（交接 §9.1）

| 项目 | 交接所写 | 本地核对 |
|---|---|---|
| 离线包 zip SHA-256 | — | 上传件与 `output/kelmarsh_hourly/CLAUDE_V26_COMPLETE_HANDOFF_20261025.zip` 相同（`21c74854…`） |
| completed.json | `5ad7e261…` | 一致 |
| 协议 ID / SHA | `kelmarsh_anchored_residual_v26` / `e11f80a6…` | 一致（`protocol.json`） |
| training code / 全源码身份 | `01ba99c3…` / `1ecae5eb…` | 一致（`source_manifest.json`） |
| 终态 | COMPLETED | TRAIN_COMPLETED / REPORT_COMPLETED |
| G1/I1/G2/I2 | 过/过/过/不过 | 一致（`inner_gate_*.json`） |
| preflight、独立审计、权重审计 | 通过 | `preflight.json`、`independent_audit.json`、`independent_weight_audit.json` 均为 passed |
| NumPy 独立审计 | 774 项通过 | 在临时副本上重跑 `audit_v26_run.py`：774/774 通过，18 fit、142 epoch；未写入仓库 |

## 2. v26 的证据链：哪些已证实，哪些只是假设（交接 §9.3）

**已证实（数字可复算）**

- 内部 G2 通过的幅度很小。单 seed 宏 MAE 从 160.92 到 160.54 kW（−0.39），每折只有 110–136 个 issue time。这些 selection 窗口同时被用来选残差 epoch（锚点的 epoch 当年也在这里选），所以内部增益偏乐观。
- I2 失败：三 seed 平均后只有 fold1 的 Full 同时优于 PowerOnly。
- outer 宏 MAE 只改善 0.46 kW，三折区间都跨零，Holm 校正后 p 为 1/1/0.257。+1h 和 +2h 变差，+3h 和 +4h 各好约 1 kW。
- 残差修正的 seed 间相关只有 0.28–0.35；修正量与锚点误差的相关为 0.05–0.13；三 seed 平均预测在内部和 outer 的三折都优于锚点（`claude_v26_architecture_diagnostics_output.txt` D1/D2/D10）。
- MBE 为负且在远步长加深，中位误差接近 0，误差右偏（D4）。
- outer 段的功率水平高于训练与 selection 段（D6）。

**假设（v26 本身无法区分）**

- H1：残差的训练收益本身不能向后迁移，内部 G2 通过主要来自 selection 乐观性。
- H2：残差的主要问题是 seed 方差；固定 K 个 seed 平均能稳定变好。
- H3：辅助通道确有增量，但小且不稳定，所以 I2 失败。
- H4：+1h 退化是因为 DLinear 在 +1h 已接近最优，残差只加噪声。
- 季节漂移对 outer 的贡献有多大：本诊断**测不到**（见第 6 节）。

以上描述性数字都来自已开发查看的段，只能提出假设。本轮在 v20–v26 的 fit 前缀里做前向诊断；更早协议曾使用其中部分数据，不能称从未查看的数据。R2 只能量化平均的误差抵消，不能证明 H2 是主因；R4 只能检查是否伤害近端，不能证明 H4 所说的锚点已达最优。

## 3. 设计（一页）

**范围**：只用已登记的 fold-0 FIT 窗口（目标在 fold-0 `calibration_early_end` = 2016-09-10T07:00 之前；窗口哈希须等于 v20/v25/v26 登记的 `9c843ac5…`）。这段在 v20–v26 中始终只是训练数据，从未作为 selection 或 outer。更早的 v19 及 Wiki 协议曾把其中一段用作内部选择或记忆，但从未作为外层评估。

**分块**：把 [2016-05-02T00, 2016-09-10T07) 均分为 6 块，边界为 05-02、05-23T21、06-14T18、07-06T15、07-28T12、08-19T09、09-10T07。第 k = 3..6 步：

| 步 | 训练段 fit / selection 分界 | 训练结束 | 评估块 |
|---|---|---|---|
| 3 | 2016-06-10T09 | 06-14T18 | 06-14T18 → 07-06T15 |
| 4 | 06-30T01 | 07-06T15 | → 07-28T12 |
| 5 | 07-19T18 | 07-28T12 | → 08-19T09 |
| 6 | 08-08T10 | 08-19T09 | → 09-10T07 |

窗口的四个目标必须全部落在对应段内；跨边界的窗口丢弃。

**每一步只用过去的数据重做 v26 的完整流水线**：
- **scaler**：只用 fit 分界之前的行。
- **DLinear 锚点**：v20 配方，由 v25 训练函数原样训练（90/10 选 epoch，epoch0 可选）。必须重训，因为冻结的 v20 锚点见过这些标签。
- **残差**：AnchoredFull 和 AnchoredPowerOnly 各 5 个 seed（2024–2028），由 v26 训练函数原样训练，叠在该步锚点上（epoch1 起选、patience 5）。
- **评估**：只在下一块上算锚点、每个残差（选中 epoch，另记录最后 epoch 作诊断）、5-seed 平均和 Persistence。评估块不参与任何拟合、scaler、epoch 或阈值。

**预算**：4 步 × 11 fit = 44 fit，每个 ≤ 20 epoch（≤ 880）。v26 的 18 个同规模 fit 用了约 3 分钟，这里预计 10 分钟量级。

**预声明判据**（`decision`，全部在评估块上）：

| 规则 | 内容 | 检验什么 |
|---|---|---|
| R1 | Full 5-seed 平均在 ≥3/4 步同时优于该步锚点的 MAE 和 RMSE，且所有评估窗口合并后的 MAE 增益 95% 区间下界 > 0（issue-time 聚类、8 块 bootstrap、5000 次） | H1：残差收益能否向后迁移 |
| R2 | Full 5-seed 平均的 MAE 在 ≥3/4 步优于单 seed 平均 | 描述误差抵消；MAE 凸性已保证非劣，不能识别方差是否主因 |
| R3 | Full 5-seed 平均在 ≥3/4 步联合优于 PowerOnly 5-seed 平均，且合并区间下界 > 0 | H3：辅助通道归因 |
| R4 | 合并后的 +1h MAE 和 RMSE 不劣于锚点 | H4：近端保护 |
| R5 | 预声明 seed2024 在 ≥3/4 步联合优于锚点，合并 MAE 收益区间下界>0，且其自身 +1h MAE/RMSE 不劣 | 无集成路线必须由实际单 seed 执行器提供证据 |

结论只有三种：
- **R1 ∧ R2 ∧ R4** → `BAGGED_RESIDUAL_MAY_ENTER_CONFIRMATION`：允许再设计**一个**确认协议（K=5 平均残差，最好在从未查看过的年份上），并事先冻结。
- **R1 ∧ R4 ∧ ¬R2 ∧ R5** → `RESIDUAL_MAY_ENTER_CONFIRMATION_WITHOUT_BAGGING`（固定 seed2024，不能从评估中选 seed）。
- **其余情况** → `STOP_RESIDUAL_LINE_ON_THIS_DATA`：不再在 2016 开发窗口上提出残差变体，下一步转向信息/数据预算（多年份、小时内末端状态、风机身份，各自单独立协议）。

辅助通道归因只在 R3 成立时允许。任何结果都不能当作预测精度结论，也不触发 outer 评估。

另外输出描述性诊断（不进入判据）：每步长修正均值/幅度、修正与锚点误差的相关、seed 间相关、选中 epoch 与最后 epoch 在评估块上的差距、按最后观测功率的物理分档和按风机的误差（附样本量）。

## 4. 与 v26 的差异

| 项目 | v26 | v27 |
|---|---|---|
| 目的 | 候选底座的开发评估 | 训练侧诊断，不产生精度结论 |
| 数据段 | 各 fold 的 fit/selection，G2 后 outer | 只用 fold-0 FIT 段的 6 个前向块；不实例化任何 selection、outer 或封存段 |
| 锚点 | 冻结 v20 seed2024 | 每步只用过去数据重训（同 v20 配方、同 v25 代码） |
| scaler | v20 继承（outer-train 前 60%） | 每步只用 fit 分界前的行 |
| 残差结构/配方 | TSMixerBasicSmall 零头，epoch1 起选 | 完全相同，由 v26 训练函数原样调用 |
| seed | 2024；条件追加 2025/2026 | 每步固定 5 个（2024–2028），事先固定，不追加 |
| 判据 | G1/G2/I1/I2 | R1–R5 → 三种结论之一 |
| 预算 | 18 fit | 44 fit（模型同样小） |

## 5. 新增的文件（没有修改任何已有文件）

| 文件 | 内容 |
|---|---|
| `configs/kelmarsh_forward_block_residual_protocol_v27.json` | 审查后冻结协议，SHA-256（LF）`b9918206a61ca655957212fd03f4831dc9eda1d032cfe45ef7b328b154d6e4b4` |
| `utils/kelmarsh_forward_block_residual.py` | 不依赖 torch：分块、步内分段、逐步 scaler、两个训练函数的协议视图、指标、合并统计、R1–R5 判据、描述性诊断 |
| `scripts/train_kelmarsh_forward_block_residual.py` | 控制器 B0/B1/B2、断点复用、锁、指针、preflight（测试、真实数据 B0、两个训练函数串联的合成冒烟、完整协议 ID 命名检查） |
| `scripts/report_kelmarsh_forward_block_residual.py` | 不依赖 torch 的报告：从登记数组独立重算全部记录、合并统计与判据，与登记值不一致即拒绝；2 张图 |
| `scripts/train/Kelmarsh_launch_forward_block_residual.sh` | 启动器（LF） |
| `tests/test_kelmarsh_forward_block_residual.py` | 协议、分块隔离、scaler 截断、视图键、判据各分支、报告、篡改拒绝；需要 torch 的数据视图等价与端到端控制器测试 |
| `docs/KELMARSH_FORWARD_BLOCK_RESIDUAL_V27.md` | 本文件 |

训练数值依赖的代码（含被复用的 `scripts/train_kelmarsh_backbone_information.py` 与 `scripts/train_kelmarsh_anchored_residual.py`）全部进入 v27 的 training code 指纹。v25/v26 的协议、代码与产物都没有改动。

审查修订：读取器先只读取时间戳，再以行号跳过前缀外的数值列；保持原未封存网格作为元数据和 NaN 占位，以维持登记窗口编号。仅步内 fit 行拟合 scaler。训练指纹现在包括整个本轮控制器及依赖，完成入口核对最终产物，报告核对全部规则和配对数值。

## 6. 局限

- 六块都在 5–9 月，只能检验“夏季内向下一块的迁移”，检验不了 v26 outer 的夏到秋漂移。季节漂移只能靠多年份数据回答。
- 第 3 步的训练段最短（约 44 天，selection 约 30 个 issue time），选 epoch 很嘈杂。这本身就是 v26 配方的一部分，所以如实保留。
- 每个评估块约 22 天、约 170 个 issue time。合并后约 700 个 issue time，比 v26 的单折 outer 多，但仍是同一年。
- 判据只决定“是否值得设计下一个确认协议”。即便得到 BAGGED 结论，也不能在 2016 的 outer 上做确认（那些窗口已多次查看）。
- 四个 expanding-history 步骤并非四次独立复现；合并区间是条件于本次配方、连续夏季时段的近似诊断，不能证明跨季节或跨年份泛化。R3 只允许在本次消融条件下讨论辅助通道增量。

## 7. 训练机上的执行顺序（需用户授权）

```bash
bash scripts/train/Kelmarsh_launch_forward_block_residual.sh --preflight-only   # 测试 + 真实数据 B0 + 合成冒烟 + 命名检查
bash scripts/train/Kelmarsh_launch_forward_block_residual.sh --detach           # 预检通过后：4 步 × 11 fit，随后判据与报告
bash scripts/train/Kelmarsh_launch_forward_block_residual.sh --status
bash scripts/train/Kelmarsh_launch_forward_block_residual.sh --resume --detach  # 中断后恢复，已完成的 fit 不重训
bash scripts/train/Kelmarsh_launch_forward_block_residual.sh --report-only
```

结果目录：`outputs/logs/Kelmarsh/<日期>/forward_block_residual_v27_XXXXXX/`，指针 `outputs/logs/Kelmarsh/forward_block_residual_latest.txt`。

## 8. 测试位置

本机沙箱没有 torch，按用户的工作方式只做了语法检查和不依赖 torch 的测试（v27 有 14 项通过，v25/v26 的不依赖 torch 回归测试仍通过）。需要 torch 的 3 项 v27 测试，以及 v26 模型与控制器测试，由 preflight 在训练机上运行，任何一项不通过都不能开始训练。
