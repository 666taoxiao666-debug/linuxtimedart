# v26 设计说明与修改清单：冻结 DLinear 预测锚定 + 零初始化残差

日期：2026-10-25。状态：**用户已授权复审、提交部署和训练；必须先通过服务器 preflight**。实际提交、测试和结果以本轮审核及最终交接文档为准。
逐项审查见 [CLAUDE_REVIEW_OF_V25_HANDOFF_20261025.md](CLAUDE_REVIEW_OF_V25_HANDOFF_20261025.md)。

## 1. 一页设计说明

**问题。** v25 中 TSMixerFull 比 PowerOnly 好 8.92 kW MAE，却在三折都输给只用历史功率的 DLinear，损失有 48% 集中在 +1h。TSMixer 的函数族能表示 DLinear 的仿射功率映射；与 DLinear 信息相同的 TSMixerPowerOnly 反而差 14.24 kW。这支持检验估计与优化是否可以改善，但不能单独认定初始化或某一个模块是根因，也不能排除信息不足和时间漂移（见审查文档 B1）。

**单一机制。** 数据、队列、scaler、残差支路结构、loss、优化器、学习率调度、每次 fit 的预算都与 v25 相同，只把“从头预测功率”改成

    ŷ_h(x) = b_h(x) + r_h(x)        （标准化功率空间）

其中 b 是同 fold 已冻结的 v20 seed2024 DLinear（只用功率槽，始终 eval，不求梯度，不进优化器）。r 是原封不动的 TSMixerBasicSmall（3618 参数），**只把最后的 `Linear(24→4)` 的权重和偏置置零**，不加门控或缩放。因此训练前预测与 DLinear 逐样本相等；r 只能学习“相对 DLinear 的增量”。AnchoredPowerOnly 在 b 和 r 之前都把 0–5 通道置 0（DLinear 的功率输出与其它通道无关，所以锚点在两臂中完全相同），用来判断增量能否归因于辅助通道。

**为什么先做这个。** v25 说明更大的函数族不保证更好，尚无证据证明加深一定改善或一定恶化。清洗和插补会改变对照队列，当前也缺少支持证据。换数据或加 NWP 是新的信息预算，需要另立协议。锚定残差以已知参照为起点，测试增量是否可学；训练后并不保证保留参照精度，最多 18 次小模型 fit。

**与文档建议的一处不同。** epoch0 记录但**不可被选中**（审查文档 B3），目的是显式披露训练后退化。纳入 epoch0 只保证 selection MAE 不劣，不能让要求严格联合改善的 G1 自动通过。另有 lead1 门槛及预算变化，已逐项列出，不能称整个实验只改变了一个设置。

**允许的结论。** G1 失败则保留负结果并停止这条机制，不调参重试。G1 通过只说明数值上有改进；只有 I1（以及跑了 A2 时的 I2）也成立，才能说辅助信息带来了增量。任何 outer 数字都是开发证据（outer 以前看过，锚点的 epoch 当年也是在这段内部 selection 上选的）。这条机制不构成论文创新。

## 2. 与 v25 的逐项差异

| 项目 | v25 | v26 |
|---|---|---|
| 预测 | y = TSMixer(x) | y = DLinear_frozen(x) + TSMixer_zero_head(x) |
| 可训练 / 冻结参数 | 3618 / 0 | 3618 / 200（锚点） |
| 输出头初始化 | Xavier，bias 0 | 先按原 RNG 抽取 Xavier，再把权重和 bias 置 0（mixer block 与 v25 同 seed 完全相同） |
| 锚点构造 RNG | — | `torch.random.fork_rng`，不消耗全局随机数 |
| 置零消融位置 | TSMixer 之前 | 锚点与残差之前（锚点不受影响） |
| epoch0 | 参与选择（随机网络） | 记录且必须等于 DLinear，**不可选**；patience 从 epoch1 计 |
| 优化器 / 裁剪 | 全部参数 | 只含可训练残差参数 |
| 门槛 | G1/G2：宏 MAE+RMSE 优于 DLinear/Persistence，≥2 fold 联合优于 | 同上，**加**：lead-1 宏 MAE 与 RMSE 不劣于 DLinear（1e-6 kW） |
| 归因 | I1/I2 诊断 | I1/I2 是声称“辅助信息增量”的必要条件 |
| A2 | Full、PowerOnly、DLinear 共 18 fit | 只训练两臂共 12 fit；锚点不重训，所有残差 seed 都对同一个锚点比较 |
| 预算 | 24 fit / 480 epoch | 18 fit / 360 epoch |
| A3 | outer 一次评估 | 同上，另要求锚点在 outer 上重放 = v20 登记的 DLinear outer 预测（逐样本 ≤ 0.01 kW） |
| 额外记录 | — | 每个 fit 的 selected/last/epoch0 指标、逐步长指标、fit 窗口 eval 指标、锚点状态哈希；报告新增“改动幅度/是否有益”和 fit 与 selection 同尺度对照，以及 v25 历史列（只读） |
| 不变 | 数据/CSV 哈希、窗口、7 通道、标签、scaler、loss、LR、batch、epoch、patience、clip、cosine、dropout、残差结构、统计方法 | 同左 |

## 3. 新增与修改的文件

全部为**新增**。v25 协议、`models/TSMixerBasicSmall.py`、`models/DLinear.py`、数据类与 v25 任何产物都**未修改**（测试会核对 v25 协议哈希仍等于 v25 控制器常量）。

| 文件 | 内容 |
|---|---|
| `configs/kelmarsh_anchored_residual_protocol_v26.json` | 训练前纠正 epoch0 说明文字后冻结，数值设置未变；SHA-256（LF）`e11f80a64ae50f3782c99ec16b7eb591099714866f7842d6b226677aeb698eb9`（原草案 `f8e0680b…` 尚未执行） |
| `models/DLinearAnchoredResidual.py` | 锚定残差模型 |
| `utils/kelmarsh_anchored_residual.py` | 不依赖 torch 的协议校验、epoch 规则、G1/G2（含 lead-1）、I1/I2、访问审计（复用 v25 通用工具） |
| `scripts/train_kelmarsh_anchored_residual.py` | 控制器 A0–A4、断点恢复、preflight（含 GPU 与 CPU 的 epoch0 一致性探针） |
| `scripts/report_kelmarsh_anchored_residual.py` | 不依赖 torch 的报告：独立重算、配对统计、Holm、各类披露、5 张图 |
| `scripts/train/Kelmarsh_launch_anchored_residual.sh` | 启动器（LF 换行） |
| `utils/kelmarsh_fit_side_diagnostics.py`、`scripts/diagnose_kelmarsh_fit_side_v25.py` | 只读诊断：冻结 v25/v20 模型在 fit 与内部 selection 上的同尺度、逐步长、分工况、分风机误差，不训练 |
| `tests/test_dlinear_anchored_residual.py` | 模型测试（需要 torch） |
| `tests/test_kelmarsh_anchored_residual.py` | 协议/门槛/报告测试（不需要 torch）+ fit/恢复/控制器测试（需要 torch） |
| `docs/KELMARSH_ANCHORED_RESIDUAL_V26.md`、`docs/CLAUDE_REVIEW_OF_V25_HANDOFF_20261025.md` | 本文件与逐项审查 |

## 4. 运行时的身份校验

- **A0（不训练）**：核对 v20 协议、completed.json、嵌套清单、三个 checkpoint 哈希、best epoch、窗口哈希、scaler 数组、登记的内部 MAE/RMSE；在 CPU 上重放锚点，复现登记指标（≤ 0.001 kW）；在 CPU 上逐样本检查两臂训练前的预测**严格等于**锚点（标准化与 kW 空间的最大差都必须为 0），并检查两臂初始状态哈希相同。
- **每个 fit 开始**：在训练设备上，epoch0 预测与登记的 CPU 重放逐样本相差 ≤ 0.01 kW，才写 init.pt 并开始训练。
- **每个 fit 结束**：best.pt 和 last.pt 中锚点子状态哈希不变；best.pt 能复现登记的 selection MAE；选中的 epoch 与独立的选择规则函数一致。
- **恢复**：last.pt 同时保存被选中的状态，是唯一来源；恢复时据此重写 best.pt，因此崩溃在写 best.pt 与写 last.pt 之间也不会留下不一致。已完成的 fit 只做哈希核验，不重训；契约或锚点改变则拒绝。

## 5. 训练机上的执行顺序（需要用户授权）

```bash
# 0) 可选、只读、不训练：v25/v20 冻结模型的 fit 侧诊断
python scripts/diagnose_kelmarsh_fit_side_v25.py
# 1) 预检：跑 v26 全部测试和回归测试、真实数据 A0、GPU 与 CPU 一致性探针、合成数据冒烟、启动器语法
bash scripts/train/Kelmarsh_launch_anchored_residual.sh --preflight-only
# 2) 预检通过后才能训练（A1 共 6 fit；G1 通过才进入 A2/A3）
bash scripts/train/Kelmarsh_launch_anchored_residual.sh --detach
bash scripts/train/Kelmarsh_launch_anchored_residual.sh --status
# 恢复 / 只重做报告
bash scripts/train/Kelmarsh_launch_anchored_residual.sh --resume --detach
bash scripts/train/Kelmarsh_launch_anchored_residual.sh --report-only
```

结果目录：`outputs/logs/Kelmarsh/<日期>/anchored_residual_v26_f0-1-2_XXXXXX/`，指针 `outputs/logs/Kelmarsh/anchored_residual_latest.txt`。

## 6. 测试覆盖与运行位置

Claude 原沙箱没有 torch。部署复审使用 Windows 本地 Torch 和 Linux 服务器：本地检查发现独立参照未冻结导致精确身份测试失败，已统一参照与嵌入锚点的冻结执行状态，未放宽容差。完整控制器测试由服务器 preflight 运行，**任何一项不通过都不能开始训练**。GPU 身份探针也执行本进程显存上限及低余量 CPU 回退策略。

| 第 7 节要求 | 测试 | 需要 torch |
|---|---|---|
| 1 输出形状、目标时间与队列一致 | A0 窗口哈希/队列核对；`test_input_validation`；报告对齐检查 | 部分 |
| 2 epoch0 与冻结 DLinear 逐样本一致（标准化与 kW） | `test_initial_forecast_is_exactly_the_frozen_dlinear`；A0 `epoch0_identity`；`test_fit_starts_at_anchor...` | 是 |
| 3 锚点冻结、不在优化器中、训练后哈希不变 | `test_gradients_head_first_then_blocks_anchor_never`；`test_fit_starts_at_anchor...` | 是 |
| 4 头首步有限非零梯度，block 第二步起有梯度 | `test_gradients_head_first_then_blocks_anchor_never` | 是 |
| 5 置零对照不受辅助通道影响；初始化/参数/顺序/设备相同 | `test_anchor_is_channel_independent...`；`test_paired_arms_have_identical_state`；控制器测试 | 是 |
| 6 身份不符即停止 | `test_wrong_anchor_reference...`；`test_pairs_share_init_and_changed_anchor...`；A0 哈希检查 | 是 |
| 7 恢复与不中断一致、完成不重训、篡改拒绝、报告失败只补报告 | `test_interruptions_resume_identically...`；`test_refused_start_leaves_no_init...`；`test_report_failure_never_reruns_training...`；报告篡改测试 | 部分 |
| 8 G1 失败/打平/缺记录/单 fold/epoch0 的处理；不偷偷请求 outer | `GateTests`；`AccessTests`；`test_failed_g1_stops_without_outer...` | 否（控制器部分需要） |
| 9 从 NPZ 独立复算全部指标 | 报告 `recomputation`（含逐步长）；`ReportTests` | 否 |
| 10 issue-time 聚合、块 bootstrap/NW、Holm | 复用 v25 已验证的统计实现，并与 `paired_forecast_statistics` 交叉核对 | 否 |

## 7. 已知局限

- 内部 selection 只有 110–136 个 issue time；epoch 和锚点 epoch 都在这段上选，内部增益偏乐观。
- 门槛用的 DLinear 总体 MAE/RMSE 来自 v20 登记历史（与 v25 相同），lead-1 来自 CPU 重放；两者差 < 0.001 kW，只有极接近打平时才可能影响 1e-6 的判定。
- 锚点在 fit 窗口上训练过，残差学习的是训练内残差；不能作为可迁移的收益证书。
- 0.01 kW 的设备容差在 preflight 的 GPU 探针中实测；若探针失败，需先检查原因，不能直接放宽。
