# v25 Kelmarsh 数值底座信息实验：实现、协议与恢复说明

日期：2026-10-24。协议：`configs/kelmarsh_backbone_information_protocol_v25.json`（CRLF→LF 归一后 SHA-256 `e111a7947004c5a58b486d57c4544a97e14192f966a33740523b16857b947653`）。

## 0. 当前真实状态

实现、协议冻结文件和测试已完成，但本轮**尚未部署、未训练、没有任何新精度数字**。原因是实施所用的沙箱无法连到服务器（`100.89.238.37` 网络不可达）和 GitHub（代理 403），沙箱里也没有 torch。

实际已验证的内容：

- 不依赖 torch 的测试已在沙箱运行并通过：门槛、配对统计、RMSE 区间、Holm、对齐、设备策略、数据访问审计、锁、协议冻结，以及 A4 报告的端到端合成测试（`tests/test_kelmarsh_backbone_information.py` 中 20 项通过、6 项因无 torch 跳过；`tests/test_kelmarsh_backbone_resume.py` 中 3 项报告测试通过、7 项跳过）。
- 依赖 torch 的测试（模型、数据边界、与 v20 `train_stage` 的逐位等价、恢复、控制器全流程）已编写，但只能在服务器的 `timedart` 环境里运行。`--preflight-only` 会强制运行这些测试，测试不通过就不能开始任何真实训练。
- 上游 TSMixer 的 commit 和文件 SHA-256 尚未固定（`configs/tsmixer_upstream_pin_v25.json` 中仍为 null）。固定之前，控制器会拒绝开始 A1。

## 1. 问题与边界

在不改动的 v20 队列上（24 小时输入、4 小时直接输出、6 台风机、3 个滚动 fold、v20/v19 scaler），本轮回答两个问题：

1. 一个小型 TSMixer-basic 移植能否在内部选择段稳定超过冻结的 DLinear 和 Persistence？
2. 六个辅助 SCADA 通道作为一个整体，对这套架构和配方是否有用？

TSMixer 是已发表的工作，**不是创新点**。Wiki 关闭。本轮不加特征、不追加 seed、不清洗标签，也不碰封存 test。所有数字都是开发证据（outer 和 inner 在历史版本中都已被看过）。

## 2. 新增文件（旧文件均未修改）

| 文件 | 作用 |
|---|---|
| `models/TSMixerBasicSmall.py` | 纯模型，`forward(history)` 只接受历史输入 |
| `utils/kelmarsh_backbone_information.py` | 不依赖 torch：G1/G2/I1/I2、配对损失统计、RMSE 差 bootstrap、Holm、对齐检查、独立重算指标、图片片段规则、设备决策、数据访问审计、锁、内部队列构造 |
| `scripts/train_kelmarsh_backbone_information.py` | 状态机 A0–A4、单臂训练与恢复、preflight、状态查询 |
| `scripts/report_kelmarsh_backbone_information.py` | 不依赖 torch 的 A4 报告，只读取已注册的预测 |
| `scripts/train/Kelmarsh_launch_backbone_information.sh` | 首跑、`--status`、`--resume`、`--preflight-only`、`--report-only`、`--detach` |
| `scripts/deploy/v25_server_ops.sh` | 在本地执行：固定上游 pin、push、只读巡检、ff-only 同步、bundle 同步、启动、拉回报告 |
| `configs/kelmarsh_backbone_information_protocol_v25.json` | 冻结的可执行协议 |
| `configs/tsmixer_upstream_pin_v25.json` | 上游 commit、文件和 LICENSE 的 SHA-256（待固定） |
| `tests/test_tsmixer_basic_small.py`、`tests/test_kelmarsh_backbone_information.py`、`tests/test_kelmarsh_backbone_resume.py` | 契约与恢复测试 |

## 3. 模型与移植差异

每个 block 的结构是：`LN(24,7)` → 时间维 `Linear(24,24)`+ReLU → Dropout → 残差；再接 `LN(24,7)` → 特征维 `Linear(7,32)`+ReLU → Dropout → `Linear(32,7)` → Dropout → 残差。共两个 block，之后取 power 槽，经 `Linear(24,4)` 输出 `[B,4,1]`。LayerNorm 的 eps 为 1e-3、affine 打开，只在单个样本的 (时间, 通道) 平面内归一化。Linear 用 Xavier-uniform 初始化、bias 置 0；LayerNorm 初始化为 1/0。

参数核算：每个 block 为 672+600+487=1759，两个 block 加输出头 100，合计 **3,618**。构造时如果参数量不等于 3,618 会直接报错，不会通过“放大网络”来修正。

与上游 Keras 实现的差异：

- 框架换成 PyTorch。
- 不用 RevIN（上游为可选项）。
- 配置固定为 2 个 block、ff 32、dropout 0.1，不做搜索。
- 只对 power 槽做输出投影。
- 增加 `power_only` 消融掩码：在任何计算之前把标准化后的通道 0–5 置为 0。用的是 `zeros_like`，不会读取被遮蔽的值，所以这些值即使是 NaN/inf 也影响不到输出。

测试用一份独立的 numpy 前向实现来核对时间轴和特征轴的语义。

Full 和 PowerOnly 的 state_dict 逐字节相同（掩码模式是普通属性，不是 buffer）。训练时会用张量摘要验证 PowerOnly 的初始状态等于 Full 的 `init.pt`。

## 4. 数据契约

- 内部队列沿用 v20：fit 为目标结束于 `calibration_early_end`（outer-train 的 90%）之前的窗口，selection 为目标完全落在 `[calibration_early_end, train_end)` 内的窗口。A0 会把窗口哈希与 v20 DLinear contract 和本协议设计值三方比对。
- scaler 必须与 v20 contract 的数组完全相等，且不重新拟合。
- A0–A2 只请求 fit/selection。每次请求都会写入 `events.jsonl`。outer validation 只有在 A3 才可请求，前提是已注册的 G2 通过。封存 test 在任何阶段都会被拒绝。
- 不插值、不裁剪、不改标签。PowerOnly 与 Full 共用同一个窗口对象，所以掩码不会放宽缺失过滤。

## 5. 训练配方（与 v20 DLinear 相同）

AdamW（lr 1e-3，wd 1e-4），batch 64，最多 20 epoch，patience 5（在每个 epoch 开始时检查），梯度裁剪 1.0，CosineAnnealing（T_max 20，eta_min 0，每个 epoch 结束后 step），损失为标准化功率上的 `0.8|e|+0.2e²`，FP32，不用 AMP/TF32，num_workers 0，shuffle 生成器用 `seed+epoch` 播种。

epoch 选择只看内部原尺度 MAE：从 epoch 0 开始，严格更低才更新，打平时保留更早的 epoch。选择后不 refit。

`tests/test_kelmarsh_backbone_information.py::test_fit_loop_reproduces_v20_train_stage_for_dlinear` 要求 v25 的训练循环与 v20 的 `shared.train_stage` 在 DLinear 上逐位一致（同一批合成数据、CPU）。

## 6. 阶段与门槛

- **A0**：校验 v20 协议、completed 清单、全部嵌套哈希和数据哈希；逐 fold 核对 checkpoint SHA、best epoch、窗口哈希、scaler、注册的内部 MAE/RMSE（精确相等）和参数量（200），然后在 CPU 上只读重放 selection（容差 1e-3 kW）。不做任何训练。
- **A1**：顺序为 f0 Full → f0 PowerOnly → f1 … → f2，共 6 次 fit。之后计算 G1 和 I1。
- **A2**：仅在 G1 通过时运行。seed 2025、2026 各跑三折的 Full、PowerOnly、DLinear，最多 18 次 fit。DLinear 先查找完全同契约的缓存，有就复用，没有才训练。之后计算 G2 和 I2。
- **A3**：仅在 G2 通过时运行。先写 `outer_freeze.json`，再一次性评估全部 fold/seed/arm。seed2024 的 DLinear 直接使用 v20 已注册的 outer 预测（校验哈希）。
- **A4**：从原始预测独立重算所有指标（与注册值差超过 1e-6 就报错），计算统计、作图、写报告，然后停止。

门槛的判定规则：

- 改善的定义是 `参照误差 − 候选误差 > 1e-6 kW`，差值在 1e-6 以内算作打平。
- 记录缺失一律判失败，不会当作通过。
- G2 同时要求三个条件：跨 seed 平均后的宏 MAE/RMSE 优于 DLinear；至少两个 fold 同时改善；至少两个 seed 同时改善。这样单个 seed 的大幅改善不能掩盖其他 seed 上的损害。
- I1/I2 只做记录，不改变主候选，也不授权用 PowerOnly 作为 outer 上的备选。
- 门槛是预算分配规则，不是显著性检验。

预算上限为 24 次新 fit、480 个 epoch。

## 7. 统计

- 主配对是 Full vs DLinear，次配对是 Full vs Persistence 和 Full vs PowerOnly。
- 多个 seed 时，在同一窗口上先对损失求平均，再做配对检验。这不是 ensemble，也不会把 seed×窗口当作独立样本。
- 方法：window-uniform；以 issue time 为簇（同一时刻的 6 台风机合为一簇）；移动块长度为 8 个发出时刻，重复 5000 次，种子 2024；Newey–West lag 7。这些实现直接复用 `utils.paired_forecast_statistics` 的原语，测试保证单 seed 时结果与库函数逐项一致。
- RMSE 差的区间单独计算：按簇重抽样平方误差之和，每次重抽样对两个臂分别开根再相减。它不是 MAE 区间的复用。MSE 差的 HAC 检验单位是 kW²。
- 主 p 值族为三个 fold 的 Full vs DLinear 绝对损失检验，做 Holm 校正，原始 p 值一并保留。
- 报告同时给出时间缺口数和覆盖天数。

## 8. 产物、指纹与恢复

运行目录：`outputs/logs/Kelmarsh/<date>/backbone_information_v25_f0-1-2_XXXXXX/`，结构与设计第 10 节一致。

- 根目录：`protocol.json`（含 pin）、`source_manifest.json`、`environment.json`、`data_manifest.json`、`reference_manifest.json`、`parameter_changes.json`、`events.jsonl`（只追加）、`experiment_log.md`、`a0_result.json`、`inner_gate_*.json`、`outer_freeze.json`、`outer_index.json`、`training_completed.json`。
- 每个 arm 的子目录：`init.pt`、`best.pt`、`last.pt`、`training_history.json`、`selection_predictions.npz`、`trained.json`，以及 A3 之后才有的 `outer_predictions.npz`、`result.json`、`completed.json`。
- 报告产物：`metrics_all_runs.csv`、`metrics_by_lead.csv`、`paired_statistics.json`、`independent_recomputation.json`、`resources.csv`、`result.json`、`report.md`、`figures/*.png`、`report_manifest.json`、`completed.json`。

训练指纹与报告指纹分离：

- 每个 fit 的 contract 包含：`training_code_sha256`（只覆盖模型、数据、指标文件和训练函数的源码）、协议、pin、数据、窗口、scaler、配方、设备类型和确定性设置。
- 改报告、改门槛代码或改编排代码都不会使已注册的 fit 失效。但只要代码有任何改动，就必须重新跑 preflight 才能继续新的 fit。
- 已注册的门槛判定如果会因代码改动而改变，控制器会拒绝继续，等人工审查。

`last.pt` 在每个 epoch 边界原子写入，包含模型、optimizer、scheduler、早停计数、历史，以及 Python/NumPy/torch CPU/CUDA 的 RNG 状态。半个 epoch 的工作会被丢弃，并记入事件日志。测试要求中断后恢复的结果与不中断的结果逐位一致。

报告失败时状态记为 `REPORT_FAILED`，训练状态仍为 `TRAIN_COMPLETED`；之后用 `--report-only` 重跑报告，不会重新训练。

## 9. 资源策略

- 自身只用 2 个 CPU 线程，DataLoader 为 0 个 worker。
- 每个 fold/seed 配对在开始前决定设备：GPU 空闲显存不少于 6 GiB 才用 GPU；有其他进程时把自身 allocator 上限设为 4 GiB，独占时为 90%；否则用 CPU。同一配对内的所有 arm 使用同一设备。
- 恢复时如果注册的设备暂时不可用，就等待（最长 6 小时），不会悄悄换设备。
- allocator 上限不能隔离显存，也不保证算力份额。
- 不结束、不暂停、不 renice 任何其他进程，不改整卡、MPS 或网络设置。

## 10. 部署与运行命令（在本地 Git Bash 执行）

```bash
bash scripts/deploy/v25_server_ops.sh resolve-pin      # 固定上游 commit/SHA；对照 output/kelmarsh_hourly/tsmixer_upstream/ 检查移植
git add configs/tsmixer_upstream_pin_v25.json && git commit -m "Pin upstream TSMixer source for v25"
bash scripts/deploy/v25_server_ops.sh push
bash scripts/deploy/v25_server_ops.sh inspect          # 实时核对 HEAD、tracked 改动（两份 docs）、v25 指针、GPU、进程
bash scripts/deploy/v25_server_ops.sh sync             # 只做 ff；若 GitHub 不通改用 bundle-sync
bash scripts/deploy/v25_server_ops.sh preflight        # 后台运行：v25 测试 + 旧回归测试 + A0 + 合成 smoke
bash scripts/deploy/v25_server_ops.sh preflight-status # 必须 passed=true
bash scripts/deploy/v25_server_ops.sh launch           # 后台启动 A0→A4
bash scripts/deploy/v25_server_ops.sh status           # 每次等待不超过 30 秒；超时就重连查状态，不要重复启动
bash scripts/deploy/v25_server_ops.sh resume           # 中断后恢复（已完成的 fit 不重跑）
bash scripts/deploy/v25_server_ops.sh fetch            # 拉回报告和图（不含权重）
```

服务器侧也可以直接运行：`KELMARSH_PYTHON=/home/XT/miniconda3/envs/timedart/bin/python bash scripts/train/Kelmarsh_launch_backbone_information.sh --status`。

## 11. 已知局限

- torch 相关代码在沙箱中**没有执行过**，第一次执行是在服务器 preflight 中。如果出错，属于实现错误，只做最小修复。修复后训练代码哈希会变化；已完成的真实 fit 在 contract 不一致时会被拒绝复用，而不会被静默覆盖。
- 三个 fold 和三个 seed 都来自同一风场、同一年，seed 差异只反映优化随机性。outer 已在历史版本中被看过，即使显著也不能称为独立确认。
- PowerOnly 消融只能说明六个辅助通道作为一个整体是否有用，不能推断单个通道的因果贡献。

## 12. 下一步（只写建议，本轮不实施）

- 对公开 Kelmarsh 多年份数据做字段可用性和曝光台账审查，然后另立一个冻结的跨年确认协议。
- 单独设立信息预算实验：发行时刻可得的 NWP、合法的小时内末值/波动摘要、同步的上游机组历史。
- 新的 Wiki 证书必须绑定实际的底座、专家、记忆、scaler、事件定义和校准队列；旧证书不能迁移到新底座。

如果 TSMixer 没有稳定胜过 DLinear，就保留 DLinear，把预算转向跨年证据和新增信息。不应因为模型名称或创新叙事而保留没有收益的复杂度。

上游参考：[TSMixer 论文](https://arxiv.org/abs/2303.06053)、[官方代码](https://github.com/google-research/google-research/tree/master/tsmixer)。
