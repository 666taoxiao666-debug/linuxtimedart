# Claude/GPT 单文件交接：v28 多年份底座基准，2026-10-26 审核更新

这是**当前唯一需要读的交接文件**：用户希望只维护一份交接，之后每轮在本文件上更新，不再另开新文档。本文件自包含，附录是全部新增文件的**完整源码**，由脚本从实际文件重新生成，并已逐个核对与工作树一致。

- 仓库：`D:/Users/XT/Desktop/moxing/gpt/v2/TimeDARTFirst`，分支 `codex/self-evolving-wiki`，HEAD `6454304`（含 GPT 的 `4acaf79` 数据修复）。
- 最新模型/实验：完整协议 ID **`kelmarsh_multiyear_base_benchmark_v28`**（不要简写成 v28）。审核前 SHA-256（LF）为 `2a488b44734b0ad193388c5c7a4b513a300a60df48cb2e407f55f258d0f68cc0`；训练前修订后重新冻结为 **`86f7a8df11e42ceb041e08a381e739166b4c88c4038eacbfe05c48963292df3d`**。
- 状态：审核修正已写入这 7 个新增代码/协议文件；尚无本轮训练结果。服务器已开始独立目录的数据准备。旧预处理、v25–v27 和旧结果均未修改。

## 本轮 GPT 审查记录（优先于下文原始方案说明）

审查结论：原稿不宜直接训练；以下修正均发生在真实 fit 之前，保持 18 fits、原模型结构/参数量、损失、学习率、seed 和年份不变。

| 级别 | 问题 | 修正与边界 |
|---|---|---|
| P1 | C4 同时改变训练队列和选 epoch 的验证队列，无法单独解释训练筛选收益 | 所有模型统一在 2020 normal 队列选择；DLinear_alltrain 仅训练队列不同 |
| P1 | 原 D0 将 test 交给数据类，虽未调用 windows(test)，已经计算测试窗口和工况数量 | S1 前只声明 train/validation，并裁切年份边界；S1 后才构建 test 视图，逐一检查训练/验证队列、标签及 scaler 没有改变。文件校验和及数据准备仍会读取原始文件，这不等于预测评估 |
| P1 | Holm 校正有输出但不参与正结论判定 | 支持判据同时要求原 bootstrap 条件及对应 Holm p≤0.05；不选择有利检验 |
| P2 | 报告只复核部分指标，可能漏掉 MSE、区间上界、校正 p 等错误 | 递归重算核对全部指标、统计和判据，要求每份预测进入哈希清单 |
| P2 | 缺真实断点等价测试、已完成报告完整性检查、逐预测恢复登记 | 增加 TSMixer 中断恢复与完整训练参数/结果一致性测试；完成入口校验全清单；逐份测试预测绑定 checkpoint/cohort 哈希复用 |
| P2 | 缺 Skill、容量容差命中率和部分逐小时指标/预测图 | 增加两种 Persistence 的 MAE/RMSE Skill、±5/10%容量命中率、逐小时 MSE/RMSE；预测图固定最小风机首个连续时段，不能按误差挑图 |
| 解释限制 | 7→9 通道同时增加 LinearFlat/TSMixer 参数量；C3 也改变结构与优化几何 | 保留预算，不增设消融；只称输入方案/架构对照，不能宣称纯信息贡献或纯容量因果效应 |

参数复核：DLinear 200；LinearFlat base/last10 676/868；TSMixer base/last10 3618/4262。所有模型直接预测未来 4 个小时，输入 24 小时；没有 Wiki、残差专家或新物理阈值。normal 使用目标期间状态，是事后条件队列；必须同时报告 all，不能将 normal 精度冒充全样本部署精度。

代码提交 `26f0ad3` 已推送并在服务器安全快进。服务器完整 **48 项测试通过**（日志 `outputs/logs/Kelmarsh/multiyear_benchmark_preflight/review_26f0ad3_tests.log`），包含断点等价和完整合成控制器；耗时 41.753 秒（不含冷启动库加载）。本地 Python/Bash 语法通过；Windows 对 11 项纯指标/协议/报告测试使用仅用于导入的空 `fcntl` 模块运行通过，该本地结果不覆盖锁或训练，服务器真实 Linux 测试才覆盖这些功能。

服务器原 `docs/EXPERIMENT_REPLAY.md` 和 `docs/KELMARSH_MODEL_REVIEW.md` 改动保留。其他用户 GPU 任务不做任何管理操作；本任务遵守共享时 4 GiB 分配器上限和 CPU 2 线程。

数据准备 PID `896486`，日志 `outputs/launch_logs/multiyear_v28_prepare_20261026_101813.log`。只准备 2016–2021，2022–2024 不读。2016–2019 将与已登记开发 CSV 逐列核验，功率曲线只拟合 2016–2019。若预检失败，只修实现问题并重新预检，不启动 fit。

## 0. 请 GPT 做什么（测试 + 训练）

1. **审查**：按第 6 节审查附录源码，有问题直接修，并写明改了哪里。协议若需改动，必须在任何真实 fit 之前改并重新冻结，同时更新控制器里的 SHA 常量。
2. **提交与同步**：只提交第 3 节的 7 个文件和本交接文件（见 0.1，不要提交已过时的文档）。推送后让服务器安全快进。
3. **测试**：在服务器运行 v28 测试和回归测试（第 7 节）。需要 torch 的 3 项测试只能在服务器上跑。
4. **数据**：按第 7 节第 1–2 步，把已校验的原始 ZIP 复制到新根目录，准备 2016–2021。
5. **预检与训练**：预检通过、用户授权后启动：18 个 fit → S1 冻结 → 2021 年一次评估 → 判据 → 报告。
6. **回传**：结束时把结果、改动、证据位置**合并成一份**单文件交接给 Claude（第 8 节列出了需要的内容）。

## 0.1 本轮本地仓库的变化（GPT 需要知道）

**本地清理（2026-10-26，用户授权）。** 只删除了“别处已有完整副本”的未跟踪文件，约 108 MB。**没有删除任何 git 跟踪文件**，测试照常通过。

| 删除项 | 判定依据 |
|---|---|
| `backups/TimeDARTFirst_pre_llm_wiki_20260911_202d9f6.zip`（41 MB） | 提交 202d9f6 是 HEAD 的祖先；339 个成员对应的 blob 全部在 git 对象库中 |
| `output/kelmarsh_hourly/v25_review_source_e9f0c39.tar`（45 MB） | 全部成员在 git 对象库中 |
| `output/kelmarsh_hourly/v25_claude_audit_package_20261025.zip` | 94/94 个成员与 `v25_claude_audit_package_20261025/`、`backbone_information_v25_review/` 逐字节相同 |
| `output/kelmarsh_hourly/CLAUDE_V26_COMPLETE_HANDOFF_20261025.zip` | 285/289 与 `anchored_residual_v26_review/` 逐字节相同；另 3 个 `source_5d0f904/*` 与 `git show 5d0f904:` 一致；`PACKAGE_MANIFEST.json` 已保存到 `anchored_residual_v26_review/` |
| `output/kelmarsh_hourly/CLAUDE_FORWARD_BLOCK_RESIDUAL_COMPLETE_20261025.zip` | 358/359 与 `forward_block_residual_diagnostic_review/` 逐字节相同；`PACKAGE_MANIFEST.json` 已保存到该文件夹 |
| `output/kelmarsh_hourly/hourly_review_5629269.tar.gz` | 437/437 与 `review_5629269/` 逐字节相同 |
| `output/kelmarsh_hourly/v25_review_fix3.tar.gz` | 全部成员在 git 中 |
| `output/` 下 27 个 `*.bundle` | 每个 bundle 的 head 都是 HEAD 的祖先 |
| 8 个 `.chart-data-*`、10 个 `__pycache__`、`.pytest_cache` | 运行时自动生成的临时文件 |

因此，早先交接中提到的这些 zip 包本地已不存在，其解压后的文件夹（含审计脚本、预测、权重、图）都保留着。**保留未动的**：
- `v25_review_patch.tar.gz`、`v25_review_fix2.tar.gz`、`forward_block_residual_review_patch.tar`：里面有从未提交过的中间版本代码，只此一份；
- `code_payload.tar.gz`：未核查，所以没动；
- `datasets/`（SDWPF 原始数据）和 `outputs/`（早期 SDWPF 实验输出）；
- 所有 review 文件夹和服务器上的一切。

**已过时、不要提交的未跟踪文档：**
- `docs/CLAUDE_TO_GPT_HANDOFF_MULTIYEAR_PREPROCESSING_20261025.md`：附录里是 GPT 修复前的旧源码，**不能拿来覆盖工作树**；
- `docs/KELMARSH_MULTIYEAR_PREPROCESSING.md`：部分描述已被 `4acaf79` 修正，例如默认队列现为 `all`、全年容量门槛已移除。

以本文件和 GPT 的 `6454304` 记录为准。

**后续工作方式（用户偏好）：** 尽量修改和扩展已有文件，不再每轮整份复制控制器和启动器。下一轮先把锁、状态、恢复、预检、训练循环这些通用部分抽成公共模块，启动器也改成一个带参数的脚本。已完成的 v25–v27 文件保持不动，因为它们的哈希已登记。

## 1. 对上轮 GPT 交接的回应

全部接受：
- provider 默认队列改为 `all`；
- 去掉全年容量门槛（因果问题）；
- 曲线年必须属于实际训练年；
- 拒绝覆盖已有准备产物；
- 未核实参考系前不启用偏航误差通道；
- 不能预先保证 R² 上升；
- 数据准备完成不等于训练完成。

我上轮说“网络容量必然不是瓶颈”“已强制每张表同时输出两套队列”属于说过头。这次由控制器和报告实际强制同时输出 normal 和 all（见 4.4）。

## 2. v28 设计（一页）

**为什么是这个划分。** 已登记的开发准备只有 2016–2019，功率曲线也用 2016–2019 拟合，而 provider 要求曲线年 ⊂ train。所以只能 train = 2016–2019，验证和测试必须是 2020 及以后的年份。v28 冻结如下：
- train：2016–2019
- validation（选 epoch，并选出下一轮的起点配置）：2020
- test：2021，只评估一次
- 封存：2022–2024，v28 中不准备也不读取

2020 和 2021 的准备放在**新根目录** `datasets/wind_public_v28`，因为已有的准备产物不允许被覆盖。D0 会逐行核验新准备里 2016–2019 的每一行与已登记的开发 CSV（`50134f7b…`）完全一致，功率曲线也完全一致。

**公共窗口。** 只建一个序列，包含所有臂所需的通道并集（6 个基础通道 + `power_last10` + `wind_last10` + power）。所有臂用完全相同的窗口，输入较少的臂只是丢掉一部分通道（`ChannelView`）。两种 Persistence 也在同一批窗口上计算。

**臂（共 18 个 fit = 6 种配置 × 3 个 seed）：**

| 臂 | 家族 | 输入 | 训练与选 epoch 的队列 | 参数 |
|---|---|---|---|---:|
| PersistenceHour | 上一小时均值 | — | — | 0 |
| PersistenceLast10 | 最后 10 分钟功率 | — | — | 0 |
| DLinear_normal | DLinear | base（只有功率进入输出） | normal | 200 |
| DLinear_alltrain | DLinear | base | **all**（训练队列对照） | 200 |
| LinearFlat_base | 展平线性 | base 7 通道 | normal | 676 |
| LinearFlat_last10 | 展平线性 | base + 小时末 2 通道 | normal | 868 |
| TSMixer_base | TSMixerBasicSmall | base | normal | 3618 |
| TSMixer_last10 | TSMixerBasicSmall | base + 小时末 2 通道 | normal | 4262 |

**训练配方：** 与 v20/v25 相同。AdamW，lr 1e-3，wd 1e-4，batch 64，最多 20 epoch，patience 5，clip 1，cosine T_max 20，标准化功率上的 0.8 MAE + 0.2 MSE，按验证集原尺度 MAE 选 epoch（含 epoch0，严格更小才更新），不 refit。评估 batch 512。三个 seed：2024、2025、2026。

**阶段：**
1. D0：数据身份核验。
2. T1：18 个 fit。
3. S1：选择冻结。写入每个选中状态的哈希、验证集指标，以及验证集上选出的配置。此后才允许创建任何 test 窗口；代码里有硬拦截，在此之前创建 test 窗口会直接报错。
4. E1：一次性在 2021 上预测所有臂。
5. 判据与报告。

**预声明比较**（normal 和 all 两个队列各算一遍；每个队列内对 C1–C5 的 MAE Newey–West p 值做 Holm 校正）：

| 编号 | 候选 vs 参照 | 回答的问题 |
|---|---|---|
| C1 | LinearFlat_last10 vs LinearFlat_base | 小时末信息（线性模型） |
| C2 | TSMixer_last10 vs TSMixer_base | 小时末信息（非线性模型） |
| C3 | TSMixer_base vs LinearFlat_base | 有了 4 年数据后，非线性容量是否有用 |
| C4 | DLinear_normal vs DLinear_alltrain | 训练集剔除异常是否有用 |
| C5 | 验证集选出的配置 vs PersistenceLast10 | 是否打败更强的朴素基线 |

**判据（预声明）：**
- `information_supported`：C1 或 C2 在 test normal 上 MAE 增益区间下界 > 0 且 RMSE 增益 > 0，并且同一比较在 test all 上 MAE 增益 ≥ 0。
- `capacity_supported`：C3 满足同样条件。
- `training_cleaning_supported`：C4 在 normal 上区间下界 > 0，且在 all 上增益 ≥ 0。
- `beats_naive`：C5 在 normal 和 all 上区间下界都 > 0。

**下一轮：** 从验证集选出的配置出发，最多再加一组信息，用 2022 年做确认（由下一轮自己的冻结协议准备）。2021 不再用作确认年；2023–2024 继续封存。任何臂、seed、epoch、阈值、通道都不能依据 test 结果来选。

**不能声称：** normal 队列的结果只是条件精度；部署精度看 all 队列。开发年份 2016–2019 已在早先协议中被查看过。三个 seed 共用同一批天气窗口，不是独立样本。这些模型都不是创新点。

## 3. 新增文件

| 文件 | 行数 | SHA-256（LF，前 16 位） | 作用 |
|---|---:|---|---|
| `configs/kelmarsh_multiyear_base_benchmark_protocol_v28.json` | 119 | `2a488b44734b0ad1` | 冻结协议 |
| `models/LinearFlat.py` | 22 | `b81c347fd2aea5c3` | 展平线性：`Linear(24·C → 4)` |
| `utils/kelmarsh_multiyear_benchmark.py` | 240 | `cd577677d8bb5417` | 不依赖 torch：协议校验、参数公式、D0 身份核验、指标、配对比较、Holm、验证集选择、判据、`evaluate_registered` |
| `scripts/train_kelmarsh_multiyear_benchmark.py` | 745 | `2fd952608e77a15c` | 控制器：D0/T1/S1/E1、通用训练器、断点恢复、锁、指针、预检 |
| `scripts/report_kelmarsh_multiyear_benchmark.py` | 167 | `937efb6a4813e8b0` | 不依赖 torch 的报告：从登记数组独立重算，与登记判据不一致即拒绝 |
| `scripts/train/Kelmarsh_launch_multiyear_benchmark.sh` | 89 | `2bd0aa347a75f4d8` | 启动器（由 v26 启动器改名，LF） |
| `tests/test_kelmarsh_multiyear_benchmark.py` | 295 | `f5bed3e984a358f7` | 测试 |

## 4. 关键实现说明

1. **训练器 `fit_config`** 沿用 v25 `fit_arm` 的顺序与语义（seed_all → 构造 → epoch0 评估 → seed+epoch 打乱 → clip/AdamW → 严格 MAE 最优 → epoch 开头检查 patience → 每个 epoch 后 cosine 步进），模型由工厂按家族构造。采纳了 v26 审过的恢复修复：last.pt 同时保存选中状态，恢复时据此重写 best.pt。数值工具（predict、make_loader、state_digest、atomic_save、RNG 保存/恢复、设备探测）直接调用已审计的 v26 模块。
2. **参数量**：按公式计算，构造时逐个核对，不一致就拒绝。TSMixer 每块 = 161·C + 632，因此 7 通道时为 3618（与 v25 登记一致），9 通道时为 4262。
3. **D0** 依次核对：v28 准备元数据（schema 2、年份、曲线年、CSV 哈希、标签未改）；开发 CSV 与其元数据的哈希；新准备中 2016–2019 的每一行、每一列与开发 CSV 完全一致（浮点按 NaN 相等比较）；功率曲线 JSON 完全一致；序列通道与协议一致。
4. **两个队列都强制输出**：`evaluate_registered`（不依赖 torch，控制器和报告共用）对每个臂分别计算 normal 和 all 的指标，并在两个队列上分别做 C1–C5。报告的每张表都有 cohort 列。
5. **报告**从登记的 NPZ 重新计算全部指标、比较和判据；与 `decision.json` 不一致（容差 1e-9）就拒绝。登记文件的哈希也会核对。
6. **设备**：每次运行只决定一次（`device.json`），18 个 fit 用同一设备。资源策略复用 v26：共享 GPU 时本进程 allocator 上限 4 GiB；空闲时最多用 90%；剩余显存不足 6 GiB 时退回 CPU。

## 5. 已知风险与限制（请 GPT 重点看）

1. **复制的原始文件**：新根目录需要能找到已校验的 ZIP 和缓存的 Zenodo 元数据，否则会重新下载（仍会按官方 MD5 校验，不影响正确性）。第 7 节给出复制命令。
2. **2019 年末边界**：如果 2020 的 ZIP 含有 2019-12-31 的 tick，可能改变 2020-01-01 00:00 这个标签行；D0 会因行不一致而拒绝。若发生，需要先查清原因，不能直接放宽核验。
3. **训练量**：每个 normal 训练集约 5.6 万个窗口 × 最多 20 epoch，TSMixer 在 CPU 上会较慢。
4. **比较的含义**：C4 比较的是“训练集是否剔除异常”，两个模型在同一个 test 队列上比，属于训练数据的消融。
5. **LinearFlat** 是标准的展平线性基线（与 NLinear/Linear 类似），不是创新。
6. **偏航误差通道**：按 GPT 要求禁用，协议校验也会拒绝它。
7. **TSMixer 回退**：v25 中 TSMixer 输给 DLinear。这次数据量约为当时的 8 倍，C3 就是要回答“数据多了以后容量是否有用”；结果仍可能是否定的。

## 6. 请 GPT 审查的清单

- 协议的划分、封存年份、比较族与判据是否合理；判据只在训练前可改。
- `fit_config` 与 v25 `fit_arm` 的语义一致性；断点恢复；best/last 的一致性。
- `ChannelView` 保证功率在最后一个通道、scaler 取对应子集；DLinear 只对功率求损失。
- S1 冻结前不能创建 test 窗口（`V28Run.windows` 的硬拦截）；事件顺序有测试覆盖。
- `verify_development_rows` 的比较口径（字符串列、布尔列、浮点 NaN）。
- `evaluate_registered`：配对统计按 issue time 聚类、多 seed 先平均损失、各队列分别做 Holm。
- 报告的重算和篡改拒绝。

## 7. 验证情况与执行步骤

**Claude 沙箱（无 torch）已运行：**
- `python -m unittest tests.test_kelmarsh_multiyear_benchmark`：9 项通过，3 项因缺 torch 跳过。通过的覆盖：协议哈希与设计、参数公式、各种偏离被拒绝、准备身份、开发行逐行一致（1e-9 的改动也会被检出）、缺行拒绝、曲线不一致拒绝、两个队列的指标与 C1–C5 及判据各分支、验证集选择与平局规则、指标定义、报告重算与篡改拒绝。
- 回归：`tests.test_kelmarsh_multiyear_preprocessing` 与 `tests.test_hourly_wind_acquisition` 共 23 项通过（1 项需要 torch 而跳过）。

**需在服务器运行（含 torch）：** 模型、ChannelView、端到端控制器（合成数据，检查 S1 先于任何 test 窗口、6 个 fit、两个队列、重跑时复用）、预检拦截、test 窗口拦截。

**执行步骤（仓库根目录）：**

```bash
# 1) 把已校验的原始文件复制到新根目录（prepare 仍会逐个按官方 MD5 校验）
OLD=datasets/wind_public/resampled_hourly_scada
NEW=datasets/wind_public_v28/resampled_hourly_scada
mkdir -p $NEW/kelmarsh_2016/raw $NEW/kelmarsh_2016/metadata $NEW/kelmarsh_multiyear/raw
cp -p $OLD/kelmarsh_2016/raw/Kelmarsh_SCADA_2016_3082.zip $OLD/kelmarsh_2016/raw/Kelmarsh_WT_static.csv $NEW/kelmarsh_2016/raw/
cp -p $OLD/kelmarsh_2016/metadata/zenodo_16807551.json $NEW/kelmarsh_2016/metadata/
cp -p $OLD/kelmarsh_multiyear/raw/Kelmarsh_SCADA_2017_3083.zip $OLD/kelmarsh_multiyear/raw/Kelmarsh_SCADA_2018_3084.zip \
      $OLD/kelmarsh_multiyear/raw/Kelmarsh_SCADA_2019_3085.zip $NEW/kelmarsh_multiyear/raw/

# 2) 准备 2016-2021（会下载 2020、2021；2022-2024 不碰）
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -u scripts/prepare_kelmarsh_multiyear.py --root datasets/wind_public_v28 \
  --years 2016 2017 2018 2019 2020 2021 --curve-years 2016 2017 2018 2019 --download

# 3) 预检：测试 + 真实数据 D0（逐行核验 2016-2019）+ 合成冒烟 + 命名唯一 + 启动器语法
bash scripts/train/Kelmarsh_launch_multiyear_benchmark.sh --preflight-only

# 4) 用户授权后启动（18 fit -> S1 冻结 -> 2021 一次评估 -> 判据 -> 报告）
bash scripts/train/Kelmarsh_launch_multiyear_benchmark.sh --detach
bash scripts/train/Kelmarsh_launch_multiyear_benchmark.sh --status
# 恢复（已完成的 fit 不重训）/ 只重做报告
bash scripts/train/Kelmarsh_launch_multiyear_benchmark.sh --resume --detach
bash scripts/train/Kelmarsh_launch_multiyear_benchmark.sh --report-only
```

结果目录：`outputs/logs/Kelmarsh/<日期>/multiyear_benchmark_v28_XXXXXX/`。关键产物：`data_manifest.json`、`selection_freeze.json`、`decision.json`、`test_metrics.csv`、`test_comparisons.csv`、`report.md`、`figures/`。

## 8. 结束后请交回 Claude 的内容（合并成一份交接）

- 实际提交、协议 SHA、预检结果；
- D0 核验结果（2016–2019 行一致、曲线一致、各划分在两个队列下的窗口数）；
- 18 个 fit 的选中 epoch 与验证集 MAE；验证集选出的配置；
- 2021 年两个队列下每个臂的 MAE/RMSE/R²（含逐步长）；C1–C5 的增益、区间、Holm p；四条判据的结果；
- 任何偏离、失败或修复。

---

# 附录：新增文件完整源码

## A1. `configs/kelmarsh_multiyear_base_benchmark_protocol_v28.json`

```json
{
  "protocol_id": "kelmarsh_multiyear_base_benchmark_v28",
  "status": "FROZEN_EXECUTABLE",
  "launch_authorization": "Implemented and unit-tested (torch-free part) only; nothing prepared, fitted or evaluated by the author. Requires the user's decision, the D0 preparation below and a passed --preflight-only.",
  "created_date": "2026-10-26",
  "design_source": "GPT handoff 2026-10-26 section 7 (multi-year protocol, controller, fixed baselines, one information group per comparison, all+normal reporting); user decisions 2026-10-25 (add years; normal-operation training and primary evaluation)",
  "scope": "first multi-year numeric base benchmark; no Wiki; no novelty claim",
  "questions": [
    "Q1 information: does the end-of-hour state (power_last10, wind_last10) improve the same architecture?",
    "Q2 capacity: with four training years, does a small nonlinear mixer beat a flattened linear map on the same inputs?",
    "Q3 training cohort: does training DLinear on normal-operation windows instead of all windows help, on each test cohort?",
    "Q4 naive: does the validation-chosen trained configuration beat the last-10-minute Persistence?"
  ],
  "novelty_claim": "none; DLinear, a flattened linear map and TSMixer-basic are published baselines",
  "data": {
    "source": "Zenodo 16807551, Kelmarsh six Senvion MM92, CC BY 4.0",
    "development_preparation": {
      "csv": "datasets/wind_public/resampled_hourly_scada/kelmarsh_multiyear/hourly/kelmarsh_multiyear_hourly_enriched.csv",
      "csv_sha256": "50134f7b6c23c9608d7ef7926ff67692463e9846fa7204a9515b2b7d473dfd0e",
      "preparation_json_sha256": "ff4f6ebd9ed3dd59016295a81bb9b614cbb77dbfe6d3355a3efa170c1ff09aa2",
      "years": [2016, 2017, 2018, 2019],
      "use": "identity reference only: every 2016-2019 row of the v28 preparation must equal this registered CSV"
    },
    "v28_preparation": {
      "root": "datasets/wind_public_v28",
      "csv": "datasets/wind_public_v28/resampled_hourly_scada/kelmarsh_multiyear/hourly/kelmarsh_multiyear_hourly_enriched.csv",
      "years": [2016, 2017, 2018, 2019, 2020, 2021],
      "curve_years": [2016, 2017, 2018, 2019],
      "schema_version": 2,
      "command": "python scripts/prepare_kelmarsh_multiyear.py --root datasets/wind_public_v28 --years 2016 2017 2018 2019 2020 2021 --curve-years 2016 2017 2018 2019 --download",
      "rule": "prepared only after this protocol is frozen; 2022-2024 are never prepared or read in v28"
    },
    "start_utc": "2016-05-01T00:00:00Z",
    "rated_power_kw": 2050.0,
    "input_len": 24,
    "pred_len": 4,
    "stride": 3,
    "extras_in_common_series": ["power_last10", "wind_last10"],
    "common_window_rule": "ONE series is built with the union of all channels; every arm uses exactly these windows (finite history in all nine channels, finite four power targets); smaller-input arms only drop channels",
    "channels": ["Wspd", "Etmp", "Ndir_sin", "Ndir_cos", "Pab_mean", "rotor_speed", "power_last10", "wind_last10", "power"],
    "channel_sets": {"base": [0, 1, 2, 3, 4, 5, 8], "last10": [0, 1, 2, 3, 4, 5, 6, 7, 8]},
    "splits": {"train": [2016, 2017, 2018, 2019], "validation": [2020], "test": [2021]},
    "sealed_years": [2022, 2023, 2024],
    "scaler": "provider rule: mean/std of all training-year rows (ddof 0, NaN ignored); not refitted per cohort",
    "labels_modified": false,
    "yaw_error_channels_enabled": false,
    "yaw_error_reason": "wind-direction reference frame not verified (GPT 2026-10-26)"
  },
  "cohorts": {
    "normal": "all four target hours are normal-operation hours (ex-post, uses the target period's operating state; conditional accuracy only)",
    "all": "every common window with original labels (deployment accuracy)",
    "training_cohort_policy": "every arm trains on normal windows of the train years and selects its epoch on normal windows of the validation year, except the declared control DLinear_alltrain which trains and selects on all windows",
    "reporting": "every metric table reports BOTH test cohorts; normal is primary by user decision, all is mandatory secondary; cohort sizes and normal share are reported"
  },
  "arms": {
    "PersistenceHour": {"family": "Persistence", "anchor": "last hourly mean power", "fit": false},
    "PersistenceLast10": {"family": "Persistence", "anchor": "last observed 10-minute power at issue", "fit": false},
    "DLinear_normal": {"family": "DLinear", "channels": "base", "train_cohort": "normal"},
    "DLinear_alltrain": {"family": "DLinear", "channels": "base", "train_cohort": "all"},
    "LinearFlat_base": {"family": "LinearFlat", "channels": "base", "train_cohort": "normal"},
    "LinearFlat_last10": {"family": "LinearFlat", "channels": "last10", "train_cohort": "normal"},
    "TSMixer_base": {"family": "TSMixerBasicSmall", "channels": "base", "train_cohort": "normal"},
    "TSMixer_last10": {"family": "TSMixerBasicSmall", "channels": "last10", "train_cohort": "normal"}
  },
  "families": {
    "DLinear": {"factory": "models.DLinear.Model", "args": {"task_name": "finetune", "downstream_task": "forecast", "individual": 0, "moving_avg": 25},
                "parameters": {"base": 200}},
    "LinearFlat": {"factory": "models.LinearFlat.Model", "parameters": {"base": 676, "last10": 868}},
    "TSMixerBasicSmall": {"factory": "models.TSMixerBasicSmall.Model", "args": {"blocks": 2, "ff_dim": 32, "dropout": 0.1, "norm_eps": 0.001},
                          "parameters": {"base": 3618, "last10": 4262}}
  },
  "seeds": [2024, 2025, 2026],
  "training": {
    "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 64, "eval_batch_size": 512,
    "max_epochs": 20, "patience": 5, "grad_clip_norm": 1.0, "scheduler": "CosineAnnealingLR", "scheduler_t_max": 20,
    "scheduler_eta_min": 0.0, "loss_space": "standardized power", "loss_mae_weight": 0.8, "loss_mse_weight": 0.2,
    "loss_leads": "uniform", "selection_cohort": "normal", "selection_metric": "validation original-scale MAE on the same normal cohort for every arm, including DLinear_alltrain",
    "epoch0_included": true, "tie_policy": "strictly lower MAE updates; ties keep the earlier epoch",
    "refit_after_selection": false, "dtype": "float32", "amp": false, "tf32": false, "num_workers": 0,
    "shuffle": "torch.Generator().manual_seed(seed + epoch)", "same_recipe_as": "v20 DLinear / v25 recipe"
  },
  "budget": {"fits": 18, "new_fits_max": 18, "new_epochs_max": 360, "additional_seeds": false, "additional_arms": false,
             "hyperparameter_search": false},
  "stages": [
    {"id": "D0", "name": "data identity", "fits": 0, "test_access": false,
     "checks": "v28 preparation metadata (schema 2, years, curve years, CSV hash); every 2016-2019 row equal to the registered development CSV; identical power curves; common-window counts"},
    {"id": "T1", "name": "fit 6 configurations x 3 seeds", "fits": 18, "test_access": false},
    {"id": "S1", "name": "selection freeze", "fits": 0, "test_access": false,
     "freeze": "every selected state hash and validation metrics, plus the validation-chosen configuration for Q4 and for any next round, written before any test window is instantiated"},
    {"id": "E1", "name": "single test evaluation of every arm on both cohorts", "fits": 0, "test_access": true},
    {"id": "R1", "name": "independent recomputation and report", "fits": 0, "test_access": "registered predictions only"}
  ],
  "comparisons": {
    "C1": {"candidate": "LinearFlat_last10", "reference": "LinearFlat_base", "question": "Q1"},
    "C2": {"candidate": "TSMixer_last10", "reference": "TSMixer_base", "question": "Q1"},
    "C3": {"candidate": "TSMixer_base", "reference": "LinearFlat_base", "question": "Q2"},
    "C4": {"candidate": "DLinear_normal", "reference": "DLinear_alltrain", "question": "Q3"},
    "C5": {"candidate": "VALIDATION_CHOSEN", "reference": "PersistenceLast10", "question": "Q4"}
  },
  "validation_choice": "among LinearFlat_base, LinearFlat_last10, TSMixer_base, TSMixer_last10: lowest seed-mean validation normal MAE (ties: fewer parameters)",
  "statistics": {"cohort": "computed separately on test normal and test all", "seed_handling": "losses averaged over the 3 seeds on identical windows",
                 "cluster": "issue time across turbines", "block_length": 8, "replicates": 5000, "seed": 2024, "hac_max_lag": 7,
                 "multiple_testing": "Holm over the C1-C5 MAE Newey-West p-values within each cohort; raw p retained",
                 "rmse": "issue-time block bootstrap of squared-error sums; separate from the MAE CI"},
  "rules": {
    "information_supported": "C1 or C2 on test normal: MAE gain CI95 lower > 0, Holm p <= 0.05 and RMSE gain > 0, AND the same comparison on test all has MAE gain >= 0; supports the wider input variant, not pure information attribution at equal parameter count",
    "capacity_supported": "C3 on test normal: MAE gain CI95 lower > 0, Holm p <= 0.05 and RMSE gain > 0, AND on test all MAE gain >= 0",
    "training_cleaning_supported": "C4 on test normal: MAE gain CI95 lower > 0 and Holm p <= 0.05, AND on test all MAE gain >= 0; both arms select epochs on the same normal validation cohort",
    "beats_naive": "C5 on test normal AND test all: MAE gain CI95 lower > 0 and Holm p <= 0.05",
    "attribution_limits": "C1/C2 change input width and parameter count. C3 changes architecture, parameter count and optimization geometry, not capacity alone. No Wiki contribution or causal attribution is established by these contrasts.",
    "next_round": "the next protocol starts from the validation-chosen configuration, adds at most one further information group, and confirms on 2022 prepared under its own frozen protocol; 2021 is not reused as a confirmation year; 2023-2024 stay sealed",
    "no_post_hoc": "no arm, seed, epoch, threshold, cohort rule or channel may be chosen from test results"
  },
  "operations": {"cpu_threads": 2, "gpu_free_gib_min_for_new_pair": 6, "shared_gpu_pytorch_allocator_cap_gib": 4,
                 "idle_gpu_allocator_fraction_max": 0.9, "cpu_fallback_before_pair_start": true,
                 "modify_other_users_processes": false, "duplicate_launch_lock": true, "atomic_epoch_checkpoint": true},
  "forbidden": ["preparing or reading 2022-2024", "test access before the S1 freeze", "selection or thresholds from test",
                "label clipping, imputation or modification", "yaw-error channels", "loading old Wiki certificates",
                "modifying v25-v27 or the registered development preparation"]
}
```

## A2. `models/LinearFlat.py`

```python
"""Flattened linear direct forecaster: one Linear(seq_len * channels -> pred_len) on the standardized history.

The simplest model that can combine channels (unlike channel-independent DLinear). Used in v28 as the low-variance
architecture for the end-of-hour information comparison. Output [B, pred_len, 1] = standardized power. PyTorch default
Linear initialisation; no normalisation, no hidden layer, no dropout. Not a contribution.
"""
from __future__ import annotations

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, *, seq_len: int = 24, channels: int = 7, pred_len: int = 4):
        super().__init__()
        self.seq_len, self.channels, self.pred_len = int(seq_len), int(channels), int(pred_len)
        self.linear = nn.Linear(self.seq_len * self.channels, self.pred_len)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or tuple(history.shape[1:]) != (self.seq_len, self.channels):
            raise ValueError(f"Expected [B, {self.seq_len}, {self.channels}] history, got {tuple(history.shape)}")
        return self.linear(history.reshape(history.shape[0], -1)).unsqueeze(-1)
```

## A3. `utils/kelmarsh_multiyear_benchmark.py`

```python
"""Torch-free helpers for v28 (multi-year Kelmarsh base benchmark): protocol checks, data identity, arm configs,
metrics, paired comparisons, validation choice and the predeclared rules. Generic hashing / statistics come from the
v25 module ``utils.kelmarsh_backbone_information`` (its bytes enter the v28 fingerprint)."""
from __future__ import annotations

import numpy as np

from utils.kelmarsh_backbone_information import (  # noqa: F401  (re-exported)
    acquire_exclusive_lock, append_event, canonical_sha256, choose_pair_device, contract_mismatches, file_sha256,
    holm_adjust, paired_loss_statistics, read_json, rmse_difference_bootstrap, safe, text_sha256_lf,
    verify_registered_files, verify_training_completed, window_hash, write_json)

PROTOCOL_ID = "kelmarsh_multiyear_base_benchmark_v28"
PERSISTENCE_ARMS = ("PersistenceHour", "PersistenceLast10")
TRAINED_ARMS = ("DLinear_normal", "DLinear_alltrain", "LinearFlat_base", "LinearFlat_last10", "TSMixer_base", "TSMixer_last10")
CHOICE_CANDIDATES = ("LinearFlat_base", "LinearFlat_last10", "TSMixer_base", "TSMixer_last10")
COHORTS = ("normal", "all")
HOUR = np.timedelta64(1, "h")


# --------------------------------------------------------------------------- protocol
def expected_parameters(family, channels, seq_len=24, pred_len=4, ff_dim=32, blocks=2):
    if family == "DLinear":
        return 2 * (seq_len * pred_len + pred_len)
    if family == "LinearFlat":
        return seq_len * channels * pred_len + pred_len
    if family == "TSMixerBasicSmall":
        block = 2 * (2 * seq_len * channels) + (seq_len * seq_len + seq_len) + (channels * ff_dim + ff_dim) + (ff_dim * channels + channels)
        return blocks * block + seq_len * pred_len + pred_len
    raise ValueError(f"Unknown family {family!r}")


def validate_protocol(p):
    d = p["data"]
    if p["protocol_id"] != PROTOCOL_ID:
        raise ValueError("Unexpected protocol id")
    if tuple(p["arms"]) != PERSISTENCE_ARMS + TRAINED_ARMS:
        raise ValueError("Arm list departs from the frozen design")
    if (d["input_len"], d["pred_len"], d["stride"]) != (24, 4, 3):
        raise ValueError("Window design departs from 24 -> 4, stride 3")
    if d["channels"][-1] != "power" or any(s[-1] != len(d["channels"]) - 1 for s in d["channel_sets"].values()):
        raise ValueError("Power must be the last channel of every channel set")
    if d["channels"][6:-1] != d["extras_in_common_series"]:
        raise ValueError("Common-series extras and channel list disagree")
    splits = d["splits"]
    if not (max(splits["train"]) < min(splits["validation"]) <= max(splits["validation"]) < min(splits["test"])
            < min(d["sealed_years"])):
        raise ValueError("Years must be ordered train < validation < test < sealed")
    if not set(d["v28_preparation"]["curve_years"]) <= set(splits["train"]):
        raise ValueError("Power-curve years must belong to train")
    if set(d["v28_preparation"]["years"]) != set(splits["train"]) | set(splits["validation"]) | set(splits["test"]):
        raise ValueError("Prepared years must be exactly train+validation+test")
    if d.get("yaw_error_channels_enabled") or any(c.startswith("yaw") for c in d["channels"]):
        raise ValueError("Yaw-error channels are disabled in v28")
    for arm in TRAINED_ARMS:
        spec = p["arms"][arm]
        family = p["families"][spec["family"]]
        channels = len(d["channel_sets"][spec["channels"]])
        declared = family["parameters"].get(spec["channels"])
        if declared != expected_parameters(spec["family"], channels):
            raise ValueError(f"{arm}: declared parameters {declared} != formula")
        if spec["train_cohort"] not in COHORTS:
            raise ValueError(f"{arm}: unknown training cohort")
    if p["budget"]["fits"] != len(TRAINED_ARMS) * len(p["seeds"]) or p["budget"]["fits"] > p["budget"]["new_fits_max"]:
        raise ValueError("Budget does not match arms x seeds")
    if p["budget"]["fits"] * p["training"]["max_epochs"] > p["budget"]["new_epochs_max"]:
        raise ValueError("Epoch budget exceeded")
    if set(p["comparisons"]) != {"C1", "C2", "C3", "C4", "C5"}:
        raise ValueError("Comparison family departs from the frozen design")
    if p["training"].get("selection_cohort") != "normal":
        raise ValueError("Every arm must select on the same normal validation cohort")
    return True


def arm_config(p, arm):
    spec = p["arms"][arm]
    channels = p["data"]["channel_sets"][spec["channels"]]
    return {"arm": arm, "family": spec["family"], "channel_set": spec["channels"], "channel_indices": list(channels),
            "channels": len(channels), "train_cohort": spec["train_cohort"],
            "selection_cohort": p["training"]["selection_cohort"],
            "expected_parameters": p["families"][spec["family"]]["parameters"][spec["channels"]],
            "args": dict(p["families"][spec["family"]].get("args", {}))}


# --------------------------------------------------------------------------- data identity (D0)
def verify_preparation(meta, protocol, csv_sha256):
    prep = protocol["data"]["v28_preparation"]
    problems = []
    if meta.get("schema_version") != prep["schema_version"]:
        problems.append("schema_version")
    if sorted(meta.get("years", [])) != sorted(prep["years"]):
        problems.append(f"years {meta.get('years')}")
    if sorted(meta.get("curve_years", [])) != sorted(prep["curve_years"]):
        problems.append(f"curve_years {meta.get('curve_years')}")
    if meta.get("prepared", {}).get("sha256") != csv_sha256:
        problems.append("prepared csv sha256")
    if meta.get("labels_modified") or meta.get("clipping") or meta.get("interpolation"):
        problems.append("labels modified/clipped/interpolated")
    if any(int(y) in protocol["data"]["sealed_years"] for y in meta.get("years", [])):
        problems.append("sealed year prepared")
    if problems:
        raise ValueError(f"v28 preparation identity failed: {problems}")
    return True


def verify_development_rows(v28_frame, dev_frame):
    """Every registered 2016-2019 development row must reappear unchanged in the v28 preparation."""
    key = ["turbine_id", "timestamp_hour_end_utc"]
    missing = [c for c in dev_frame.columns if c not in v28_frame.columns]
    if missing:
        raise ValueError(f"v28 preparation lacks development columns {missing}")
    dev = dev_frame.sort_values(key, kind="mergesort").reset_index(drop=True)
    stamps = set(zip(dev["turbine_id"].astype(str), dev["timestamp_hour_end_utc"].astype(str)))
    mask = [(str(t), str(s)) in stamps for t, s in zip(v28_frame["turbine_id"], v28_frame["timestamp_hour_end_utc"])]
    new = v28_frame.loc[mask, list(dev.columns)].sort_values(key, kind="mergesort").reset_index(drop=True)
    if len(new) != len(dev):
        raise ValueError(f"v28 preparation has {len(new)} of {len(dev)} development rows")
    for column in dev.columns:
        a, b = dev[column], new[column]
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            same = np.array_equal(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)
        else:
            same = a.astype(str).equals(b.astype(str))
        if not same:
            raise ValueError(f"Development column {column!r} differs in the v28 preparation")
    return {"rows_compared": int(len(dev)), "columns_compared": int(len(dev.columns))}


def verify_power_curves(v28_meta, dev_meta):
    if v28_meta.get("power_curves") != dev_meta.get("power_curves"):
        raise ValueError("Power curves differ from the registered development preparation")
    return True


# --------------------------------------------------------------------------- metrics
def metrics(prediction, truth, rated):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 2 or not p.size or rated <= 0 or not (np.isfinite(p).all() and np.isfinite(y).all()):
        raise ValueError("Aligned finite [window, lead] arrays required")
    e = p - y
    sst = float(((y - y.mean()) ** 2).sum())
    lead_sst = ((y - y.mean(0)) ** 2).sum(0)
    return {"windows": int(len(y)), "mae": float(np.abs(e).mean()), "mse": float((e ** 2).mean()),
            "rmse": float(np.sqrt((e ** 2).mean())), "r2": 1 - float((e ** 2).sum()) / sst if sst > 0 else None,
            "mbe": float(e.mean()), "nmae_pct_rated": float(100 * np.abs(e).mean() / rated),
            "nrmse_pct_rated": float(100 * np.sqrt((e ** 2).mean()) / rated),
            "hit_5pct_capacity_pct": float(100 * (np.abs(e) <= .05 * rated).mean()),
            "hit_10pct_capacity_pct": float(100 * (np.abs(e) <= .10 * rated).mean()),
            "lead_mse": (e ** 2).mean(0).tolist(),
            "lead_mae": np.abs(e).mean(0).tolist(), "lead_rmse": np.sqrt((e ** 2).mean(0)).tolist(),
            "lead_r2": (1 - (e ** 2).sum(0) / np.where(lead_sst > 0, lead_sst, np.nan)).tolist()}


def seed_mean(per_seed):
    keys = ("mae", "mse", "rmse", "r2", "mbe", "nmae_pct_rated", "nrmse_pct_rated",
            "hit_5pct_capacity_pct", "hit_10pct_capacity_pct")
    out = {k: None if any(m[k] is None for m in per_seed) else float(np.mean([m[k] for m in per_seed])) for k in keys}
    for k in ("lead_mae", "lead_mse", "lead_rmse", "lead_r2"):
        out[k] = safe(np.mean([m[k] for m in per_seed], axis=0).tolist())
    out["seeds"] = len(per_seed)
    return out


def validation_choice(validation_records, p):
    """Lowest seed-mean validation (normal) MAE among the choice candidates; ties -> fewer parameters."""
    rows = []
    for arm in CHOICE_CANDIDATES:
        maes = [r["validation"]["mae"] for r in validation_records if r["arm"] == arm]
        if len(maes) != len(p["seeds"]):
            raise ValueError(f"Incomplete validation records for {arm}")
        rows.append((float(np.mean(maes)), arm_config(p, arm)["expected_parameters"], arm))
    rows.sort()
    return {"arm": rows[0][2], "ranking": [{"arm": a, "validation_mae_seed_mean": m, "parameters": n} for m, n, a in rows]}


# --------------------------------------------------------------------------- comparisons and rules
def paired(candidate_preds, reference_preds, truth, issue_times, settings):
    """Seed-averaged losses on identical windows (not an ensemble)."""
    cand_abs = np.mean([np.abs(c - truth) for c in candidate_preds], axis=0)
    ref_abs = np.mean([np.abs(r - truth) for r in reference_preds], axis=0)
    cand_sq = np.mean([(c - truth) ** 2 for c in candidate_preds], axis=0)
    ref_sq = np.mean([(r - truth) ** 2 for r in reference_preds], axis=0)
    kwargs = {"block_length": settings["block_length"], "replicates": settings["replicates"], "seed": settings["seed"]}
    return {"absolute_error": paired_loss_statistics(cand_abs, ref_abs, issue_times, **kwargs),
            "rmse_difference": rmse_difference_bootstrap(cand_sq, ref_sq, issue_times, **kwargs),
            "per_lead_mae_gain_kw": (ref_abs - cand_abs).mean(0).tolist()}


def compare_all(predictions, truth, issue_times, comparisons, choice, settings):
    """predictions: {arm: [seed predictions]} on one cohort. Returns C1-C5 with Holm over MAE NW p-values."""
    out = {}
    for name, spec in comparisons.items():
        candidate = choice if spec["candidate"] == "VALIDATION_CHOSEN" else spec["candidate"]
        out[name] = {"candidate": candidate, "reference": spec["reference"], "question": spec["question"],
                     **paired(predictions[candidate], predictions[spec["reference"]], truth, issue_times, settings)}
    names = sorted(out)
    adjusted = holm_adjust([out[n]["absolute_error"]["newey_west"]["p_value_two_sided"] for n in names])
    for n, a in zip(names, adjusted):
        out[n]["holm_p"] = float(a)
    return out


def evaluate_registered(directory, p, choice):
    """Metrics (both cohorts), C1-C5 and the rules from the registered test arrays only (controller and report)."""
    from pathlib import Path
    folder = Path(directory) / "evaluation"
    with np.load(folder / "test_cohort.npz", allow_pickle=False) as saved:
        cohort = {k: saved[k] for k in saved.files}
    n, horizon = cohort["truth"].shape
    if horizon != p["data"]["pred_len"] or not n or cohort["times"].shape != (n, horizon):
        raise ValueError("Invalid common test queue dimensions")
    if cohort["normal_mask"].shape != (n,) or cohort["normal_mask"].dtype != bool or not cohort["normal_mask"].any():
        raise ValueError("A nonempty boolean normal cohort is required")
    if len(np.unique(cohort["window_starts"])) != n or cohort["turbines"].shape != (n,):
        raise ValueError("Duplicate or misaligned common windows")
    rated, settings = p["data"]["rated_power_kw"], p["statistics"]
    predictions = {arm: [cohort[arm]] for arm in PERSISTENCE_ARMS}
    for arm in TRAINED_ARMS:
        predictions[arm] = []
        for seed in p["seeds"]:
            with np.load(folder / f"{arm}_s{seed}.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["window_starts"], cohort["window_starts"]):
                    raise ValueError(f"{arm} s{seed} test windows differ from the common cohort")
                if saved["prediction"].shape != cohort["truth"].shape or not np.isfinite(saved["prediction"]).all():
                    raise ValueError(f"{arm} s{seed} prediction shape or finiteness differs")
                predictions[arm].append(saved["prediction"])
    out = {"windows": {"all": int(len(cohort["truth"])), "normal": int(cohort["normal_mask"].sum())},
           "normal_share_pct": float(100 * cohort["normal_mask"].mean()), "metrics": {}, "comparisons": {},
           "validation_choice": choice}
    issue = cohort["times"][:, 0] - HOUR
    for name in COHORTS:
        mask = cohort["normal_mask"].astype(bool) if name == "normal" else np.ones(len(cohort["truth"]), bool)
        truth = cohort["truth"][mask]
        subset = {arm: [q[mask] for q in preds] for arm, preds in predictions.items()}
        out["metrics"][name] = {arm: seed_mean([metrics(q, truth, rated) for q in preds]) for arm, preds in subset.items()}
        for arm, result in out["metrics"][name].items():
            for anchor in PERSISTENCE_ARMS:
                reference = out["metrics"][name][anchor]
                for loss in ("mae", "rmse"):
                    result[f"{loss}_skill_pct_vs_{anchor}"] = (100 * (1 - result[loss] / reference[loss])
                                                               if reference[loss] > 0 else None)
        out["comparisons"][name] = compare_all(subset, truth, issue[mask], p["comparisons"], choice, settings)
    out["rules"] = rules(out["comparisons"])
    return out


def _significant(comparison):
    return comparison["absolute_error"]["ci95_low"] > 0 and comparison["holm_p"] <= .05


def _positive(comparison):
    return _significant(comparison) and comparison["rmse_difference"]["rmse_gain_kw"] > 0


def rules(cohort_comparisons):
    """Predeclared protocol rules on {cohort: comparisons}."""
    normal, everything = cohort_comparisons["normal"], cohort_comparisons["all"]

    def holds_all(name):
        return everything[name]["absolute_error"]["gain"] >= 0
    flags = {"information_supported": any(_positive(normal[c]) and holds_all(c) for c in ("C1", "C2")),
             "information_supported_by": [c for c in ("C1", "C2") if _positive(normal[c]) and holds_all(c)],
             "capacity_supported": bool(_positive(normal["C3"]) and holds_all("C3")),
             "training_cleaning_supported": bool(_significant(normal["C4"]) and holds_all("C4")),
             "beats_naive": bool(_significant(normal["C5"]) and _significant(everything["C5"]))}
    return flags
```

## A4. `scripts/train_kelmarsh_multiyear_benchmark.py`

```python
#!/usr/bin/env python3
"""v28: first multi-year Kelmarsh numeric base benchmark (24h -> 4h, six turbines, Wiki off).

Train 2016-2019 (normal-operation windows; DLinear_alltrain on all windows), epoch selection on 2020, ONE test
evaluation on 2021 reported on both the normal (primary, conditional) and all (deployment) cohorts; 2022-2024 sealed.
Arms: PersistenceHour, PersistenceLast10 (no fit) and 6 trained configurations x 3 seeds = 18 fits:
DLinear_normal, DLinear_alltrain, LinearFlat_base, LinearFlat_last10, TSMixer_base, TSMixer_last10.
Every arm uses the SAME windows (one series with the union of channels). Stages:
  D0 data identity (v28 preparation vs registered 2016-2019 development CSV) -> T1 18 fits -> S1 selection freeze
  -> E1 single test evaluation -> decision (predeclared rules) -> report. No test window exists before the S1 freeze.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import inspect
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from data_provider.kelmarsh_multiyear import KelmarshMultiYearSeries  # noqa: E402
from models.DLinear import Model as DLinearModel  # noqa: E402
from models.LinearFlat import Model as LinearFlatModel  # noqa: E402
from models.TSMixerBasicSmall import Model as TSMixerModel  # noqa: E402
from scripts import train_kelmarsh_anchored_residual as v26run  # noqa: E402  (audited numerics helpers)
from utils import kelmarsh_multiyear_benchmark as b  # noqa: E402
from utils.metrics import forecast_metrics  # noqa: E402

PROTOCOL = ROOT / "configs/kelmarsh_multiyear_base_benchmark_protocol_v28.json"
PROTOCOL_SHA256 = "86f7a8df11e42ceb041e08a381e739166b4c88c4038eacbfe05c48963292df3d"
LOG_ROOT = ROOT / "outputs/logs/Kelmarsh"
POINTER = LOG_ROOT / "multiyear_benchmark_latest.txt"
LOCK = LOG_ROOT / "multiyear_benchmark.lock"
PREFLIGHT_DIR = LOG_ROOT / "multiyear_benchmark_preflight"
TRAINING_FILES = ("models/DLinear.py", "models/LinearFlat.py", "models/TSMixerBasicSmall.py", "models/__init__.py",
                  "models/DLinearAnchoredResidual.py", "data_provider/kelmarsh_hourly.py", "data_provider/__init__.py",
                  "data_provider/kelmarsh_multiyear.py", "utils/kelmarsh_enrichment.py", "utils/metrics.py",
                  "scripts/train_kelmarsh_anchored_residual.py", "utils/kelmarsh_backbone_information.py",
                  "utils/kelmarsh_anchored_residual.py", "utils/kelmarsh_multiyear_benchmark.py",
                  "scripts/train_kelmarsh_multiyear_benchmark.py", "utils/paired_forecast_statistics.py")
V28_FILES = TRAINING_FILES + ("scripts/report_kelmarsh_multiyear_benchmark.py", "utils/paired_forecast_statistics.py",
                              "scripts/train/Kelmarsh_launch_multiyear_benchmark.sh", "tests/test_kelmarsh_multiyear_benchmark.py")
V28_TESTS = ("tests/test_kelmarsh_multiyear_benchmark.py",)
REGRESSION_TESTS = ("tests/test_kelmarsh_multiyear_preprocessing.py", "tests/test_tsmixer_basic_small.py",
                    "tests/test_hourly_wind_acquisition.py")
V28_FILES += REGRESSION_TESTS + ("scripts/prepare_kelmarsh_multiyear.py", "scripts/acquire_hourly_wind_datasets.py")
VOLATILE = {"status.json", "status.env", "progress.json", "launch.log", "launcher.pid", "controller.pid",
            "events.jsonl", "experiment_log.md", "fit_progress.json"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def registrable(path, directory):
    path = Path(path)
    return (path.is_file() and path.name not in VOLATILE and not path.name.endswith(".tmp")
            and path != Path(directory) / "completed.json")


class SimulatedInterrupt(RuntimeError):
    """Tests only."""


# =========================================================================== data views and models
class ChannelView(Dataset):
    """Same windows, a subset of channels (power stays last); scaler restricted accordingly."""

    def __init__(self, windows, indices):
        self.windows, self.indices = windows, np.asarray(indices, dtype=np.int64)
        if self.indices[-1] != len(windows.feature_columns) - 1:
            raise ValueError("Power must stay the last channel")
        self.window_starts = windows.window_starts
        self.scaler = SimpleNamespace(mean_=np.asarray(windows.scaler.mean_)[self.indices],
                                      scale_=np.asarray(windows.scaler.scale_)[self.indices])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        x, y, mx, my = self.windows[index]
        return x[:, self.indices], y[:, self.indices], mx, my

    def truth_and_anchor(self):
        return self.windows.truth_and_anchor()


def build_model(config):
    family, channels = config["family"], config["channels"]
    if family == "DLinear":
        model = DLinearModel(SimpleNamespace(**config["args"], seq_len=24, pred_len=4, enc_in=channels))
    elif family == "LinearFlat":
        model = LinearFlatModel(seq_len=24, channels=channels, pred_len=4)
    elif family == "TSMixerBasicSmall":
        a = config["args"]
        model = TSMixerModel(seq_len=24, channels=channels, pred_len=4, blocks=a["blocks"], ff_dim=a["ff_dim"],
                             dropout=a["dropout"], norm_eps=a["norm_eps"], target_slot=channels - 1, input_mode="full")
    else:
        raise ValueError(f"Unknown family {family!r}")
    count = int(sum(q.numel() for q in model.parameters() if q.requires_grad))
    if count != int(config["expected_parameters"]):
        raise ValueError(f"{config['arm']}: {count} parameters, protocol requires {config['expected_parameters']}")
    return model, count


# =========================================================================== trainer (v25 fit_arm semantics)
def verify_trained(out, contract):
    trained = b.read_json(out / "trained.json")
    mismatch = b.contract_mismatches(trained["contract"], contract)
    if mismatch:
        raise ValueError(f"Registered fit {out} has a different contract ({mismatch})")
    b.verify_registered_files(out, trained["artifacts"])
    return trained


def fit_config(config, p, *, fit, selection, contract, seed, device, out, events=None, stop_after_epoch=None):
    """seed_all -> construct -> epoch-0 assessment (selectable) -> epochs with loader(seed+epoch) shuffling,
    (1-w)|e|+w e^2 on standardized power, clip, AdamW, strict-MAE best, patience at the top of each epoch, cosine step
    after each epoch. last.pt carries the selected state (atomic source of truth on resume)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "trained.json").exists():
        return verify_trained(out, contract) | {"reused_from_registry": True}
    cfg = p["training"]
    weight = float(cfg["loss_mse_weight"])
    if abs((1 - weight) - float(cfg["loss_mae_weight"])) > 1e-12:
        raise ValueError("Loss weights must be (1-w, w)")
    device = torch.device(device)
    v26run.seed_all(seed)
    model, count = build_model(config)
    init_state = v26run.cpu_state(model)
    init_digest = v26run.state_digest(init_state)
    init_path, last_path, best_path = out / "init.pt", out / "last.pt", out / "best.pt"
    if init_path.exists():
        saved = torch.load(init_path, map_location="cpu", weights_only=False)
        if saved["state_sha256"] != init_digest or b.contract_mismatches(saved["contract"], contract):
            raise ValueError("Saved initial state differs from the deterministic construction")
    else:
        v26run.atomic_save({"model": init_state, "state_sha256": init_digest, "contract": contract}, init_path)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["scheduler_t_max"], eta_min=cfg["scheduler_eta_min"])
    truth, _ = selection.truth_and_anchor()
    rated = p["data"]["rated_power_kw"]

    def assess():
        begin = time.monotonic()
        prediction = v26run.predict(model, selection, seed, device, cfg["eval_batch_size"])
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise ValueError("Nonfinite or misaligned selection predictions")
        metrics = forecast_metrics(prediction, truth, rated_power=rated)
        return metrics["mae"], metrics, time.monotonic() - begin

    def save_last(epoch):
        v26run.atomic_save({"model": v26run.cpu_state(model), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "contract": contract, "epoch": epoch, "best": best,
                            "best_epoch": best_epoch, "bad_epochs": bad, "best_model": best_state, "history": history,
                            "timing": timing, "rng": v26run.capture_rng(device)}, last_path)
        b.write_json(out / "fit_progress.json", {"last_completed_epoch": epoch, "epoch_in_progress": None})

    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if b.contract_mismatches(checkpoint["contract"], contract):
            raise ValueError("Interrupted fit identity changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history, start = checkpoint["history"], int(checkpoint["epoch"])
        best, best_epoch, bad, timing = checkpoint["best"], checkpoint["best_epoch"], checkpoint["bad_epochs"], checkpoint["timing"]
        best_state = checkpoint["best_model"]
        v26run.restore_rng(checkpoint["rng"], device)
        v26run.atomic_save({"model": best_state, "epoch": best_epoch, "contract": contract, "selection_mae_kw": best}, best_path)
        if events is not None:
            b.append_event(events, "fit_resumed", out=str(out), from_epoch=start)
    else:
        score, metrics, seconds = assess()
        history = [{"epoch": 0, "selection": metrics, "assess_seconds": seconds, "lr": optimizer.param_groups[0]["lr"]}]
        best, best_epoch, bad, start, best_state = score, 0, 0, 0, v26run.cpu_state(model)
        timing = {"train_seconds": 0.0, "assess_seconds": seconds}
        v26run.atomic_save({"model": best_state, "epoch": 0, "contract": contract, "selection_mae_kw": best}, best_path)
        save_last(0)
    for epoch in range(start + 1, int(cfg["max_epochs"]) + 1):
        if bad >= int(cfg["patience"]):
            break
        if stop_after_epoch is not None and epoch > stop_after_epoch:
            raise SimulatedInterrupt(f"after epoch {epoch - 1}")
        b.write_json(out / "fit_progress.json", {"last_completed_epoch": epoch - 1, "epoch_in_progress": epoch})
        model.train()
        total, batches, lr, begin = 0., 0, optimizer.param_groups[0]["lr"], time.monotonic()
        for x, y, _, _ in v26run.make_loader(fit, int(seed) + epoch, shuffle=True, batch=cfg["batch_size"], device=device):
            x, y = x.float().to(device), y.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            difference = model(x)[..., -1] - y[..., -1]
            loss = (1 - weight) * difference.abs().mean() + weight * difference.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            optimizer.step()
            total += loss.item()
            batches += 1
        train_seconds = time.monotonic() - begin
        score, metrics, assess_seconds = assess()
        timing["train_seconds"] += train_seconds
        timing["assess_seconds"] += assess_seconds
        history.append({"epoch": epoch, "train_loss": total / batches, "lr": lr, "selection": metrics,
                        "train_seconds": train_seconds, "assess_seconds": assess_seconds})
        if score < best:
            best, best_epoch, bad, best_state = score, epoch, 0, v26run.cpu_state(model)
            v26run.atomic_save({"model": best_state, "epoch": epoch, "contract": contract, "selection_mae_kw": best}, best_path)
        else:
            bad += 1
        scheduler.step()
        save_last(epoch)
        b.write_json(out / "training_history.json", history)
        if events is not None:
            b.append_event(events, "epoch", out=out.name, seed=seed, epoch=epoch, train_loss=total / batches,
                           validation_mae_kw=score, best_epoch=best_epoch)
        print(f"[EPOCH] {out.parent.name}/{out.name} epoch={epoch} train_loss={total / batches:.6f} "
              f"validation_mae={score:.4f} best_epoch={best_epoch}", flush=True)
    b.write_json(out / "training_history.json", history)
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    if int(best_checkpoint["epoch"]) != int(best_epoch):
        raise ValueError("best.pt does not register the selected epoch")
    model.load_state_dict(best_checkpoint["model"], strict=True)
    prediction = v26run.predict(model, selection, seed, device, cfg["eval_batch_size"])
    replay = forecast_metrics(prediction, truth, rated_power=rated)
    recorded = history[best_epoch]["selection"]
    if abs(replay["mae"] - recorded["mae"]) > 1e-6:
        raise ValueError("best.pt does not reproduce its registered validation MAE")
    np.savez_compressed(out / "validation_predictions.npz", prediction=prediction, truth=truth,
                        window_starts=selection.window_starts)
    trained = {"contract": contract, "arm": config["arm"], "seed": int(seed), "best_epoch": int(best_epoch),
               "epochs_completed": len(history) - 1, "validation_best": recorded,
               "validation_last": history[-1]["selection"], "parameters": count, "init_state_sha256": init_digest,
               "best_state_sha256": v26run.state_digest(best_checkpoint["model"]), "timing": timing,
               "device": str(device), "completed_utc": now(),
               "artifacts": {name: b.file_sha256(out / name) for name in
                             ("init.pt", "best.pt", "last.pt", "training_history.json", "validation_predictions.npz")}}
    b.write_json(out / "trained.json", trained)
    return trained


TRAINING_FUNCTIONS = (ChannelView, build_model, verify_trained, fit_config)


def training_code_sha256():
    return b.canonical_sha256({"files": {n: b.text_sha256_lf(ROOT / n) for n in TRAINING_FILES},
                               "functions": {f.__name__: inspect.getsource(f) for f in TRAINING_FUNCTIONS},
                               "torch_major_minor": ".".join(torch.__version__.split(".")[:2])})


def load_protocol(path=PROTOCOL, expected=PROTOCOL_SHA256):
    digest = b.text_sha256_lf(path)
    if expected is not None and digest != expected:
        raise ValueError("Frozen v28 protocol bytes changed; a new protocol needs a new id")
    protocol = b.read_json(path)
    b.validate_protocol(protocol)
    return protocol, digest


def full_source_digest(protocol_sha):
    return b.canonical_sha256({"protocol_sha256": protocol_sha,
                               "sources": {k: b.text_sha256_lf(ROOT / k) for k in V28_FILES}})


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as error:  # informational
        return f"unavailable: {type(error).__name__}"


# =========================================================================== synthetic data (smoke and tests only)
def synthetic_prepared_frame(years=(2016, 2017, 2018), days=14, seed=0):
    """Enriched-CSV-like frame with the columns the provider reads; NEVER real data."""
    rng = np.random.default_rng(seed)
    parts = []
    for year in years:
        dates = pd.date_range(f"{year}-12-{31 - days + 1:02d} 01:00", periods=days * 24, freq="h", tz="UTC")
        t = np.arange(len(dates), dtype=float)
        for unit in range(1, 7):
            wind = 7 + 3 * np.sin(t / 9 + unit) + rng.normal(0, .3, len(t))
            power = np.clip(2050 / (1 + np.exp(-(wind - 8) / 1.4)) + rng.normal(0, 30, len(t)), -10, 2060)
            parts.append(pd.DataFrame({
                "timestamp_hour_end_utc": dates.astype(str), "turbine_id": unit, "wind_speed_mps": wind,
                "ambient_temp_c": 5 + np.sin(t / 24), "nacelle_position_deg": (t * 7) % 360, "pitch_a_deg": 1.,
                "pitch_b_deg": 1.1, "pitch_c_deg": .9, "rotor_speed_rpm": 8 + wind / 2, "power_kw": power,
                "power_last10_kw": power + rng.normal(0, 40, len(t)), "wind_last10_mps": wind + rng.normal(0, .3, len(t)),
                "normal_operation": rng.random(len(t)) > .04}))
    return pd.concat(parts, ignore_index=True)


SYNTHETIC_PREPARATION = {"schema_version": 2, "curve_years": [2016], "prepared": {"sha256": "synthetic"}}


# =========================================================================== controller
class V28Run:
    def __init__(self, directory, *, protocol=None, protocol_sha=None, series_factory=None, gpu_index=0,
                 require_preflight=True, write_pointer=True, lock_path=LOCK, verify_data=True):
        """``series_factory``/``verify_data=False`` are for synthetic tests only (never from the CLI)."""
        self.dir = Path(directory)
        if protocol is None:
            protocol, protocol_sha = load_protocol()
        self.p, self.protocol_sha = protocol, protocol_sha
        self.series_factory, self.verify_data = series_factory, verify_data
        self.gpu_index, self.require_preflight, self.write_pointer, self.lock_path = gpu_index, require_preflight, write_pointer, lock_path
        self.events = self.dir / "events.jsonl"
        self.fit_calls = []

    # ------------------------------------------------------------------ bookkeeping
    def rel(self, path):
        return str(Path(path).relative_to(self.dir)).replace("\\", "/")

    def set_status(self, **fields):
        path = self.dir / "status.json"
        status = b.read_json(path) if path.exists() else {"train": "NOT_STARTED", "report": "PENDING", "stage": None}
        status.update(fields, updated_utc=now())
        b.write_json(path, status)
        (self.dir / "status.env").write_text(f"TRAIN={status['train']}\nREPORT={status['report']}\nSTAGE={status.get('stage')}\n",
                                             encoding="utf-8")
        return status

    def log(self, title, **items):
        with (self.dir / "experiment_log.md").open("a", encoding="utf-8") as handle:
            handle.writelines([f"\n## {now()} {title}\n"] + [f"- {k}: {v}\n" for k, v in items.items()])

    def freeze_protocol_record(self):
        record = {"protocol_id": self.p["protocol_id"], "protocol_sha256_lf": self.protocol_sha, "protocol": self.p}
        path = self.dir / "protocol.json"
        if path.exists():
            if b.canonical_sha256(b.read_json(path)) != b.canonical_sha256(record):
                raise ValueError("Frozen protocol changed on resume")
        else:
            b.write_json(path, record)

    def claim_pointer(self):
        if not self.write_pointer:
            return
        POINTER.parent.mkdir(parents=True, exist_ok=True)
        if POINTER.exists() and POINTER.read_text().strip() and Path(POINTER.read_text().strip()).resolve() != self.dir.resolve():
            raise ValueError(f"Pointer already registers another v28 run: {POINTER.read_text().strip()}")
        POINTER.write_text(str(self.dir.resolve()), encoding="utf-8")

    def record_sources(self):
        manifest = {"utc": now(), "training_code_sha256": training_code_sha256(),
                    "full_source_digest": full_source_digest(self.protocol_sha), "git_head": git_head(),
                    "environment": {"host": socket.gethostname(), "python": sys.version, "platform": platform.platform(),
                                    "torch": torch.__version__, "cuda": torch.cuda.is_available(),
                                    "determinism": v26run.determinism_record()}}
        path = self.dir / "source_manifest.json"
        if not path.exists():
            b.write_json(path, manifest)
        elif b.read_json(path)["full_source_digest"] != manifest["full_source_digest"]:
            b.append_event(self.events, "source_changed_on_resume", current=manifest["full_source_digest"])
        self.training_code, self.source_digest = manifest["training_code_sha256"], manifest["full_source_digest"]

    def check_preflight(self):
        if not self.require_preflight:
            return
        record = PREFLIGHT_DIR / f"passed_{self.source_digest[:16]}.json"
        if not record.exists() or not b.read_json(record).get("passed") or b.read_json(record).get("full_source_digest") != self.source_digest:
            raise ValueError("No passed --preflight-only record for the current code/protocol; run it before any fit")

    # ------------------------------------------------------------------ D0
    def load_series(self, include_test=False):
        d = self.p["data"]
        if include_test and not (self.dir / "selection_freeze.json").exists():
            raise PermissionError("Test series requires the S1 selection freeze")
        splits = d["splits"] if include_test else {k: v for k, v in d["splits"].items() if k != "test"}
        kwargs = {"splits": splits, "input_len": d["input_len"], "horizon": d["pred_len"], "stride": d["stride"],
                  "extras": tuple(d["extras_in_common_series"]), "start": d["start_utc"]}
        # Hour-ending midnight belongs to the preceding label year. Exclude the next hour.
        if not include_test:
            kwargs["end"] = f"{max(splits['validation']) + 1}-01-01T01:00:00Z"
        if self.series_factory is not None:
            return self.series_factory(kwargs)
        return KelmarshMultiYearSeries(ROOT / d["v28_preparation"]["csv"], **kwargs)

    def stage_d0(self):
        self.set_status(stage="D0", train="RUNNING")
        d = self.p["data"]
        identity = {"verification": "SKIPPED_SYNTHETIC_TEST_ONLY"}
        if self.verify_data:
            csv = ROOT / d["v28_preparation"]["csv"]
            meta_path = csv.parent.parent / "metadata" / "preparation.json"
            v28_meta = b.read_json(meta_path)
            b.verify_preparation(v28_meta, self.p, b.file_sha256(csv))
            dev = d["development_preparation"]
            dev_csv = ROOT / dev["csv"]
            dev_meta_path = dev_csv.parent.parent / "metadata" / "preparation.json"
            if b.file_sha256(dev_csv) != dev["csv_sha256"] or b.file_sha256(dev_meta_path) != dev["preparation_json_sha256"]:
                raise ValueError("Registered development preparation changed")
            rows = b.verify_development_rows(pd.read_csv(csv), pd.read_csv(dev_csv))
            b.verify_power_curves(v28_meta, b.read_json(dev_meta_path))
            identity = {"v28_csv_sha256": v28_meta["prepared"]["sha256"], "v28_preparation_json_sha256": b.file_sha256(meta_path),
                        "development_rows_identical": rows, "power_curves_identical": True,
                        "prepared_years": v28_meta["years"], "curve_years": v28_meta["curve_years"]}
        self.series = self.load_series()
        audit = self.series.audit
        if audit["channels"] != d["channels"]:
            raise ValueError("Series channels differ from the protocol")
        manifest = {"identity": identity, "series_audit": audit, "test_windows_instantiated": False}
        path = self.dir / "data_manifest.json"
        if path.exists():
            old = b.read_json(path)
            if b.canonical_sha256({k: old[k] for k in ("identity", "series_audit")}) != \
                    b.canonical_sha256({k: manifest[k] for k in ("identity", "series_audit")}):
                raise ValueError("D0 data identity differs from the registered D0 of this run")
        else:
            b.write_json(path, manifest)
            self.log("D0 data identity", identity=identity, windows=audit["windows"])
        b.append_event(self.events, "D0_verified", windows=audit["windows"])

    def windows(self, split, cohort):
        if split == "test" and not (self.dir / "selection_freeze.json").exists():
            raise PermissionError("Test windows may only be instantiated after the S1 selection freeze")
        b.append_event(self.events, "windows_instantiated", split=split, cohort=cohort)
        return self.series.windows(split, cohort)

    # ------------------------------------------------------------------ T1
    def device(self):
        path = self.dir / "device.json"
        if path.exists():
            record = b.read_json(path)
        else:
            record = b.choose_pair_device(v26run.gpu_snapshot(self.gpu_index), self.p["operations"]) | {"decided_utc": now()}
            b.write_json(path, record)
        if record["device"] == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Registered CUDA device unavailable; resume on the same host class")
            device = torch.device(f"cuda:{self.gpu_index}")
            torch.cuda.set_per_process_memory_fraction(record["memory_fraction"], device)
            return device
        return torch.device("cpu")

    def registered_fits(self):
        return sorted(self.rel(q.parent) for q in (self.dir / "runs").glob("*/s*/trained.json"))

    def stage_t1(self):
        device = self.device()
        cache = {}
        for arm in b.TRAINED_ARMS:
            config = b.arm_config(self.p, arm)
            key = (config["train_cohort"], config["selection_cohort"])
            if key not in cache:
                cache[key] = (self.windows("train", key[0]), self.windows("validation", key[1]))
            fit_w, val_w = cache[key]
            fit, selection = ChannelView(fit_w, config["channel_indices"]), ChannelView(val_w, config["channel_indices"])
            for seed in self.p["seeds"]:
                out = self.dir / "runs" / arm / f"s{seed}"
                contract = {"protocol_sha256": self.protocol_sha, "training_code_sha256": self.training_code,
                            "arm": arm, "config": config, "seed": int(seed), "fit_window_hash": b.window_hash(fit.window_starts),
                            "selection_window_hash": b.window_hash(selection.window_starts),
                            "scaler_mean": fit.scaler.mean_.tolist(), "scaler_scale": fit.scaler.scale_.tolist(),
                            "training": self.p["training"], "device_type": device.type,
                            "determinism": v26run.determinism_record()}
                if not (out / "trained.json").exists():
                    self.check_preflight()
                    if not (out / "last.pt").exists() and len(self.registered_fits()) + 1 > self.p["budget"]["new_fits_max"]:
                        raise ValueError("The 18-fit budget would be exceeded; refuse")
                    self.set_status(stage="T1", train="RUNNING")
                    b.write_json(self.dir / "progress.json", {"arm": arm, "seed": seed, "utc": now()})
                    b.append_event(self.events, "fit_started", arm=arm, seed=seed, device=str(device))
                self.fit_calls.append((arm, seed))
                trained = fit_config(config, self.p, fit=fit, selection=selection, contract=contract, seed=seed,
                                     device=device, out=out, events=self.events)
                if not trained.get("reused_from_registry"):
                    print(f"[FIT] {arm} s{seed} best_epoch={trained['best_epoch']} "
                          f"val_MAE={trained['validation_best']['mae']:.4f}", flush=True)
                gc.collect()
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ S1
    def stage_s1(self):
        records, fits = [], {}
        for arm in b.TRAINED_ARMS:
            for seed in self.p["seeds"]:
                folder = self.dir / "runs" / arm / f"s{seed}"
                trained = b.read_json(folder / "trained.json")
                b.verify_registered_files(folder, trained["artifacts"])
                records.append({"arm": arm, "seed": seed, "best_epoch": trained["best_epoch"],
                                "validation": {k: trained["validation_best"][k] for k in ("mae", "rmse", "mse")},
                                "train_cohort": b.arm_config(self.p, arm)["train_cohort"]})
                records[-1]["selection_cohort"] = b.arm_config(self.p, arm)["selection_cohort"]
                fits[self.rel(folder)] = {"trained_sha256": b.file_sha256(folder / "trained.json"),
                                          "best_sha256": b.file_sha256(folder / "best.pt")}
        choice = b.validation_choice([r for r in records if r["train_cohort"] == "normal"], self.p)
        freeze = {"records": records, "fits": fits, "validation_choice": choice, "frozen_utc": now(),
                  "rule": "written before any test window is instantiated"}
        path = self.dir / "selection_freeze.json"
        if path.exists():
            old = b.read_json(path)
            if b.canonical_sha256({k: old[k] for k in ("records", "fits", "validation_choice")}) != \
                    b.canonical_sha256({k: freeze[k] for k in ("records", "fits", "validation_choice")}):
                raise ValueError("Selected states changed after the S1 freeze")
        else:
            b.write_json(path, freeze)
            self.log("S1 selection freeze", validation_choice=choice["arm"], ranking=choice["ranking"])
        b.append_event(self.events, "S1_frozen", choice=choice["arm"])
        return b.read_json(path)

    # ------------------------------------------------------------------ E1
    def stage_e1(self, freeze):
        self.set_status(stage="E1", train="RUNNING")
        if b.canonical_sha256(freeze) != b.canonical_sha256(b.read_json(self.dir / "selection_freeze.json")):
            raise ValueError("Evaluation freeze identity changed")
        b.verify_registered_files(self.dir, {name + "/best.pt": entry["best_sha256"]
                                            for name, entry in freeze["fits"].items()})
        development = self.series
        self.series = self.load_series(include_test=True)
        for attr in ("scaler_mean", "scaler_scale"):
            if not np.array_equal(getattr(development, attr), getattr(self.series, attr)):
                raise ValueError("Adding the test year changed the frozen training scaler")
        for split in ("train", "validation"):
            for cohort_name in b.COHORTS:
                before, after = development.windows(split, cohort_name), self.series.windows(split, cohort_name)
                pairs = [(before.target_times(), after.target_times()),
                         (before.turbines[before.window_starts], after.turbines[after.window_starts])]
                pairs += list(zip(before.truth_and_anchor(), after.truth_and_anchor()))
                if any(not np.array_equal(a, c) for a, c in pairs):
                    raise ValueError(f"Adding test year changed the {split}/{cohort_name} queue or labels")
        b.append_event(self.events, "test_series_opened_after_freeze", windows=self.series.audit["windows"]["test"])
        test_all = self.windows("test", "all")
        test_normal = self.windows("test", "normal")
        normal_mask = np.isin(test_all.window_starts, test_normal.window_starts)
        truth, persistence_hour = test_all.truth_and_anchor()
        folder = self.dir / "evaluation"
        folder.mkdir(exist_ok=True)
        cohort = {"truth": truth, "times": test_all.target_times(), "turbines": test_all.turbines[test_all.window_starts],
                  "window_starts": test_all.window_starts, "normal_mask": normal_mask,
                  "PersistenceHour": persistence_hour, "PersistenceLast10": test_all.last10_anchor()}
        self._register_npz(folder / "test_cohort.npz", cohort)
        cpu, batch = torch.device("cpu"), self.p["training"]["eval_batch_size"]
        for arm in b.TRAINED_ARMS:
            config = b.arm_config(self.p, arm)
            view = ChannelView(test_all, config["channel_indices"])
            for seed in self.p["seeds"]:
                run = self.dir / "runs" / arm / f"s{seed}"
                saved_path = folder / f"{arm}_s{seed}.npz"
                registry = saved_path.with_suffix(".json")
                identity = {"best_sha256": b.file_sha256(run / "best.pt"),
                            "cohort_sha256": b.file_sha256(folder / "test_cohort.npz")}
                if registry.exists():
                    registered = b.read_json(registry)
                    if registered["identity"] != identity or registered["prediction_sha256"] != b.file_sha256(saved_path):
                        raise ValueError("Registered test prediction identity changed")
                    continue
                trained = b.read_json(run / "trained.json")
                state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)["model"]
                if v26run.state_digest(state) != trained["best_state_sha256"]:
                    raise ValueError(f"{run}/best.pt differs from the frozen selected state")
                model, _ = build_model(config)
                model.load_state_dict(state, strict=True)
                prediction = v26run.predict(model, view, seed, cpu, batch)
                self._register_npz(saved_path, {"prediction": prediction, "window_starts": test_all.window_starts})
                b.write_json(registry, {"identity": identity, "prediction_sha256": b.file_sha256(saved_path)})
        b.append_event(self.events, "E1_test_predicted", windows=int(len(truth)), normal=int(normal_mask.sum()))
        return self.decide(freeze)

    @staticmethod
    def _register_npz(path, arrays):
        if path.exists():  # a resume must reproduce registered arrays exactly (CPU, deterministic)
            with np.load(path, allow_pickle=False) as saved:
                for key, value in arrays.items():
                    if not np.array_equal(np.asarray(value), saved[key]):
                        raise ValueError(f"{path.name}:{key} differs from the registered evaluation")
        else:
            temporary = path.with_name(path.stem + ".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, path)

    def decide(self, freeze):
        payload = b.evaluate_registered(self.dir, self.p, freeze["validation_choice"]["arm"])
        path = self.dir / "decision.json"
        if path.exists():
            if b.read_json(path)["rules"] != payload["rules"]:
                raise ValueError("Registered decision would change; refuse")
        else:
            b.write_json(path, payload | {"utc": now()})
        self.log("E1 single test evaluation", rules=payload["rules"], choice=freeze["validation_choice"]["arm"])
        print(f"[RULES] {payload['rules']}", flush=True)
        return payload

    # ------------------------------------------------------------------ finish
    def finish_training(self):
        hashes = {self.rel(q): b.file_sha256(q) for q in sorted(self.dir.rglob("*"))
                  if registrable(q, self.dir) and q.name != "training_completed.json"}
        b.write_json(self.dir / "training_completed.json", {"hashes": hashes, "fits": self.registered_fits(), "utc": now()})
        self.set_status(stage="TRAINING_DONE", train="TRAIN_COMPLETED")

    def report(self):
        from scripts import report_kelmarsh_multiyear_benchmark as reporting
        self.set_status(stage="REPORT", report="RUNNING")
        try:
            summary = reporting.build_report(self.dir)
        except Exception as error:
            self.set_status(report="REPORT_FAILED", report_error=f"{type(error).__name__}: {error}")
            raise
        self.set_status(report="REPORT_COMPLETED", report_error=None)
        b.write_json(self.dir / "completed.json", {self.rel(q): b.file_sha256(q) for q in sorted(self.dir.rglob("*"))
                                                   if registrable(q, self.dir)})
        self.set_status(stage="COMPLETED")
        return summary

    def run(self, report_only=False):
        self.dir.mkdir(parents=True, exist_ok=True)
        lock = b.acquire_exclusive_lock(self.lock_path)
        try:
            if (self.dir / "completed.json").exists() and not report_only:
                self.freeze_protocol_record()
                b.verify_training_completed(self.dir)
                b.verify_registered_files(self.dir, b.read_json(self.dir / "completed.json"))
                print("[REUSE] v28 already completed; nothing retrained", flush=True)
                return "COMPLETED"
            v26run.configure_determinism()
            (self.dir / "controller.pid").write_text(str(os.getpid()), encoding="utf-8")
            self.freeze_protocol_record()
            self.claim_pointer()
            self.record_sources()
            b.append_event(self.events, "controller_started", argv=sys.argv, report_only=report_only)
            if (self.dir / "training_completed.json").exists():
                b.verify_training_completed(self.dir)
                self.report()
                return "COMPLETED"
            if report_only:
                raise ValueError("--report-only requires a registered TRAIN_COMPLETED run")
            self.stage_d0()
            self.stage_t1()
            freeze = self.stage_s1()
            self.stage_e1(freeze)
            self.finish_training()
            self.report()
            return "COMPLETED"
        finally:
            lock.close()


evaluate_registered = b.evaluate_registered  # torch-free; shared by controller and report


# =========================================================================== preflight and CLI
def protocol_namespace_check(p):
    own, configs, pointers = p["protocol_id"], [], {}
    for path in sorted((ROOT / "configs").glob("*protocol*.json")):
        if path.resolve() == PROTOCOL.resolve():
            continue
        try:
            if b.read_json(path).get("protocol_id") == own:
                configs.append(str(path))
        except (ValueError, AttributeError):
            continue
    for path in sorted(LOG_ROOT.glob("*latest.txt")):
        if path.resolve() == POINTER.resolve():
            continue
        target = Path(path.read_text(errors="replace").strip())
        marker = (target if target.is_absolute() else ROOT / target) / "protocol.json"
        if marker.exists() and b.read_json(marker).get("protocol_id") == own:
            pointers[str(path)] = str(target)
    return {"protocol_id": own, "configs": configs, "pointers": pointers, "passed": not configs and not pointers}


def smoke(p, folder):
    """Synthetic end-to-end smoke of the three families on the provider's window views (CPU, 2 epochs)."""
    q = {**p, "training": {**p["training"], "max_epochs": 2, "scheduler_t_max": 2}}
    series = KelmarshMultiYearSeries(synthetic_prepared_frame(), splits={"train": (2016,), "validation": (2017,), "test": (2018,)},
                                     extras=tuple(p["data"]["extras_in_common_series"]), preparation=SYNTHETIC_PREPARATION)
    out = {}
    for arm in ("DLinear_normal", "LinearFlat_last10", "TSMixer_base"):
        config = b.arm_config(p, arm)
        fit = ChannelView(series.windows("train", "normal"), config["channel_indices"])
        val = ChannelView(series.windows("validation", "normal"), config["channel_indices"])
        trained = fit_config(config, q, fit=fit, selection=val, contract={"smoke": arm}, seed=7, device=torch.device("cpu"),
                             out=folder / arm)
        out[arm] = {"best_epoch": trained["best_epoch"], "validation_mae": trained["validation_best"]["mae"],
                    "parameters": trained["parameters"]}
    return {"passed": True, "arms": out, "data": "synthetic frame only"}


def preflight(gpu_index=0):
    p, protocol_sha = load_protocol()
    digest = full_source_digest(protocol_sha)
    folder = PREFLIGHT_DIR / f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}_{digest[:12]}"
    folder.mkdir(parents=True, exist_ok=False)
    v26run.configure_determinism()
    checks = {"namespace": protocol_namespace_check(p)}
    try:
        import pytest  # noqa: F401
        command = [sys.executable, "-m", "pytest", "-q", *V28_TESTS, *REGRESSION_TESTS]
    except ImportError:
        command = [sys.executable, "-m", "unittest", *[t.replace("/", ".")[:-3] for t in V28_TESTS + REGRESSION_TESTS]]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (folder / "tests.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    checks["tests"] = {"command": command, "exit_code": result.returncode, "passed": result.returncode == 0,
                       "tail": (result.stdout + result.stderr).splitlines()[-15:]}
    code = subprocess.run(["bash", "-n", "scripts/train/Kelmarsh_launch_multiyear_benchmark.sh"], cwd=ROOT).returncode
    checks["bash_syntax"] = {"exit_code": code, "passed": code == 0}
    try:
        d0 = V28Run(folder / "d0", protocol=p, protocol_sha=protocol_sha, require_preflight=False, write_pointer=False,
                    lock_path=PREFLIGHT_DIR / "preflight.lock")
        d0.dir.mkdir(parents=True)
        d0.freeze_protocol_record()
        d0.record_sources()
        d0.stage_d0()
        checks["D0"] = {"passed": True, **b.read_json(d0.dir / "data_manifest.json")}
    except Exception as error:
        checks["D0"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    try:
        checks["synthetic_smoke"] = smoke(p, folder / "smoke")
    except Exception as error:
        checks["synthetic_smoke"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    checks["resources"] = {"gpu": v26run.gpu_snapshot(gpu_index), "cpu_count": os.cpu_count()}
    passed = all(checks[k]["passed"] for k in ("namespace", "tests", "bash_syntax", "D0", "synthetic_smoke"))
    record = {"passed": bool(passed), "full_source_digest": digest, "protocol_sha256": protocol_sha,
              "training_code_sha256": training_code_sha256(), "git_head": git_head(), "utc": now(), "checks": checks}
    b.write_json(folder / "preflight.json", record)
    if passed:
        b.write_json(PREFLIGHT_DIR / f"passed_{digest[:16]}.json", record)
    print(f"[PREFLIGHT] passed={passed} tests={checks['tests']['passed']} D0={checks['D0']['passed']} "
          f"smoke={checks['synthetic_smoke']['passed']} namespace={checks['namespace']['passed']} record={folder}", flush=True)
    return record


def print_status():
    if not POINTER.exists():
        print("STATUS=NOT_STARTED")
        return
    directory = Path(POINTER.read_text().strip())
    print(f"RESULT_DIR={directory}")
    for name in ("status.env", "progress.json"):
        if (directory / name).exists():
            print((directory / name).read_text(encoding="utf-8").strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.status:
        print_status()
        return 0
    if args.preflight_only:
        return 0 if preflight(args.gpu)["passed"] else 5
    if args.output is None:
        raise ValueError("An explicit independent --output directory is required")
    run = V28Run(args.output.resolve(), gpu_index=args.gpu)
    try:
        run.run(report_only=args.report_only)
        return 0
    except BlockingIOError:
        print("[BUSY] Another v28 controller holds the lock; nothing changed", flush=True)
        return 3
    except Exception as error:
        if args.output.exists():
            status = run.dir / "status.json"
            if not status.exists() or b.read_json(status).get("train") != "TRAIN_COMPLETED":
                run.set_status(train="FAILED", error=f"{type(error).__name__}: {error}")
            b.append_event(run.events, "controller_failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    sys.exit(main())
```

## A5. `scripts/report_kelmarsh_multiyear_benchmark.py`

```python
#!/usr/bin/env python3
"""v28 report: recompute every metric, comparison and rule from the REGISTERED test arrays (torch-free); refuse to
report if they differ from the registered decision. Both cohorts in every table: normal (primary, conditional) and all
(deployment). Never trains, never selects, never reads unregistered data."""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils import kelmarsh_multiyear_benchmark as b  # noqa: E402
from utils.kelmarsh_backbone_information import select_trace_segment  # noqa: E402

REPORT_SOURCES = ("scripts/report_kelmarsh_multiyear_benchmark.py", "utils/kelmarsh_multiyear_benchmark.py",
                  "utils/kelmarsh_backbone_information.py", "utils/paired_forecast_statistics.py")
TOLERANCE = 1e-9
ARMS = b.PERSISTENCE_ARMS + b.TRAINED_ARMS


def table(rows, columns, digits=3):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            cells.append(f"{value:.{digits}f}" if isinstance(value, float) and math.isfinite(value) else
                         ("undefined" if value is None else str(value)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def check_against_registered(fresh, registered):
    def compare(a, c, path):
        if isinstance(a, dict) and isinstance(c, dict) and set(a) == set(c):
            for key in a:
                compare(a[key], c[key], f"{path}/{key}")
            return
        if isinstance(a, list) and isinstance(c, list) and len(a) == len(c):
            for i, (left, right) in enumerate(zip(a, c)):
                compare(left, right, f"{path}/{i}")
            return
        if isinstance(a, (int, float)) and isinstance(c, (int, float)):
            if math.isfinite(a) and math.isfinite(c) and abs(a - c) <= TOLERANCE:
                return
        elif a == c:
            return
        raise ValueError(f"Recomputed {path} disagrees with the registered decision")
    compare(b.safe(fresh), {k: v for k, v in registered.items() if k != "utc"}, "decision")


def figures(fresh, folder):
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    x = np.arange(len(ARMS))
    for ax, key, label in zip(axes, ("mae", "rmse"), ("MAE (kW)", "RMSE (kW)")):
        for offset, cohort in ((-.2, "normal"), (.2, "all")):
            values = [fresh["metrics"][cohort][arm][key] for arm in ARMS]
            ax.bar(x + offset, values, .4, label=f"test {cohort}")
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("_", "\n") for a in ARMS], fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("v28 test 2021 (single evaluation): normal = conditional primary cohort, all = deployment cohort", fontsize=9)
    fig.tight_layout()
    fig.savefig(folder / "test_metrics.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, cohort in zip(axes, b.COHORTS):
        for arm in ARMS:
            ax.plot([1, 2, 3, 4], fresh["metrics"][cohort][arm]["lead_mae"], marker="o", label=arm,
                    ls="--" if arm.startswith("Persistence") else "-")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("lead (hours)")
        ax.set_title(f"test {cohort}: MAE by lead (kW)", fontsize=10)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(folder / "lead_mae.png", dpi=130)
    plt.close(fig)


def forecast_trace(directory, p, choice):
    with np.load(directory / "evaluation/test_cohort.npz", allow_pickle=False) as saved:
        cohort = {k: saved[k] for k in saved.files}
    with np.load(directory / f"evaluation/{choice}_s{p['seeds'][0]}.npz", allow_pickle=False) as saved:
        prediction = saved["prediction"]
    selection = select_trace_segment(cohort["times"][:, 0] - b.HOUR, cohort["turbines"], stride_hours=p["data"]["stride"])
    rows = selection["rows"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharey=True)
    for ax, lead in zip(axes, (0, 3)):
        x = cohort["times"][rows, lead]
        ax.plot(x, cohort["truth"][rows, lead], label="Observed", color="black")
        ax.plot(x, prediction[rows, lead], label=f"{choice}, seed {p['seeds'][0]}")
        ax.plot(x, cohort["PersistenceLast10"][rows, lead], label="PersistenceLast10", ls="--")
        ax.set_title(f"Turbine {selection['turbine']}, lead {lead + 1} h; first eligible 72 h of issue times", fontsize=9)
        ax.set_ylabel("Power (kW)")
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(directory / "figures/forecast_trace.png", dpi=130)
    plt.close(fig)
    b.write_json(directory / "forecast_trace_selection.json", selection)


def build_report(directory):
    directory = Path(directory)
    p = b.read_json(directory / "protocol.json")["protocol"]
    training = b.verify_training_completed(directory)
    freeze = b.read_json(directory / "selection_freeze.json")
    registered = b.read_json(directory / "decision.json")
    required = ["selection_freeze.json", "decision.json", "evaluation/test_cohort.npz"]
    required += [f"evaluation/{arm}_s{seed}.npz" for arm in b.TRAINED_ARMS for seed in p["seeds"]]
    for name in required:
        if training["hashes"].get(name) != b.file_sha256(directory / name):
            raise ValueError(f"Registered artifact changed or missing: {name}")
    fresh = b.evaluate_registered(directory, p, freeze["validation_choice"]["arm"])
    check_against_registered(fresh, registered)
    rows = []
    for cohort in b.COHORTS:
        for arm in ARMS:
            m = fresh["metrics"][cohort][arm]
            rows.append({"cohort": cohort, "arm": arm, "seeds": m["seeds"], "mae": m["mae"], "mse": m["mse"], "rmse": m["rmse"],
                         "r2": m["r2"], "mbe": m["mbe"], "nmae_pct_rated": m["nmae_pct_rated"],
                         "nrmse_pct_rated": m["nrmse_pct_rated"],
                         **{k: v for k, v in m.items() if "skill_pct" in k or k.startswith("hit_")},
                         **{f"lead{h + 1}_mae": v for h, v in enumerate(m["lead_mae"])},
                         **{f"lead{h + 1}_mse": v for h, v in enumerate(m["lead_mse"])},
                         **{f"lead{h + 1}_rmse": v for h, v in enumerate(m["lead_rmse"])},
                         **{f"lead{h + 1}_r2": v for h, v in enumerate(m["lead_r2"])}})
    pd.DataFrame(rows).to_csv(directory / "test_metrics.csv", index=False)
    comparison_rows = []
    for cohort in b.COHORTS:
        for name, c in sorted(fresh["comparisons"][cohort].items()):
            a, r = c["absolute_error"], c["rmse_difference"]
            comparison_rows.append({"cohort": cohort, "comparison": name, "candidate": c["candidate"], "reference": c["reference"],
                                    "mae_gain_kw": a["gain"], "mae_ci95_low": a["ci95_low"], "mae_ci95_high": a["ci95_high"],
                                    "nw_p": a["newey_west"]["p_value_two_sided"], "holm_p": c["holm_p"],
                                    "rmse_gain_kw": r["rmse_gain_kw"], "rmse_ci95_low": r["ci95_low_kw"],
                                    "rmse_ci95_high": r["ci95_high_kw"]})
    pd.DataFrame(comparison_rows).to_csv(directory / "test_comparisons.csv", index=False)
    figures(fresh, directory / "figures")
    forecast_trace(directory, p, freeze["validation_choice"]["arm"])
    result = {"protocol_id": p["protocol_id"], "rules": fresh["rules"], "validation_choice": freeze["validation_choice"],
              "windows": fresh["windows"], "normal_share_pct": fresh["normal_share_pct"],
              "scope": "single 2021 test evaluation; normal = ex-post conditional cohort; all = deployment cohort; "
                       "development years were inspected in earlier protocols; 2022-2024 sealed",
              "independent_recomputation": "all metrics, comparisons and rules recomputed from registered arrays"}
    b.write_json(directory / "result.json", result)
    write_markdown(directory, p, fresh, freeze, rows, comparison_rows)
    outputs = ["test_metrics.csv", "test_comparisons.csv", "result.json", "report.md", "figures/test_metrics.png", "figures/lead_mae.png",
               "figures/forecast_trace.png", "forecast_trace_selection.json"]
    b.write_json(directory / "report_manifest.json", {
        "report_sources": {n: b.file_sha256(ROOT / n) for n in REPORT_SOURCES},
        "inputs": {"training_completed.json": b.file_sha256(directory / "training_completed.json")},
        "outputs": {n: b.file_sha256(directory / n) for n in outputs},
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    return result


def write_markdown(directory, p, fresh, freeze, rows, comparison_rows):
    validation = [{"arm": r["arm"], "seed": r["seed"], "best_epoch": r["best_epoch"], "train_cohort": r["train_cohort"],
                   **{f"val_{k}": v for k, v in r["validation"].items()}} for r in freeze["records"]]
    lines = [f"# v28 multi-year Kelmarsh base benchmark report ({dt.date.today().isoformat()})", "",
             f"**Rules:** {fresh['rules']}", "",
             f"Validation-chosen configuration (2020, normal, frozen before test): **{freeze['validation_choice']['arm']}**", "",
             f"Test 2021 windows: all {fresh['windows']['all']}, normal {fresh['windows']['normal']} "
             f"({fresh['normal_share_pct']:.2f}%). Every arm uses the same windows. R² is not a percentage accuracy; "
             "the normal cohort uses the target period's operating state and is conditional accuracy only.", "",
             p["rules"].get("attribution_limits", ""), "",
             "Skill is reported against both Persistence anchors; capacity hit rates are tolerance statistics, not classification accuracy. "
             "Point RMSE is averaged across seeds; paired RMSE intervals use the root of seed-averaged squared errors. "
             "Eight consecutive issue-time blocks span 24 clock hours only without gaps.", "",
             "## Test metrics (seed-mean over 3 seeds for trained arms)", "",
             table(rows, ["cohort", "arm", "mae", "rmse", "r2", "mbe", "nmae_pct_rated", "lead1_mae", "lead4_mae", "lead1_r2", "lead4_r2"]), "",
             "## Predeclared comparisons (positive gain favours the candidate; Holm over C1-C5 within each cohort)", "",
             table(comparison_rows, ["cohort", "comparison", "candidate", "reference", "mae_gain_kw", "mae_ci95_low", "mae_ci95_high",
                                     "nw_p", "holm_p", "rmse_gain_kw", "rmse_ci95_low", "rmse_ci95_high"]), "",
             "## Validation (2020) selected states", "",
             table(validation, ["arm", "seed", "best_epoch", "train_cohort", "val_mae", "val_rmse"]), "",
             "![test metrics](figures/test_metrics.png)", "", "![lead MAE](figures/lead_mae.png)", "",
             "![first eligible forecast segment, not selected by error](figures/forecast_trace.png)", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(build_report(args.directory)["rules"])


if __name__ == "__main__":
    main()
```

## A6. `scripts/train/Kelmarsh_launch_multiyear_benchmark.sh`

```bash
#!/usr/bin/env bash
# v28 Kelmarsh multi-year base benchmark launcher (D0 data identity, 18 fits, S1 freeze, single 2021 test).
#   preflight only:   bash scripts/train/Kelmarsh_launch_multiyear_benchmark.sh --preflight-only [--detach]
#   first run:        bash scripts/train/Kelmarsh_launch_multiyear_benchmark.sh [--detach]
#   status:           bash scripts/train/Kelmarsh_launch_multiyear_benchmark.sh --status
#   resume:           bash scripts/train/Kelmarsh_launch_multiyear_benchmark.sh --resume [--detach]
#   report only:      bash scripts/train/Kelmarsh_launch_multiyear_benchmark.sh --report-only
# A real fit refuses to start without a passed --preflight-only record for the exact code/protocol.
# Own resources only: 2 CPU threads; GPU use is decided once per run by the controller. Never stops,
# pauses or renices other processes and never changes device, clock, MPS or network settings.
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${KELMARSH_PYTHON:-python}"
SCRIPT="scripts/train_kelmarsh_multiyear_benchmark.py"
POINTER="outputs/logs/Kelmarsh/multiyear_benchmark_latest.txt"
LAUNCH_LOCK="outputs/logs/Kelmarsh/multiyear_benchmark.launch.lock"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

usage() { echo "Usage: $0 [--status|--resume|--preflight-only|--report-only] [--detach]" >&2; }
MODE="run"; DETACH=0; PASS=()
for arg in "$@"; do
  case "${arg}" in
    --status) MODE="status" ;;
    --resume) MODE="resume"; PASS+=("${arg}") ;;
    --preflight-only) MODE="preflight"; PASS+=("${arg}") ;;
    --report-only) MODE="report"; PASS+=("${arg}") ;;
    --detach) DETACH=1 ;;
    *) usage; exit 2 ;;
  esac
done
mkdir -p outputs/logs/Kelmarsh

if [[ "${MODE}" == "status" ]]; then
  exec "${PYTHON}" "${SCRIPT}" --status
fi

if [[ "${DETACH}" == "1" ]]; then
  mkdir -p outputs/logs/Kelmarsh/launcher
  LOG="outputs/logs/Kelmarsh/launcher/multiyear_benchmark_${MODE}_$(date +%Y%m%d_%H%M%S).log"
  nohup setsid bash "${SELF}" ${PASS[@]+"${PASS[@]}"} > "${LOG}" 2>&1 < /dev/null &
  echo "DETACHED_PID=$! LOG=${LOG}"
  exit 0
fi

# One launcher at a time (the Python controller additionally holds its own flock).
exec 9>>"${LAUNCH_LOCK}"
if ! flock -n 9; then
  echo "[BUSY] another v28 launcher holds ${LAUNCH_LOCK}; nothing started" >&2
  exit 3
fi

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON}" -u "${SCRIPT}" --preflight-only
  exit $?
fi

case "${MODE}" in
  resume|report)
    [[ -s "${POINTER}" ]] || { echo "No v28 pointer; nothing to resume" >&2; exit 2; }
    RESULT_DIR="$(<"${POINTER}")"
    [[ -d "${RESULT_DIR}" ]] || { echo "Registered v28 directory missing: ${RESULT_DIR}" >&2; exit 2; }
    ;;
  run)
    if [[ -s "${POINTER}" ]]; then
      echo "Existing v28 run $(<"${POINTER}"); use --status or --resume (no duplicate study)" >&2
      exit 2
    fi
    DATE_DIR="outputs/logs/Kelmarsh/$(date +%Y%m%d)"
    mkdir -p "${DATE_DIR}"
    RESULT_DIR="$(mktemp -d "${DATE_DIR}/multiyear_benchmark_v28_XXXXXX")"
    ;;
esac
echo "RESULT_DIR=${RESULT_DIR}"
[[ -f "${RESULT_DIR}/source_commit.txt" ]] || git rev-parse HEAD > "${RESULT_DIR}/source_commit.txt"
echo "${BASHPID}" > "${RESULT_DIR}/launcher.pid"
EXTRA=()
[[ "${MODE}" == "report" ]] && EXTRA+=(--report-only)
{
  echo "[LAUNCH] $(date -u +%Y-%m-%dT%H:%M:%SZ) mode=${MODE} host=$(hostname) commit=$(git rev-parse HEAD)"
  echo "[LAUNCH] command: ${PYTHON} -u ${SCRIPT} --output ${RESULT_DIR} --gpu ${GPU:-0} ${EXTRA[*]:-}"
} | tee -a "${RESULT_DIR}/launch.log"
set +e
"${PYTHON}" -u "${SCRIPT}" --output "${RESULT_DIR}" --gpu "${GPU:-0}" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a "${RESULT_DIR}/launch.log"
CODE=${PIPESTATUS[0]}
set -e
echo "[LAUNCHER_EXIT] $(date -u +%Y-%m-%dT%H:%M:%SZ) code=${CODE}" | tee -a "${RESULT_DIR}/launch.log"
exit "${CODE}"
```

## A7. `tests/test_kelmarsh_multiyear_benchmark.py`

```python
"""v28 tests: protocol freeze, data identity, arm configs, metrics/rules, report recomputation and tamper refusal; with
torch: models, channel views and the controller end-to-end on synthetic data (test access only after the S1 freeze)."""
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import pandas as pd

from utils import kelmarsh_multiyear_benchmark as b

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/kelmarsh_multiyear_base_benchmark_protocol_v28.json"
SCRIPT = ROOT / "scripts/train_kelmarsh_multiyear_benchmark.py"
try:
    import torch
    import sklearn  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def protocol():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


class ProtocolTests(unittest.TestCase):
    def test_frozen_hash_design_and_parameters(self):
        constant = re.search(r'PROTOCOL_SHA256 = "([0-9a-f]{64})"', SCRIPT.read_text(encoding="utf-8")).group(1)
        self.assertEqual(b.text_sha256_lf(PROTOCOL), constant)
        p = protocol()
        self.assertTrue(b.validate_protocol(p))
        self.assertEqual({a: b.arm_config(p, a)["expected_parameters"] for a in b.TRAINED_ARMS},
                         {"DLinear_normal": 200, "DLinear_alltrain": 200, "LinearFlat_base": 676, "LinearFlat_last10": 868,
                          "TSMixer_base": 3618, "TSMixer_last10": 4262})
        self.assertEqual(b.arm_config(p, "DLinear_alltrain")["train_cohort"], "all")
        self.assertTrue(all(b.arm_config(p, a)["selection_cohort"] == "normal" for a in b.TRAINED_ARMS))
        self.assertEqual(p["data"]["splits"], {"train": [2016, 2017, 2018, 2019], "validation": [2020], "test": [2021]})
        self.assertEqual(p["budget"]["fits"], 18)

    def test_departures_are_refused(self):
        def mutated(path, value):
            p = protocol()
            target = p
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            return p
        for path, value in ((("data", "splits", "test"), [2020]), (("data", "v28_preparation", "curve_years"), [2016, 2020]),
                            (("data", "yaw_error_channels_enabled"), True), (("budget", "fits"), 24),
                            (("data", "channel_sets", "base"), [0, 1, 2, 3, 4, 5]), (("seeds",), [2024, 2025, 2026, 2027]),
                            (("families", "LinearFlat", "parameters", "last10"), 900), (("data", "sealed_years"), [2021])):
            with self.assertRaises(ValueError, msg=str(path)):
                b.validate_protocol(mutated(path, value))
        p = protocol()
        p["arms"].pop("TSMixer_last10")
        with self.assertRaises(ValueError):
            b.validate_protocol(p)


class DataIdentityTests(unittest.TestCase):
    def meta(self):
        return {"schema_version": 2, "years": [2016, 2017, 2018, 2019, 2020, 2021], "curve_years": [2016, 2017, 2018, 2019],
                "prepared": {"sha256": "abc"}, "labels_modified": False, "clipping": False, "interpolation": False}

    def test_preparation_identity(self):
        p = protocol()
        self.assertTrue(b.verify_preparation(self.meta(), p, "abc"))
        for change in ({"years": [2016, 2017, 2018, 2019, 2020, 2021, 2022]}, {"curve_years": [2016, 2017]},
                       {"schema_version": 1}, {"labels_modified": True}):
            with self.assertRaises(ValueError):
                b.verify_preparation(self.meta() | change, p, "abc")
        with self.assertRaises(ValueError):
            b.verify_preparation(self.meta(), p, "other")

    def frames(self):
        stamps = pd.date_range("2019-12-31 20:00", periods=8, freq="h", tz="UTC").astype(str)
        rows = [{"timestamp_hour_end_utc": s, "turbine_id": t, "power_kw": float(i * 10 + t), "flag_stopped": i == 3,
                 "power_last10_kw": np.nan if i == 2 else float(i)} for i, s in enumerate(stamps) for t in (1, 2)]
        v28 = pd.DataFrame(rows)
        dev = v28[v28.timestamp_hour_end_utc <= "2020-01-01 00:00:00+00:00"].copy()
        return v28, dev

    def test_development_rows_must_be_identical(self):
        v28, dev = self.frames()
        self.assertEqual(b.verify_development_rows(v28.sample(frac=1, random_state=0), dev)["rows_compared"], len(dev))
        changed = v28.copy()
        changed.loc[0, "power_kw"] += 1e-9
        with self.assertRaisesRegex(ValueError, "power_kw"):
            b.verify_development_rows(changed, dev)
        with self.assertRaisesRegex(ValueError, "development rows"):
            b.verify_development_rows(v28.iloc[1:], dev)
        with self.assertRaises(ValueError):
            b.verify_power_curves({"power_curves": {"1": [1]}}, {"power_curves": {"1": [2]}})


def fake_evaluation(root, p, rng):
    folder = Path(root) / "evaluation"
    folder.mkdir(parents=True)
    issues = 160
    times = np.repeat(np.datetime64("2021-03-01T00", "h") + np.arange(issues) * 3 * np.timedelta64(1, "h"), 6)
    times = (times[:, None] + np.arange(1, 5)[None] * np.timedelta64(1, "h")).astype("datetime64[ns]")
    n = len(times)
    truth = rng.normal(600, 400, (n, 4))
    cohort = {"truth": truth, "times": times, "turbines": np.tile(np.arange(1, 7, dtype=np.int16), issues),
              "window_starts": np.arange(n, dtype=np.int64), "normal_mask": rng.random(n) > .08,
              "PersistenceHour": truth + rng.normal(0, 150, truth.shape), "PersistenceLast10": truth + rng.normal(0, 130, truth.shape)}
    np.savez_compressed(folder / "test_cohort.npz", **cohort)
    noise = {"DLinear_normal": 110, "DLinear_alltrain": 112, "LinearFlat_base": 110, "LinearFlat_last10": 85,
             "TSMixer_base": 110, "TSMixer_last10": 86}
    common = rng.normal(0, 1, truth.shape)
    for arm in b.TRAINED_ARMS:
        for seed in p["seeds"]:
            prediction = truth + noise[arm] * (.8 * common + .6 * rng.normal(0, 1, truth.shape))
            np.savez_compressed(folder / f"{arm}_s{seed}.npz", prediction=prediction, window_starts=cohort["window_starts"])
    return cohort


class EvaluationTests(unittest.TestCase):
    def test_rules_and_both_cohorts(self):
        p = protocol()
        p["statistics"]["replicates"] = 300
        with tempfile.TemporaryDirectory() as tmp:
            fake_evaluation(tmp, p, np.random.default_rng(0))
            out = b.evaluate_registered(tmp, p, "LinearFlat_last10")
        self.assertEqual(set(out["metrics"]), {"normal", "all"})
        self.assertGreater(out["windows"]["all"], out["windows"]["normal"])
        self.assertEqual(set(out["comparisons"]["normal"]), {"C1", "C2", "C3", "C4", "C5"})
        self.assertEqual(out["comparisons"]["normal"]["C5"]["candidate"], "LinearFlat_last10")
        self.assertTrue(out["rules"]["information_supported"])
        self.assertEqual(out["rules"]["information_supported_by"], ["C1", "C2"])
        self.assertFalse(out["rules"]["capacity_supported"])
        self.assertTrue(out["rules"]["beats_naive"])
        self.assertTrue(all(0 <= c["holm_p"] <= 1 for c in out["comparisons"]["all"].values()))
        self.assertEqual(len(out["metrics"]["all"]["TSMixer_base"]["lead_mae"]), 4)

    def test_validation_choice_and_tie_break(self):
        p = protocol()
        records = [{"arm": a, "seed": s, "validation": {"mae": {"LinearFlat_base": 150., "LinearFlat_last10": 140.,
                                                                  "TSMixer_base": 150., "TSMixer_last10": 140.}[a]}}
                   for a in b.CHOICE_CANDIDATES for s in p["seeds"]]
        self.assertEqual(b.validation_choice(records, p)["arm"], "LinearFlat_last10")  # tie -> fewer parameters
        with self.assertRaises(ValueError):
            b.validation_choice(records[:-1], p)

    def test_metrics_definitions(self):
        truth = np.array([[0., 10.], [20., 30.]])
        m = b.metrics(truth + np.array([[1., -1.], [2., -2.]]), truth, 2050.)
        self.assertAlmostEqual(m["mae"], 1.5)
        self.assertAlmostEqual(m["mbe"], 0.)
        self.assertAlmostEqual(m["rmse"], np.sqrt(2.5))

    def test_holm_rejection_overrides_positive_raw_intervals(self):
        positive = {"absolute_error": {"ci95_low": .1, "gain": 1.}, "rmse_difference": {"rmse_gain_kw": 1.}, "holm_p": .2}
        comparisons = {cohort: {c: copy.deepcopy(positive) for c in ("C1", "C2", "C3", "C4", "C5")} for cohort in b.COHORTS}
        flags = b.rules(comparisons)
        self.assertFalse(flags["information_supported"])
        self.assertFalse(flags["capacity_supported"])
        self.assertFalse(flags["training_cleaning_supported"])
        self.assertFalse(flags["beats_naive"])


def fake_run(root, p):
    run = Path(root) / "run"
    rng = np.random.default_rng(1)
    fake_evaluation(run, p, rng)
    records = [{"arm": a, "seed": s, "best_epoch": 3, "train_cohort": b.arm_config(p, a)["train_cohort"],
                "validation": {"mae": 150. - (5 if a.endswith("last10") else 0), "rmse": 220., "mse": 48400.}}
               for a in b.TRAINED_ARMS for s in p["seeds"]]
    freeze = {"records": records, "validation_choice": b.validation_choice([r for r in records if r["train_cohort"] == "normal"], p)}
    b.write_json(run / "protocol.json", {"protocol": p})
    b.write_json(run / "selection_freeze.json", freeze)
    b.write_json(run / "decision.json", b.evaluate_registered(run, p, freeze["validation_choice"]["arm"]))
    names = ["selection_freeze.json", "decision.json", "evaluation/test_cohort.npz"]
    names += [f"evaluation/{arm}_s{seed}.npz" for arm in b.TRAINED_ARMS for seed in p["seeds"]]
    hashes = {name: b.file_sha256(run / name) for name in names}
    b.write_json(run / "training_completed.json", {"hashes": hashes})
    return run


class ReportTests(unittest.TestCase):
    def test_report_recomputes_and_writes_both_cohorts(self):
        from scripts import report_kelmarsh_multiyear_benchmark as report
        p = protocol()
        p["statistics"]["replicates"] = 300
        with tempfile.TemporaryDirectory() as tmp:
            run = fake_run(tmp, p)
            result = report.build_report(run)
            self.assertEqual(result["validation_choice"]["arm"], "LinearFlat_last10")
            frame = pd.read_csv(run / "test_metrics.csv")
            self.assertEqual(set(frame.cohort), {"normal", "all"})
            self.assertEqual(len(frame), 2 * 8)
            self.assertEqual(len(pd.read_csv(run / "test_comparisons.csv")), 10)
            for name in ("test_metrics", "lead_mae"):
                self.assertGreater((run / "figures" / f"{name}.png").stat().st_size, 1000)
            self.assertIn("conditional accuracy only", (run / "report.md").read_text(encoding="utf-8"))

    def test_tampering_is_refused(self):
        from scripts import report_kelmarsh_multiyear_benchmark as report
        p = protocol()
        p["statistics"]["replicates"] = 300
        with tempfile.TemporaryDirectory() as tmp:
            run = fake_run(tmp, p)
            with np.load(run / "evaluation/LinearFlat_last10_s2024.npz") as saved:
                arrays = {k: saved[k] for k in saved.files}
            arrays["prediction"] = arrays["prediction"] * 0.9
            np.savez_compressed(run / "evaluation/LinearFlat_last10_s2024.npz", **arrays)
            with self.assertRaises(ValueError):
                report.build_report(run)
        with tempfile.TemporaryDirectory() as tmp:
            run = fake_run(tmp, p)
            record = b.read_json(run / "decision.json")
            record["rules"]["beats_naive"] = not record["rules"]["beats_naive"]
            b.write_json(run / "decision.json", record)
            with self.assertRaisesRegex(ValueError, "changed or missing"):
                report.build_report(run)

    def test_every_statistic_is_recomputed(self):
        from scripts import report_kelmarsh_multiyear_benchmark as report
        p = protocol()
        p["statistics"]["replicates"] = 100
        with tempfile.TemporaryDirectory() as tmp:
            fake_evaluation(tmp, p, np.random.default_rng(0))
            fresh = b.safe(b.evaluate_registered(tmp, p, "LinearFlat_last10"))
            for path in (("comparisons", "all", "C1", "holm_p"),
                         ("comparisons", "normal", "C2", "rmse_difference", "ci95_high_kw"),
                         ("metrics", "all", "TSMixer_base", "mse")):
                changed = copy.deepcopy(fresh)
                node = changed
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] += .01
                with self.assertRaisesRegex(ValueError, "disagree"):
                    report.check_against_registered(fresh, changed)


@unittest.skipUnless(HAS_TORCH, "torch/sklearn unavailable on this host")
class ModelTests(unittest.TestCase):
    def test_models_and_channel_views(self):
        from scripts import train_kelmarsh_multiyear_benchmark as task
        from data_provider.kelmarsh_multiyear import KelmarshMultiYearSeries
        p = protocol()
        for arm in b.TRAINED_ARMS:
            config = b.arm_config(p, arm)
            model, count = task.build_model(config)
            out = model(torch.randn(5, 24, config["channels"]))
            self.assertEqual(count, config["expected_parameters"])
            self.assertEqual(out[..., -1].shape, (5, 4))
        series = KelmarshMultiYearSeries(task.synthetic_prepared_frame(), splits={"train": (2016,), "validation": (2017,), "test": (2018,)},
                                         extras=("power_last10", "wind_last10"), preparation=task.SYNTHETIC_PREPARATION)
        windows = series.windows("train", "normal")
        base = task.ChannelView(windows, p["data"]["channel_sets"]["base"])
        x, y, _, _ = base[0]
        full_x = windows[0][0]
        np.testing.assert_array_equal(x, full_x[:, [0, 1, 2, 3, 4, 5, 8]])
        self.assertEqual(base.scaler.mean_[-1], windows.scaler.mean_[-1])
        with self.assertRaises(ValueError):
            task.ChannelView(windows, [0, 1, 2])


@unittest.skipUnless(HAS_TORCH, "torch/sklearn unavailable on this host")
class ControllerTests(unittest.TestCase):
    def make(self, tmp):
        from scripts import train_kelmarsh_multiyear_benchmark as task
        from data_provider.kelmarsh_multiyear import KelmarshMultiYearSeries
        self.task = task
        p = copy.deepcopy(protocol())
        p["data"]["splits"] = {"train": [2016], "validation": [2017], "test": [2018]}
        p["seeds"] = [2024]
        p["training"].update(max_epochs=2, scheduler_t_max=2)
        p["budget"].update(fits=6, new_fits_max=6)
        p["statistics"]["replicates"] = 200

        def factory(kwargs):
            return KelmarshMultiYearSeries(task.synthetic_prepared_frame(), preparation=task.SYNTHETIC_PREPARATION, **kwargs)
        return task.V28Run(Path(tmp) / "run", protocol=p, protocol_sha="test", series_factory=factory, require_preflight=False,
                           write_pointer=False, lock_path=Path(tmp) / "lock", verify_data=False)

    def test_end_to_end_freeze_before_test_and_reuse(self):
        from unittest.mock import patch
        torch.set_num_threads(2)
        with patch("scripts.train_kelmarsh_anchored_residual.gpu_snapshot", return_value=None):
            with tempfile.TemporaryDirectory() as tmp:
                run = self.make(tmp)
                self.assertEqual(run.run(), "COMPLETED")
                self.assertEqual(len(run.registered_fits()), 6)
                events = [json.loads(line) for line in (run.dir / "events.jsonl").read_text().splitlines()]
                frozen = next(i for i, e in enumerate(events) if e["event"] == "S1_frozen")
                first_test = next(i for i, e in enumerate(events) if e.get("split") == "test")
                self.assertLess(frozen, first_test)
                self.assertFalse(any(e.get("split") == "test" for e in events[:frozen]))
                decision = b.read_json(run.dir / "decision.json")
                self.assertEqual(set(decision["metrics"]), {"normal", "all"})
                self.assertEqual(b.read_json(run.dir / "status.json")["report"], "REPORT_COMPLETED")
                for arm in b.TRAINED_ARMS:
                    trained = b.read_json(run.dir / f"runs/{arm}/s2024/trained.json")
                    self.assertEqual(trained["parameters"], b.arm_config(run.p, arm)["expected_parameters"])
                again = self.make(tmp)
                with patch.object(self.task, "fit_config", side_effect=AssertionError("retrained")):
                    self.assertEqual(again.run(), "COMPLETED")

    def test_test_windows_refused_before_freeze_and_preflight_required(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp)
            run.dir.mkdir(parents=True)
            run.record_sources()
            run.stage_d0()
            self.assertNotIn("test", run.series.splits)
            self.assertNotIn("test", run.series.audit["windows"])
            self.assertLessEqual(int(run.series.label_year.max()), 2017)
            with self.assertRaises(PermissionError):
                run.windows("test", "all")
            with self.assertRaises(PermissionError):
                run.load_series(include_test=True)
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp)
            run.require_preflight = True
            with patch.object(self.task, "PREFLIGHT_DIR", Path(tmp) / "pf"), \
                    patch("scripts.train_kelmarsh_anchored_residual.gpu_snapshot", return_value=None):
                with self.assertRaisesRegex(ValueError, "preflight"):
                    run.run()
            self.assertFalse(list(run.dir.rglob("trained.json")))

    def test_interrupted_training_matches_uninterrupted(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp)
            run.dir.mkdir(parents=True)
            run.stage_d0()
            task = self.task
            task.v26run.configure_determinism()
            config = b.arm_config(run.p, "TSMixer_base")
            fit = task.ChannelView(run.windows("train", "normal"), config["channel_indices"])
            val = task.ChannelView(run.windows("validation", "normal"), config["channel_indices"])
            arguments = dict(fit=fit, selection=val, contract={"test": "resume"}, seed=2024, device=torch.device("cpu"))
            full = task.fit_config(config, run.p, out=Path(tmp) / "full", **arguments)
            with self.assertRaises(task.SimulatedInterrupt):
                task.fit_config(config, run.p, out=Path(tmp) / "resumed", stop_after_epoch=1, **arguments)
            resumed = task.fit_config(config, run.p, out=Path(tmp) / "resumed", **arguments)
            self.assertEqual(full["best_state_sha256"], resumed["best_state_sha256"])
            self.assertEqual(full["validation_best"], resumed["validation_best"])
            full_last = torch.load(Path(tmp) / "full/last.pt", map_location="cpu", weights_only=False)
            resumed_last = torch.load(Path(tmp) / "resumed/last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(task.v26run.state_digest(full_last["model"]), task.v26run.state_digest(resumed_last["model"]))


if __name__ == "__main__":
    unittest.main()
```
