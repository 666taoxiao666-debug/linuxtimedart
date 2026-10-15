# Kelmarsh 数值底座：v11结果与v12多时间起点验证

## 最新工作：v12（已完成，未找到稳定优于Persistence的候选）

用户要求继续优先优化数值底座，Wiki 保持关闭。v11的训练期证据显示单段内部收益不稳定，
因此v12只改变时间验证与选择方式，保留原五个候选、损失、学习率、20epoch预算、原始标签、
输入与scaler，不在已看过的外层结果中改挑TCN或其他epoch。

协议：`configs/kelmarsh_numeric_rolling_protocol_v12.json`，ID
`kelmarsh_h2_numeric_expanding_origin_v12`，SHA256
`e329a8a26bc2161e0f516bcafa1f4d48108896d049d414e65011a077faa88dba`。
原v11实现文件不修改，保留完成产物与全部负结果。

执行规则：

- 三个 expanding origin，使用原 outer-train 时间轴的70%/80%/90%起点，各验证随后10%。
  沿用源数据类的精确边界，训练目标必须在当期预测 issue 前可用，验证目标跨界窗口不入选。
- 复用旧预训练初始化前，必须校验它的fit和selection窗口哈希，并验证两者最晚目标都早于
  第一个预测issue；不能只检查预训练fit，忽略用于选预训练epoch的标签。
- 同一模型/超参/同一epoch必须在三个时段上分别联合不劣于 Persistence，才有资格参与选择；
  以等权的三段MAE/RMSE误差比均值排序，不能拼接三段各自的最佳轮数。
- 90%起点与v11相同的已完成阶段，经数据、参数、初始化、代码及产物哈希检查后复用。
  若最终选中与v11完全相同的固定重训，也只复用其冻结预测，不再训练。
- 选定后最多进行一次完整 outer-train 固定轮数重训；外层只评估。没有合格候选则精确保留
  Persistence，明确其 learned Skill=0，而不是宣称优化成功。
- 不改变既有模型结构。最多6次内部神经训练、9次Ridge拟合（其中预期2+3次可严格复用），
  再加至多一次选中重训；不增加seed或新候选。

启动器：`scripts/train/Kelmarsh_launch_numeric_rolling.sh`（`--status` / `--resume`）。
指针：`outputs/logs/Kelmarsh/numeric_rolling_latest.txt`。共享GPU时只限制本进程；其他用户不被
停止、暂停、降速或重配置。分配器2GiB限制不包含全部CUDA/驱动开销，不保证对他人零影响。

本地36项相关测试、Bash语法通过，其中8项新增测试覆盖三起点切分、预训练选择标签的因果性、
统一epoch而非分段挑选、未完成/篡改产物拒绝、同一完整重训复用，以及外层不参与训练选择。
这些功能测试不能充当性能证据；服务器36项测试也通过。历史外层已开发查看，此轮仍是
开发证据，不是独立确认。多起点验证也不能保证未来有效，不能把底座变化冒充Wiki创新。

### v12完成记录与真实结果

实现提交 `952cf4b` 已推送并在服务器同步。2026-10-15 15:45:47启动，Bash PID1989320、
Python PID1989328，已COMPLETED/exit0、训练进程退出。结果目录
`outputs/logs/Kelmarsh/20261015/001_numeric_rolling_h2_f1_s2024_o70-80-90_ep20`，启动日志
`outputs/launch_logs/numeric_rolling_20261015_154547.log`。
共享资源策略实际执行过，并在未检测到其他GPU进程时恢复solo模式，没有停止或修改他人进程。

预训练最后使用的目标为2016-08-28 22:00，第一内部预测issue为08-29 02:00，初始化可用性校验
通过。三个fit/selection窗口数分别为3877/432、4315/822、5143/822。完整refit候选窗口5965，
外层仍为原638窗口。90%起点的五个阶段全部严格复用；只新增前两起点的4次20epoch神经训练与
6次Ridge拟合，没有重复预训练、90%起点或v11重训。

43个“配置×统一epoch”选择中，没有一个在三个时间段都联合不劣于Persistence：

- o70：只有PromptExpanded第1、2轮满足；
- o80：只有PromptExpanded第2轮满足；
- o90：PromptExpanded第1、3–9、11–14轮、ResidualTCN第1–3轮及Ridge alpha100满足。

因此问题不是“所有模型在所有时间都完全学不到”，而是同一模型、同一训练长度的稳定技能
没有通过当前预声明规则。该规则偏稳健，不能据此证明时间自适应训练算法或其他输入信息无效。
没有临时放宽规则或挑一个已知外层表现的runner-up。

最终精确回退Persistence，未进行新的外层重训，`primary_joint_success=false`：

- MAE **101.888844 kW**，MSE **24171.238030 kW²**，RMSE **155.471020 kW**；
- R² **0.783236**，MAE Skill **0.000%**，RMSE Skill **0.000%**；
- ±5%/±10%容量命中率 **64.5768% / 86.5204%**。

这避免部署v11的退化结果，但不是相对原底座的精度提升，也没有新增Wiki贡献。原模型与全部
负结果保留。本轮冻结预算已完成，没有排队中的新实验，也没有启动Wiki。

服务器189项产物哈希校验通过；下载的14项注册产物全部匹配。独立重算四项原尺度指标一致，
逐点验证输出与Persistence完全相等，独立重算共同合格候选数为0，精度图已实看。
本地证据：`output/kelmarsh_hourly/numeric_rolling_v12_review/independent_metric_audit.json`；
图：同目录 `reports/SelectedNumeric/artifacts/forecast_accuracy_overview.png`。

### 额外只读输入审计：不能把缺失列直接当作新信号

原小时CSV有31列，但数值模型只使用7个既有通道。来源代码
`scripts/acquire_hourly_wind_datasets.py::KEL_COLUMNS/prepare_kelmarsh` 明确将
`Turbine Power setpoint (kW)` 和 `Available Capacity for Production (kW)` 作为对应小时末已观测量
聚合；不能使用预测未来小时的这些量。

只检查原outer-train的24696行控制观测，不用外层或封存标签拟合：

- `power_setpoint_kw`在当前小时数据训练期有限值比例为 **0%**，不能填造为额定值后声称增加了信息。
- `available_capacity_kw`有限值 **98.8419%**，范围0–2050kW，209个不同值；有效值中仅
  **2.5440%**低于额定，**1.1184%**为0。三个内部起点的历史末点该量均100%存在，低于额定比例
  分别4.8611%、0.9732%、1.5815%。它是可审计的稀有运行状态信号，不是已证明的整体精度修复。

审计文件：`outputs/launch_logs/numeric_control_train_audit_20261015.json`，已下载到同一本地证据目录。
没有增加输入、插补功率设定值、改标签或因此追加新模型。后续候选应先研究真实可用的历史运行
状态或跨风机同步观测是否增加预测信息，不能只换网络/轮数，也不能仅凭缺失控制列认定整个数据集差。

### v12审查【最终评级】：C（完成后，效果验收未通过）

【必须立即修复 Top 5】：预训练选择标签可用性；统一epoch约束；不重复完成训练；共享资源
约束；外层不可反向选择。检查及服务器实测已完成，未发现有证据的P0；最终没有学得稳定精度增益。

【需重跑的实验与需补充的单元测试】：仅运行冻结v12缺失起点，不重跑v11相同阶段；完成后核对
`prepared.json`、`reuse.json`、`candidate_grid.json`、`selected.json` 和原尺度配对预测。新增关键
路径已测试，无需重复堆叠smoke；若报告失败只补报告。

=== v12详细问题清单 ===

- **严重级别**：P1（必须防止的时序风险，已加断言，未声称旧代码存在泄漏）。
  **位置**：`utils/numeric_rolling.py::initializer_audit`。
  **证据**：旧预训练epoch用base-selection标签选择；若这些标签晚于某新起点，即使预训练fit
  早于起点也不能复用。**影响**：会虚增内部跨时间技能。
  **修复**：核对fit/selection目标身份与最晚目标可用性。**测试**：向更早起点移植时必须拒绝。
  **置信度**：高。
- **严重级别**：P1（选择逻辑风险，已实现统一约束）。
  **位置**：`utils/numeric_rolling.py::candidate_grid/select_rolling`。
  **证据**：三个分段各自best epoch不是同一个可执行训练方案。**影响**：可能夸大稳定性。
  **修复**：用完整历史里共同epoch的真实指标选择，不能用分段best拼接。
  **测试**：各段分别有正收益但无共同epoch时必须回退。**置信度**：高。
- **严重级别**：P2（复现与资源风险，已实现严格复用）。
  **位置**：`scripts/train_kelmarsh_numeric_rolling.py::previous_evidence/reusable_stage/refit_prediction`。
  **证据**：v11已完成90%起点和一次完整重训；盲目重复不增加独立证据。
  **影响**：浪费资源、生成不必要的同配置重复结果。
  **修复**：相同状态复用且验哈希；不同起点才重新拟合。**测试**：相同完整重训不调用训练函数，
  篡改或不同窗口不能复用。**置信度**：高。
- **严重级别**：P1（稳定预测技能未建立）。
  **位置**：v12 `candidate_grid.json/selected.json/result.json` 和配对预测。
  **证据**：没有共同合格配置/epoch；最终逐点等于Persistence。**影响**：不支持新底座精度提升，
  不能因R²达到0.783就声称神经模型有效。**修复**：保留旧底座，下一研究问题是有效预测信息或
  时间适应机制，而非按旧外层重选参数。**测试**：同外层cohort与四项指标独立复算已通过，
  新信息能否改善仍需独立协议检验。**置信度**：高。

## 当前范围与状态

用户要求先把底座做好，再单独优化 Wiki。本轮只训练数值预测器，不建立知识记忆，
不加载 Wiki 残差、不改 Wiki 阈值，不改原始功率标签、外层滑窗或封存测试集。
底座变化后，旧 Wiki 收益证书不能继续作为新底座上的有效证据。

协议：`configs/kelmarsh_numeric_foundation_protocol_v11.json`。
ID：`kelmarsh_h2_numeric_foundation_trainonly_v11`。
协议 SHA256：`8e3c9ffc4885c5a763af7bf3bd02d1a4470e72501b8b483ba3d3b388fba29d33`。

截至本地实现审查，65 项相关单测与新启动器 Bash 语法检查通过。单测使用合成小样本，
只说明接口、隔离与复用行为正确，不是新模型精度证据。服务器曾离线，之后 SSH 已恢复；
恢复时 GPU 无计算进程，3090 显存占用约 90 MiB。实际部署、启动及结果另记，不能把
代码完成、GPU 空闲或训练 loss 下降当作预测改善。

部署记录（2026-10-15）：实现提交 `1d4c9d6` 已推送并在服务器快进；服务器同一组65项测试、
Bash 语法、源完成产物全部哈希、旧 scaler 一致性检查通过。实际内部 fit/selection 窗口
5143/822、完整 outer-train 重训5965，旧数值 fit/selection 为3453/424，外层评估仍为638。
14:57:00 启动，Bash PID1983019、Python PID1983027；结果目录
`outputs/logs/Kelmarsh/20261015/001_numeric_foundation_h2_f1_s2024_inner90_ep20`，独立启动日志
`outputs/launch_logs/numeric_foundation_20261015_145700.log`。已确认进入实际 GPU 训练，
PromptExpanded 内部20轮完成，ResidualTCN 继续训练；当时为 `solo_unrestricted`，无其他GPU进程。

## 完成结果：本轮没有优化成功，不替换原底座

已 `COMPLETED`、exit0，GPU 训练进程退出。五个内部候选和唯一一次固定重训均完成，
未追加候选或重选外层最佳 epoch。内部选中 PromptExpanded 第8轮，然后以同一初始化
在完整训练期固定训练8轮，Wiki 全程关闭。

- 原底座：MAE **101.888844** kW、MSE **24171.238137** kW²、RMSE **155.471020** kW、
  R² **0.783236**，实质等同 Persistence。
- 新底座：MAE **116.431975** kW、MSE **29585.416973** kW²、RMSE **172.004119** kW、
  R² **0.734682**；MAE/RMSE Skill 为 **−14.273526% / −10.634200%**。
- 新底座容量 ±5% / ±10% 命中率 **61.2853% / 84.3260%**，原底座为
  **64.5768% / 86.5204%**。不能把 R² 或容量归一化误差补数当作准确率。
- 同638窗口、1276目标点，新底座 MAE 比原底座差 **14.543130 kW**、RMSE 差 **16.533099 kW**。
  MAE增益时间块95%区间 **[−26.319059, −2.159852] kW**；Newey–West ratio-mean
  influence 检验 p=**0.020302**。这不是“微小舍入负收益”，但仍是历史开发验证集上的探索推断。

服务器全部175项产物哈希校验通过；下载的12项注册产物也匹配哈希，预测数组的时间戳、风机、
目标和窗口与原底座完全相同。本地独立重算 MAE/MSE/RMSE/R² 与服务器一致，精度图已下载并实看。
完整8MB报告包的首次SFTP下载超时，只留下不完整副本，未用它做分析；关键文件通过legacy SCP
成功下载并校验，没有因此重训。

独立本地审计：`output/kelmarsh_hourly/numeric_foundation_v11_review/independent_metric_audit.json`。
精度图：`output/kelmarsh_hourly/numeric_foundation_v11_review/reports/SelectedNumeric/artifacts/forecast_accuracy_overview.png`。

## 训练侧诊断：为什么不能把内部正收益当作成功

严格只用 original outer-train 的内部选择段，重载被冻结的实际内部 Prompt 检查点做只读推理，
没有用 outer validation/test 选择参数。内部822窗口、137个 issue 时钟、18个自然日：
MAE 141.873769 对 Persistence 146.678333 kW，净收益4.804564 kW；两个时间半段都为正，
但18天仅9天同时改善MAE与MSE，时间块95%区间 **[−1.684444, 12.124896] kW**，p=**0.169430**。
这些还是选择候选/epoch 后的描述性诊断，不能作为未经选择偏差影响的确认性显著性证据。
诊断原文件：`outputs/launch_logs/numeric_foundation_v11_train_audit_20261015.json`。

可确定的是内部收益不够稳定，且“扩大近期训练预算就能解决底座问题”的方案未通过本轮
联合验收。不能仅凭这些结果把失败全部归因数据脏、时间漂移或某个具体模块；本轮还同时改变了
训练预算和固定重训过程，不能做单因素归因。
后续数值研究应先预声明训练期多时间起点的滚动验证，检验对季节/天气变化的稳定技能，
而不是再看当前外层改选 TCN、alpha 或早期 epoch。本协议已结束，不追加新训练。
原底座与全部负结果保留，Wiki 继续暂停，底座尚未达到“已稳定优化”的要求。

## 为什么先改底座

旧流程在每个 outer-train 内只用前 60% 拟合数值模型，随后 10% 选数值 epoch，
最后 30% 留给 Wiki。当前 fold1/seed2024 的原始底座虽然进行了微调，最终却选 epoch0，
外层表现几乎等同 Persistence：MAE 101.888844 kW、RMSE 155.471020 kW、R² 0.783236。
因此，不能说原始数值网络已经学到明显优于末值预测的能力，也不能把新 Wiki 的负收益
全归咎于脏数据。

最直接、可检验的问题是：数值底座没有使用最近的可用训练标签，而且小时间尺度任务
可能更适合直接预测相对最后功率的变化。近期数据一定有效、TCN 一定胜过 Transformer，
都还只是待检验假设。本轮同时改变训练预算与模型候选，不作为单因素因果消融。

## 有限、训练期内部选择

- 任务固定为 fold1、seed2024、72 小时历史预测未来 2 小时，stride3；六台风机均可供数值训练。
- 原 outer-train 前 90% 拟合，后 10% 内选；按时钟切分，清除目标跨界窗口。
  输入 scaler 仍沿用旧训练前缀，不用 outer validation 重新拟合。
- 五个预先声明候选：扩展近期训练的原 PromptTimeDART、零头初始化的直接残差 TCN，
  以及 alpha=1/10/100 的 77 维因果 lag-difference 残差 Ridge。
- 两个神经候选各完整 20 epoch；Prompt 复用旧预训练初始化，不重新预训练。
  TCN 保留绝对风速与控制量，只有历史功率相对最后观测值中心化；不使用未来协变量。
- 只从内部 MAE 与 RMSE 都不劣于 Persistence 的候选/epoch 中，按两者误差比的均值选一次。
  第0轮及浮点舍入级差异不能入选；没有合格候选时精确回退 Persistence。
- 仅将选中候选在完整原 outer-train 上重训一次；固定内选 epoch，保留原 20 epoch 的
  cosine 学习率曲线，不看外层挑最佳轮次。Ridge 仅按冻结 alpha 重拟合。
- 固定模型后才计算 outer validation 预测与指标，不根据外层表现追加候选、seed 或重训。

报告同时包含 MAE/MSE/RMSE/R²、相对 Persistence 的 Skill、容量容差命中率、逐步长误差、
训练曲线、预测图及时间块 bootstrap/Newey–West。R² 不是“预测正确百分比”，不会承诺 R²=0.9。
原 DLinear/PatchTST/Ridge 作为同窗口历史参考，但它们旧拟合预算更少，不能称为预算匹配、
充分调优的 SOTA 对照。

## 启动与恢复

```bash
KELMARSH_PYTHON=/home/XT/miniconda3/envs/timedart/bin/python \
  bash scripts/train/Kelmarsh_launch_numeric_foundation.sh
# 查询；只有中断的原目录才使用 --resume
bash scripts/train/Kelmarsh_launch_numeric_foundation.sh --status
```

结果指针：`outputs/logs/Kelmarsh/numeric_foundation_latest.txt`。
`prepared.json` 绑定旧完成记录、数据、初始化、实现和新划分；`selected.json` 固定内部选择；
各阶段 `trained.json` 绑定检查点/历史/图；完成产物哈希校验后才复用。
报告阶段失败只能补报告，不能重新训练完成阶段。仍按已有共享 GPU 策略限制本进程，
绝不停止或修改其他用户任务；服务器空闲时无需人为限速。

## 【最终评级】：C（完成后，精度验收失败）

运行、哈希和独立指标复核已完成，但新模型明显劣于原底座，不支持“底座已优化成功”。
未发现有证据的 P0；无 GRL/MMD/目标域无监督适配，不虚构此类问题。

## 【必须立即修复 Top 5】

1. 原底座与末值预测等效：先检验数值技能，暂停 Wiki 迭代。
2. 近期训练数据保留给 Wiki：本轮释放训练预算，但明确与旧模型预算不同。
3. 只看 MAE 或外层反复选 epoch：改训练期内联合 MAE/RMSE 选择，外层固定重训。
4. 初始化/回退/证书身份混淆：记录真实检查点指标，排除第0轮，旧 Wiki 证书禁用。
5. 过度宣称泛化：单 fold 且历史开发查看过；本轮不是独立确认，不自动恢复 Wiki 优化。

## 【需重跑的实验与需补充的单元测试】

只启动上述有限 v11，不重跑已完成 v1–v10，不刷外层参数。新增 10 项单测覆盖时间与标签
隔离、77 维特征、未标记推理、零头梯度、绝对风速、联合选择、原 Prompt 迁移后的训练期
阈值重校准、检查点复用/篡改检测，以及固定重训不调用外层评估。连同相关回归共 65 项。
部署时还必须验证服务器原数据/源检查点、原外层窗口及 scaler，完成后独立复核预测指标。
只有 MAE 与 RMSE 联合改善并随后得到冻结的跨时间段确认，才考虑在新底座上重做 Wiki。

## === 详细问题清单 ===

- **严重级别**：P1（既有结论风险）。
  **位置**：`data_provider/kelmarsh_hourly.py:26` 的分段、`scripts/train_kelmarsh_hourly.py:172::train_stage`。
  **触发条件及证据**：数值 fit 仅原 train 的前60%，f1 原模型最终 epoch0；其 MAE/RMSE 与
  Persistence 仅浮点舍入差异。**影响**：不能将原模型标为具备强数值预测技能。
  **修复**：`utils/numeric_foundation.py::partitions` 使用90%内部 fit、完整 train 固定 refit。
  **验证**：分段单测与服务器 `prepared.json`；性能效果仍待运行。**置信度**：高。

- **严重级别**：P1（实验归因边界）。
  **位置**：新协议 `comparisons`、`outer_refit`。
  **证据**：新旧数值训练时间范围、标签预算不同；TCN 也是新架构。
  **影响**：即使提升也不能单独归因某模块，更不是 Wiki 贡献。
  **修复**：保留历史结果，显式报告预算，后续另设匹配消融。
  **验证**：manifest 与原始对照窗口逐项一致性检查。**置信度**：高。

- **严重级别**：P2（已修复）。
  **位置**：`scripts/train_kelmarsh_numeric_foundation.py:157::neural_fit` 的 epoch0/完成记录。
  **证据**：真实初始化检查点不应被替换的 Persistence 指标冒充。
  **影响**：候选失败可能被“零增益”掩盖。
  **修复**：记录真实保存模型的指标，独立标明是否存在合格训练 epoch，epoch0 不能选中 refit。
  **验证**：联合选择测试、实际 Prompt 构造/推理及完成记录检查。**置信度**：高。

- **严重级别**：P2（已修复）。
  **位置**：`scripts/train_kelmarsh_numeric_foundation.py:89::make_model` 的加载后校准。
  **证据**：迁移 state_dict 会覆盖已有 regime 阈值；新 fit 数据应重新校准且只用 fit。
  **影响**：可能悄悄沿用旧时间段阈值，妨碍解释近期训练。
  **修复**：先迁移兼容参数，再用当前 fit 校准。
  **验证**：植入错误阈值的真实 Prompt 预训练构造测试。**置信度**：高。

- **严重级别**：P1（内部选择不等于泛化，效果验收失败）。
  **位置**：最终 `result.json` 的 `primary_joint_success=false`、内部 `selected.json` 以及只读时间诊断。
  **证据**：内部收益区间跨0、仅9/18天联合受益；固定重训后外层 MAE/RMSE 明显恶化，
  且外层/内部时间段都曾用于历史开发。
  **影响**：不能宣布新底座有效、独立泛化或恢复 Wiki 的条件已满足。
  **修复**：保留原模型，下一独立协议先检验训练期跨时段技能，不在旧外层重选候选。
  **验证**：本地原尺度独立重算已一致；新的多时间起点验证尚未执行。**置信度**：高。
