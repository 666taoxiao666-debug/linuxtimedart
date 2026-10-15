# Kelmarsh v11：先优化数值底座，暂缓 Wiki

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

## 【最终评级】：C（运行前）

实现可进入有限探索，但目前没有 v11 准确度结果，不足以支持“底座已优化成功”或论文结论。
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
  **位置**：`data_provider/kelmarsh_hourly.py` 的分段、`scripts/train_kelmarsh_hourly.py::train_stage`。
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
  **位置**：`scripts/train_kelmarsh_numeric_foundation.py::neural_fit` 的 epoch0/完成记录。
  **证据**：真实初始化检查点不应被替换的 Persistence 指标冒充。
  **影响**：候选失败可能被“零增益”掩盖。
  **修复**：记录真实保存模型的指标，独立标明是否存在合格训练 epoch，epoch0 不能选中 refit。
  **验证**：联合选择测试、实际 Prompt 构造/推理及完成记录检查。**置信度**：高。

- **严重级别**：P2（已修复）。
  **位置**：`scripts/train_kelmarsh_numeric_foundation.py::make_model` 的加载后校准。
  **证据**：迁移 state_dict 会覆盖已有 regime 阈值；新 fit 数据应重新校准且只用 fit。
  **影响**：可能悄悄沿用旧时间段阈值，妨碍解释近期训练。
  **修复**：先迁移兼容参数，再用当前 fit 校准。
  **验证**：植入错误阈值的真实 Prompt 预训练构造测试。**置信度**：高。

- **严重级别**：NEEDS-EVIDENCE（结果未产生）。
  **位置**：最终 `result.json` 和各报告 `raw_predictions.npz`。
  **证据**：本地功能测试不代表服务器性能；外层/内部时间段也曾用于历史开发。
  **影响**：不能宣布统计显著、独立泛化或恢复 Wiki 的条件已满足。
  **修复**：先跑有限协议，核对窗口/标签与独立指标，再决定冻结确认。
  **验证**：相同外层 cohort、原尺度独立重算及时间块区间。**置信度**：高。
