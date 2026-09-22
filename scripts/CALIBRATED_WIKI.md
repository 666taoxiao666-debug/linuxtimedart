# Calibrated Wiki：本轮修复与服务器操作

## 修复对象

旧结果 `BEST_EPOCH_COUNTS=0:1` 表示最后恢复训练前的趋势基线，
不代表每一轮预测都没有变化。保留这个回退机制，但另外报告训练后最好和最后一轮。

旧版 `intervention_floor=1` 加硬阈值选择，使预测损失不能直接训练收益门控；
原有收益回归、动作分类损失仍然有梯度，不是整个模型不学习。

本版本显式使用 `UTILITY_ADAPTER_MODE=calibrated_evidence`，旧模式不变：

1. 修正分支保留下降/平稳/上升专家及单事件、组合事件知识。
2. 增加历史风速、功率、平均桨距角最近 3/6/12 个采样点的
   最后值、均值、变化量、斜率和标准差，共 45 个物理量特征。
   仅支持本实验的 10 分钟采样，尺度分别为 25 m/s、额定功率、90 度。
3. 原训练集按全局预测目标时间分成前约 80% 和后约 20%；
   跨分界的预测标签窗口排除。前段训练修正，后段冻结修正、训练门控。
   历史输入可以回看分界前观测，验证与测试数据不参与这两阶段拟合。
  基础模型和 scaler 仍使用原训练集；这不是整个模型的 out-of-fold 训练。
   两阶段分别启动学习率周期，避免门控刚开始训练时学习率已衰减到接近零。
4. 门控训练用不介入/单事件/组合事件的可导 softmax 混合，
   验证与实际推理仍用硬选择。缺少物理证据的候选概率严格为零。
5. 收益目标改成固定训练 scaler 下的绝对误差减少量，
   不再除以每个点的基线误差或裁成 [-1,1]。
   `UTILITY_MIN_GAIN` 在新模式下单位为训练集目标标准差，不是百分比。
6. 记录硬选择、软选择和逐点事后最优候选的 MAE。
   最后一项使用真实标签，仅是诊断性误差下界，不是可部署结果，
   不用它选择 checkpoint。检查是否有候选收益可供门控学习。

修正幅度默认分别限于历史窗口功率标准差的 0.2 和 0.05 倍。
两个分支使用自然样本频率，不再按稀有工况增加单窗口权重；分支损失仍等权。

## 1. 拉取

只复制代码框中的命令，不要复制终端提示符或三个反引号。

```bash
cd ~/nuist/pythoncode/TimeDARTFirst
conda activate timedart
git switch codex/self-evolving-wiki
git pull --ff-only
git log -1 --oneline
```

## 2. 启动一次小实验

```bash
bash scripts/train/SDWPF_launch_calibrated_wiki.sh
```

自动后台执行 fold 0、seed 2024、预测 12 步；最多 6 轮，其中前 2 轮
训练修正，后 4 轮训练门控，patience=3，模块最大学习率 3e-5。
复用服务器 9 月 13 日 Wiki 预训练与 9 月 14 日趋势模型，无需重新下载 Qwen
或重新预训练。需要已有对应 `cv.env`、预训练 checkpoint、趋势 checkpoint
和 Wiki 向量文件。不删除或覆盖旧实验目录。

如果这些源目录变动，可显式设置 `SOURCE_CV_DIR`、`TREND_CV_DIR`。
已有旧进程不会自动结束；启动前自行确认不重复占用同一 GPU。

## 3. 查看进度和结果（新终端同样可用）

```bash
cd ~/nuist/pythoncode/TimeDARTFirst
bash scripts/train/SDWPF_launch_calibrated_wiki.sh --status
```

运行中显示日志末尾；结束后显示状态、总体指标和配对趋势比较。
所有日志仍保存在 `outputs/logs/SDWPF/YYYYMMDD/NNN_calibrated_wiki_<参数及哈希>/`。
完整参数在 `cv.env`、`log_meta.json` 和 checkpoint 的 `run_manifest.json`。

如需实时查看详细日志：

```bash
CAL_DIR="$(cat outputs/logs/SDWPF/calibrated_wiki_latest.txt)"
if [[ -n "$CAL_DIR" && -d "$CAL_DIR" ]]; then
    tail -n 80 -f "$CAL_DIR/launch.log"
fi
```

Ctrl+C 只退出 tail，不结束后台训练。汇报时优先提供 `--status` 输出及
`runs/f0_s2024/finetune.log`。

## 4. 如何判断，而不是只看是否胜过 Persistence

- `WIKI_GAIN_VS_TREND_KW > 0`：选出的模型比相同 fold/seed 趋势基线好。
- `EPOCH_ZERO_SELECTED=1`：仍回退到原模型，没有证明本次 Wiki 改动有效。
- `TRAINED_BEST_MAE_KW`：排除 epoch 0 和修正预热后，门控训练阶段最优 checkpoint 的 MAE。
- `LAST_MAE_KW`：最后一个门控训练轮的 MAE。
- `TRAINED_BEST_GAIN_VS_TREND_KW < 0`：训练后最好仍比原模型差。
- `FORECAST_GATE_GRAD_NORM`：首个门控批次的直接预测损失梯度；若该批无证据
  或修正恰为零，可以为零，需要结合全轮可用率及其他梯度日志判断。
- `WikiRoutingMAE(soft/hard/oracle)`：软路由、硬路由、事后误差下界；
  若下界都不比基线好，优先修候选；若下界好但硬路由差，优先修收益门控。
  下界明显更低只是可能空间，不保证这些收益能够由历史信息预测。

checkpoint 目录同时保存 `checkpoint.pth`（含 epoch 0 的正式选择）、
`checkpoint_best_trained.pth`（门控训练后最好）、`checkpoint_last.pth`（最后）。
准确路径在运行日志的 `FINETUNE_CHECKPOINT` 中。

先检查本次单折，不立即再花时间跑九次；确认训练机制和配对收益后，可运行：

```bash
FOLDS='0 1 2' SEEDS='2024 2025 2026' bash scripts/train/SDWPF_launch_calibrated_wiki.sh
```

以上是待验证的新实验方案。代码测试证明可执行、梯度及分界符合设计，
不等于证明服务器真实数据上的精度提升。
