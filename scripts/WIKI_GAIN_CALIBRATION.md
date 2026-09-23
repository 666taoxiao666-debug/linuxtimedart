# 已训练事件专家的收益与幅度校准

适用于 factorized Wiki 候选局部有益、神经门控整体仍负收益的情况。
不根据验证结果手工删除事件，也不重新训练专家或下载 Qwen。

## 方法及数据边界

1. 固定加载源运行的 `checkpoint_last.pth`，不加载可能回退到 epoch 0 的
   `checkpoint.pth`，也不按验证分数挑选候选专家轮次。
2. 重建原训练集最后约 20% 的门控校准区间。核对源运行的时间边界、窗口数、
   scaler 与数据哈希（源记录有哈希时强制检查）。不调用重新拟合趋势阈值的函数。
3. 按全局目标时间分成三个区块，排除预测标签跨区块的窗口；风机不分别随机分块。
4. 对每个事件/组合、历史趋势类别和预测步长，比较固定修正比例
   `[0, 0.25, 0.5, 1]`，使用与汇报一致的物理功率裁剪计算绝对误差收益。
   分数为区块平均收益减去 0.25 倍区块收益标准差，且需多数区块收益为正。
   每块默认至少 32 个可用窗口；趋势细组不足时回退到同事件/步长的全趋势统计。
   全趋势支持也不足或无正收益时不介入。该规则是经验稳定性校准，不是统计置信保证。
5. 推理只查询“当前可观测趋势 + 物理事件支持 + 预测步长”对应的固定比例/收益，
   选择预期收益最高的可用候选。没有未来标签输入，也没有验证集自动调参。
6. 策略先保存到新 checkpoint，再读取验证集报告原趋势、上一轮神经门控、新校准
   策略三个结果。新策略即使变差也保留原结果，不用基线成绩冒充新方法。

这不是全模型 OOF：原趋势模型/scaler 使用原训练集。旧门控也曾使用校准区间，
但本次不使用它的预测分数拟合策略，只使用冻结专家的候选输出。
语义专家保持不变，不把原始文本卡片重新训练。它是新的门控对照方案，不能声称
已证明其提升；验证集已参与多轮研发，最终结论仍需要独立评估。

## 服务器操作

```bash
cd ~/nuist/pythoncode/TimeDARTFirst
conda activate timedart
git switch codex/self-evolving-wiki
git pull --ff-only
git log -1 --oneline
python -m unittest discover -s tests -p 'test_wiki_gain_calibration.py' -v
```

测试 OK 后，启动一次即可（自动后台运行）：

```bash
FOLD=0 SEED=2024 bash scripts/train/SDWPF_launch_gain_calibration.sh
```

默认源目录来自 `outputs/logs/SDWPF/factorized_wiki_latest.txt`。
需要明确指定时，在命令前设置 `SOURCE_FACTOR_DIR=实际旧目录`。
程序只运行原训练集校准区间推理、拟合收益表和验证，不跑训练 epoch、不读测试集。
新日志为 `outputs/logs/SDWPF/YYYYMMDD/NNN_wiki_gain_calibration_参数_哈希/`，
独立指针不会覆盖原 factorized 运行指针。

查看状态（重新打开终端也可使用）：

```bash
cd ~/nuist/pythoncode/TimeDARTFirst
bash scripts/train/SDWPF_launch_gain_calibration.sh --status
```

完成后关注 `CALIBRATED_MAE_KW` 和 `GAIN_VS_TREND_KW`（正值才改善）。
`ACTIVE_POLICY_CELLS=0` 表示训练区间未支持任何有效修正，不是程序故障。
每个事件保留多少趋势/步长单元在 launch.log 的 `[GAIN]` 行中，完整比例、
区块窗口数和收益在 `gain_calibration.json`，三组验证指标在 `validation_metrics.json`。
策略 buffer 随 `checkpoint.pth` 保存，旧无策略 buffer 的 checkpoint 仍可加载。
不要把此策略 checkpoint 当作下一次专家校准的源；源必须是原训练后的神经门控 checkpoint。
