# 独立事件候选与可训练语义适配：服务器操作

## 本版改变

- 保留旧模式，新入口设置 `UTILITY_FACTORIZED=1`、`calibrated_evidence`。
- 原趋势主干、预测头和原始文本向量冻结。每个事件拥有独立的三趋势修正专家，
  共享一个可训练的语义投影/历史查询层；该层属于 utility_event_adapter。
- 所有物理规则正证据、生命周期可靠度非零的事件均可成为候选，不再由旧冻结
  检索器的语义阈值/Top-K 提前排除。语义相似度改为可训练连续特征。
  这会改变候选覆盖率，必须与旧模式明确区分，不能说只改了学习率。
- 组合候选是物理支持 Top-K 事件修正的加权平均加独立有界增量。
  组合损失不反传到单事件专家。支持事件不足两个时不能选组合。
- 门控对每个事件及组合逐步长估计绝对 MAE 收益。训练软路由，推理硬选择；
  预期收益不足则回退到趋势预测。候选收益目标只在训练阶段使用标签。
- 训练集前约 80% 训练候选，后约 20% 训练门控，排除跨界目标窗口。
  沿用原趋势模型/scaler，不能称为全模型 out-of-fold 训练。
- 默认 3 轮候选 + 7 轮门控，最大学习率 1e-4、patience=7；候选幅度上限 .2、
  组合增量 .05，单位均为历史输入窗口功率标准差。原本的 epoch-0 回退保留。
- 每轮新增 `WikiFactor[事件名]`：物理可用窗口比例、该事件候选相对基线收益%、
  实际选中位置的收益%、变差位置比例%、选中位置数。n/a 表示没有相应样本，
  不是数值溢出。`WikiRoutingMAE` 的 oracle 现在遍历全部可用事件和组合，
  仅用于诊断，不能视为可部署结果。
- `WIKI_SEMANTIC_ADAPTER_GRAD_NORM` 检查语义适配梯度。零初始化预测头导致
  第一个优化步语义层可无梯度，因此在候选阶段第二个批次记录；无支持事件也可为零。
- 旧 prompt-removal 诊断不适用于新专家机制，显式报错，避免错误的“无 Wiki 增益”结论。

## 1. 拉取与检查

只复制代码框内命令，不复制终端提示符。

```bash
cd ~/nuist/pythoncode/TimeDARTFirst
conda activate timedart
git switch codex/self-evolving-wiki
git pull --ff-only
git log -1 --oneline
python -m unittest discover -s tests -p 'test_factorized_wiki.py' -v
python -m unittest discover -s tests -p 'test_calibrated_utility.py' -v
bash -n scripts/train/SDWPF_launch_factorized_wiki.sh
```

## 2. 启动一次（后台运行，不需要额外 nohup）

```bash
bash scripts/train/SDWPF_launch_factorized_wiki.sh
```

默认 fold=0、seed=2024、horizon=12；复用服务器 9 月 13 日的对应 Wiki 预训练
和 9 月 14 日趋势 checkpoint。需要保留旧 cv.env/pipeline.env、权重、Wiki NPZ。
源目录改变时显式设置 SOURCE_CV_DIR、TREND_CV_DIR。不会重新预训练或下载 Qwen。
不要同时重复启动该命令。不读最终测试集，不移除差 fold，不覆盖旧实验。

## 3. 查看结果（新终端也能直接执行）

```bash
cd ~/nuist/pythoncode/TimeDARTFirst
bash scripts/train/SDWPF_launch_factorized_wiki.sh --status
```

日志位于 `outputs/logs/SDWPF/YYYYMMDD/NNN_factorized_wiki_参数_哈希/`。
完整参数在 cv.env、finetune.env、log_meta.json、run_manifest.json。
独立指针 `outputs/logs/SDWPF/factorized_wiki_latest.txt` 不影响旧 calibrated 指针。

提取关键训练记录：

```bash
FACTOR_DIR="$(cat outputs/logs/SDWPF/factorized_wiki_latest.txt)"
if [[ -n "$FACTOR_DIR" && -d "$FACTOR_DIR" ]]; then
    grep -E '^Epoch:|Training phase:|FactorizedWiki|WIKI_SEMANTIC_ADAPTER_GRAD_NORM|FORECAST_GATE_GRAD_NORM' "$FACTOR_DIR/runs/f0_s2024/finetune.log"
fi
```

先看是否超过同折同 seed 趋势基线；epoch 0 被选中不代表新增模块有效。
只有新运行自己的覆盖率、候选收益和实际选择收益才能用于判断本次改造。
通过代码测试不等于服务器精度提升；此次是独立的新候选生成实验。
