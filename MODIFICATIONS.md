# TimeDART 预训练与微调修复说明

## 本轮三项问题复核

| 原判断 | 复核结论 | 处理方式 |
|---|---|---|
| 预训练未迁移 `sos_token` | 成立 | 已加入 checkpoint 保存白名单和严格加载必需项 |
| 分类早停仅依据 Accuracy | 成立 | 已默认改为验证集 Macro F1，并保留 Accuracy 可选项 |
| 分类随机分层逻辑与预测时间切分发生交叉 | 原表述不成立 | 两者当前用于不同数据集；删除“已经交叉泄露”的结论，并增加任务—数据集兼容性校验防止误用 |

训练 DataLoader 对已经完成时间切分的训练窗口执行 shuffle，只改变批次顺序，
不会让验证或测试时段进入训练集，因此不属于时序泄露。

## 已修复的关键问题

1. `SDWPF` 使用独立的 `Dataset_SDWPF`：
   - 按全局时间戳执行 70%/10%/20% 切分；
   - 每台风机独立排序，窗口不会跨 `TurbID`；
   - 窗口不会跨真实时间缺口；
   - `TurbID`、`Day` 仅作为元数据，不进入模型；
   - `MS` 模式下 `power` 固定放在最后一个通道；
   - 异常值先标记，在同一风机的连续片段内最多修复 6 个连续点；无法可靠修复的长异常段会切断窗口；
   - 标准化器仅使用训练时段拟合。
2. checkpoint 使用“数据集/模型/结构签名”独立目录，不同模型和结构不会再互相覆盖。
3. 显式 `--load_checkpoints` 优先级最高；路径不存在会直接报错。
4. 微调默认要求预训练 checkpoint；只有显式使用 `--allow_random_init` 才允许从头训练。
5. `TimeDART_v2` 的训练、验证、测试现在都只接收历史 `batch_x`，不再把真实未来 `batch_y` 输入模型。
6. `SimMTM` 和 `TimeDART_v2` 恢复预训练权重加载，并对编码器形状和缺失参数严格检查。
7. `PromptTimeDART` 的工况标签只从指定的功率通道生成，每个样本一个标签；标签在实例归一化之前计算。
8. Prompt 的 `regime_predictor` 和 `soft_prompt_generator` 会保存、加载，并在微调预测头之前继续调制编码特征。
9. 训练期间只评估验证集；测试集只在最佳模型加载后评估一次。
10. 验证/测试使用 `--eval_batch_size`，不再固定为 batch size 1。
11. `--use_amp` 已连接到 PyTorch 自动混合精度；CPU 加载最佳模型不再硬编码 `cuda:0`。
12. HAR、EEG、Epilepsy 的验证数据从训练文件中确定性分层划分，不再复用测试集；分类卷积输出长度按真实公式计算。
13. Linux shell 脚本已统一为 LF 换行；根目录 `fix_data.py` 已恢复为有效 UTF-8 入口。
14. 预训练 checkpoint 现在保存并严格加载 `sos_token`，不再在微调时将起始标记随机初始化。
15. 分类模型默认按验证集 Macro F1 选择 checkpoint 和早停（可用 `--classification_early_stop_metric accuracy` 切回 Accuracy）；Accuracy 与 Macro F1 均按完整数据集计算，不再对各 batch 的指标做算术平均。
16. 分层随机切分函数明确限定为分类数据；数据入口新增任务—数据集兼容性校验，预测任务无法误用分类数据加载器。预测训练集内部仍可打乱窗口顺序，这不会改变已完成的时间边界切分，也不会引入未来数据。

## SDWPF 实际检查结果

使用附件中的 `sdwpf_fixed.csv`、`input_len=336`、`pred_len=96`、步长 6：

- 处理后有效行：4,339,087；
- 连续风机时间片段：11,197；
- 训练窗口：144,086；
- 验证窗口：20,561；
- 测试窗口：50,876。

这些窗口均不会跨风机或跨非 10 分钟时间缺口。

## 运行方式

```bash
pip install -r requirements.txt
bash scripts/pretrain/SDWPF.sh
bash scripts/finetune/SDWPF.sh
```

若要使用普通 `TimeDART`，同时把两个 SDWPF 脚本中的
`--model PromptTimeDART` 改成 `--model TimeDART`，保证预训练与微调配置一致。

仅测试已微调模型时：

```bash
python run.py ... --task_name finetune --is_training 0 \
  --finetune_checkpoint /absolute/path/to/checkpoint.pth
```

`TimeDART_v2` 的 Qwen 路径仍需通过 `--llm_path` 指向本地模型；
`d_model` 必须与 Qwen backbone 的 `hidden_size` 一致。
