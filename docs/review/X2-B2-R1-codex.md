# X2-B2 R1 代码审查

结论：通过，无阻断；仅审第94次待审改动，不连接服务器、不干预A2。

## 核对结果
- EarlyStopper：热身轮次不候选、不累计patience；正式40轮配置仅epoch17起候选，TCN/DA窗口一致。
- select_after=0保留epoch0候选、严格小于才改善及连续patience停止，与既有选择语义一致。
- λ网格为{0.05,0.5}；正式DA首个候选λ分别为0.05/24、0.5/24，均大于0；λ仅由V1跨seed均值选择。
- 单轮smoke允许epoch0被选中，selected_lambda=0仅属冒烟例外，不是正式DA有效性证据。
- 训练x/y/domain常驻device，torch索引保持default_rng(seed+epoch)排列；损失、优化器、调度、梯度裁剪未改。
- bad张量累积各batch损失有限性，轮末一次读回；任何非有限损失均在验证和保存前抛错。
- predict_nn分批推理后统一回CPU；验证输入每次调用仍需上传，不应将此描述为验证数据跨epoch常驻。
- best_epoch对应selected_lambda；训练记录保留每轮λ、验证误差、用时与选中权重SHA。
- 全部B臂统一benchmark=True、deterministic=False，属于明示的新运行配置，不能声称逐位复现旧运行。

## 建议与限制
- R1-S1【建议】【不确定】实际显存和速度须由Cursor运行验证；全量设备输入与cuDNN工作区可能增加峰值，现有显存限制不保证可容纳，失败应交回而非自行改batch或窗口。
- R1-S2【建议】【不确定】多seed不能保证消除cuDNN非确定性；保留配置、环境和权重哈希，旧B与新B不得混用。

## 本地验证
- python -X utf8 -m unittest tests.test_x2 tests.test_legacy_bases：33项，28通过、5跳过，8.093秒。
- X2的22项及Legacy numpy的6项通过；5项torch测试因本地无torch跳过。
- 没有执行真实设备fit_nn、正式拟合、服务器状态查询或停止任务。
- Cursor下一步：新干净副本先按STAGE_PLAN做smoke；通过后运行B并执行首个TCNGRU epoch1 >60秒停止门禁。
- A2仍固定655e921原实例；A的同提交复现只能在A2结束且CPU空闲后执行，再生成report。
