# X3-Q2fix R1/2：第100次代码审查（2026-11-18）

结论：通过，0条阻断，2条建议。基线dee252b，只审本轮改动及直接SDK依赖；未上服务器、未安装依赖、未重跑冒烟或正式任务。

## 核对结果
- 候选预算通过：每窗口15行，ceil(microbatch/15)×accum均为18；256×1、135×2、90×3、45×6分别是18×1、9×2、6×3、3×6窗口。注册顺序先full全部候选，再lora全部候选。
- preflight_search按第一个三场都通过的组合冻结；某场失败即可排除该组合；mode_q2为每个组合/场新建目录并启动独立worker，候选选择不使用预测精度。训练前完成全场预检。
- fit_kwargs将gradient_accumulation_steps作为额外Trainer参数传入；Chronos 2.3.2 pipeline.py:245–295先设置默认1，再update(extra_trainer_kwargs)，不会覆盖成1。
- 梯度平均通过源码核对：本地回执output/x1/dependencies/chronos_ready_receipt.json记录transformers 5.23.0；其trainer.py:2098起只对labels键计数，本任务future_target没有labels，因此num_items_in_batch=None。
  当前未传compute_loss_func或loss_is_scaled_for_ga覆盖，training_step在1911–1920按current_gradient_accumulation_steps除损失，然后backward；Chronos2Trainer仅重写数据加载，不覆盖该逻辑。
  这是指定版本的源码结论，不是已在服务器完成的梯度数值等价实验；不能用“AdamW对缩放不敏感”代替损失缩放核验。
- 正式worker在on_train_begin核对实际microbatch与累积步数；candidate、Trainer设置、预算随冻结结果保存，preflight_decision登记哈希。窗口次数是规则推算而非去重实测。
- LoRA缺依赖立即不可用；run_fit把锁定SDK的peft回退警告转成异常，避免把full记为lora；依赖存在但导入失败也会报技术失败。本轮不安装peft，符合决议。
- Q2仍先按V1选上下文，再固定预算拟合、V1 MAE选点；原GPU上限0.7、学习率、步数、抽样窗口数和判定阈值未改。Q3失败回退逻辑未扩张为跳过完整冒烟门禁。
- 协议升至v3；新提交新目录完整x3smoke，旧dee252b失败目录保留；不存在覆盖旧冻结运行的修改。前阶段绑定检查修复的事后复核及预算、偏差方向措辞已落实。

## 建议（不阻塞）
- R1-S1【建议】【不确定】18窗口/优化步保持样本预算，不保证与单大批量逐位数值等价 —— 随机数消费、浮点累加及缺测损失加权可能不同；论文写“等样本预算的资源适配”，避免“只改变显存、优化轨迹完全相同”。
- R1-S2【建议】【不确定】源码结论依赖transformers 5.23.0；Cursor新冒烟需记录当前版本、冻结candidate及实际累积设置，并检查V1回调和保存重载成功，不把只通过资源预检视为完整冒烟成功。
  若服务器版本与回执不一致，先交回，不自行升级依赖或按其他版本推定损失行为。若以后安装peft，再核验适配器类型、可训练参数及重载结果。

## 验证和交接
- python -m pytest tests/test_x3.py tests/test_x2.py -q：54 passed（32+22，68.45秒）；测试覆盖等窗口预算、三场统一设置、候选短路和LoRA回退拒绝。
- git diff --check通过；无本轮执行代码代改，只新增本审查及更新HANDOFF。
- 可交Cursor本次提交：先x3smoke；通过后x3gpu；正式q1完成后x3q4可并行q2；q2结束且q4完成后x3final。候选搜索是已登记预检，不授权失败后再次启动整条链。

源码证据：
- https://raw.githubusercontent.com/amazon-science/chronos-forecasting/v2.3.2/src/chronos/chronos2/pipeline.py
- https://raw.githubusercontent.com/amazon-science/chronos-forecasting/v2.3.2/src/chronos/chronos2/trainer.py
- https://raw.githubusercontent.com/huggingface/transformers/v5.23.0/src/transformers/trainer.py
