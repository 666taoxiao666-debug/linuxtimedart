# X3-code R1/2：第98次代码审查（2026-11-18）

结论：暂不放行（C，待修复复审）。5项确定缺陷中，B1–B3已按授权最小修复，B4的时间边界已修；B4剩余信息预算/特征集和B5仍阻断。未上服务器、未运行真实模型。

- R1-B1【阻断】【确定缺陷】分组按机组数设置batch，未计14个协变量，Chronos内部会拆组 —— run_x3.py:255–259、281–283；每机15条序列，6机batch=6实际每批仅1机，不能检验P1。
  【已按最小修法先行】cross_learning调用按本组全部目标及协变量行数设置batch，仍一次一个起报组；模拟SDK拆批的回归测试验证6机90行不拆组。修后位置scripts/run_x3.py:255。
- R1-B2【阻断】【确定缺陷】冒烟只保留validation前40个起报组，V2全部为空 —— run_x3.py:560–565、619–620、887–900；三场V2分别在7月/10月开始，report必然空数组或统计失败，正式链被门禁挡住。
  【已按最小修法先行】validation分别保留V1/V2前40个完整起报组；测试集仍保留前40组，正式范围不变，双切分非空测试通过。修后位置scripts/run_x3.py:587。
- R1-B3【阻断】【确定缺陷】选择元数据和失败分支没有闭合身份校验 —— run_x3.py:500–517、670、734、835；改selection.json的赢家而保持预测/registry不变不会被拒绝，任意failed.json可使q2_state跳过Q2。
  【已按最小修法先行】读取选择文件时核对完成回执哈希，失败分支核对launch身份，拒绝完成/失败并存；正式启动核验冒烟产物哈希；固定旧LUG/N_v1的3seed及none标签，锁定实际包版本2.3.2。
  回归覆盖选择文件篡改、错误身份失败回执、完成/失败并存、冒烟产物篡改；修后位置scripts/run_x3.py:488、512、524。Q1采用已登记完成回执内的check。
- R1-B4【阻断】【确定缺陷】Q4的观测预算尚未覆盖上游缺测填充值先验，且树特征漏掉方案要求的历史GFS —— training/build_inputs复用chronos_inputs在缺测处保留冻结填充值；Q4复用Q1输出，physical_features却已把缺测上游置NaN。
  前者不能靠逆缩放消除：目标多年的填充统计在k个月之外，须明确外部先验或另做严格掩码推理；后者PHYS_NAMES无历史GFS、也未读取逐小时历史，需Claude统一逐列信息集后修改。
  证据：scripts/prepare_g1_batches.py:376–392及output/g1/preparation_78/prepared/Kelmarsh/prepared.json记载填充值拟合段2016-05-01至2021-01-01，超出Q4的1个月预算；scripts/run_x3.py:219、226、830没有历史GFS列/读取。
  【时间边界已按最小修法先行】原budget_mask/es_split减1小时，允许2021-01-01T00标签进入严格早于D的训练；改按真实标签时间比较（scripts/run_x3.py:202、210及Q4末端断言），边界回归通过，旧X2日历切分未改。
- R1-B5【阻断】【确定缺陷】Q2把批准的全量拟合段微调改成最多20000窗口，尚未作为预算变更登记 —— STAGE_PLAN:49与run_x3.py:680–685/config q2.train_windows冲突；不能把抽样子集称全量。
  单序列训练本身不是泄漏：可作为事先声明的单序列微调臂，但不能说微调了Q1的分组赢家；请Claude明确预算及命名后修订，不由审核方替选。
  当前位置：configs/x3_protocol.json:29、scripts/run_x3.py:720与docs/STAGE_PLAN.md:49；DECISIONS实现说明虽已列20000，仍不能同时声称全量。需统一“全量”或“固定抽样预算”，不擅自扩大拟合。

其他核对：同起报组逐次调用、历史截止与乱序检查保留；Q2先做三场预检再冻结full/LoRA，不在正式失败后换模式；自定义MAE选点与单序列推理一致，不能从本地测试推定实际GPU可运行。
- R1-S1【建议】【不确定】Q2应在运行回执记录实际梯度累积和每批变量数；Chronos的batch_size不是窗口数，避免误述有效训练预算。不增加实验臂。
- R1-S2【建议】【确定缺陷】status_x2的x3状态只用kill -0可能把复用PID误报为运行中；正式监控应核对完整命令，但本项不改变结果计算、不阻塞代码复审。

验证：原tests/test_x3.py + tests/test_x2.py共40项通过；新增7项回归后47项通过（25+22）。Git Bash对launch_x2.sh/status_x2.sh的bash -n通过；未运行Chronos/LightGBM实际冒烟或正式链。
证据置信度：B1–B5高（代码反例、已有元数据或方案直接冲突）；真实SDK分组效果和GPU资源需求仍须Cursor冒烟验证。以上旧行号描述审前反例，修后定位另列。
交回：Claude在X3-code R2逐条处理B4/B5并复核直接修复；通过前没有可交Cursor启动的提交号。本轮为代码第1轮，不重开已通过的方案第3轮。

证据来源：锁定版本官方源码 https://raw.githubusercontent.com/amazon-science/chronos-forecasting/v2.3.2/src/chronos/chronos2/dataset.py （batch按目标与协变量数累计）；同版本pipeline.py（cross_learning仅在内部batch内合组）。
