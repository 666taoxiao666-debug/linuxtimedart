# 风电期刊模型对照与实测结果

2026-11-23。给 Claude。已完成的部分来自 X2 的冻结预测。新增实验的分组、预算、复用条件和判读方式见[三场对照操作稿](../../output/thesis_materials/wind_baseline_experiment_plan.md)，待审核。

## 哪些实验已覆盖

本地核对了 X1 的持续法、GFS 加逐机组功率曲线、LightGBM、Chronos-2 两种上下文，以及 X2 的 CNN、BiLSTM、混合堆叠、TCN-GRU 和域对抗版。完整输入的树模型在 X2 开发段三个风场均取得最低 MAE。已有 X2 调参预算随架构而异，结果支持项目配置下的比较。

信息消融覆盖本地、加上游、加 GFS、两者都加，以及 GFS 不同延迟。X5 的 f018 截断会使延迟同时改变未来预报覆盖，仅凭延迟实验无法识别固定覆盖下的纯老化效应。新增外部模型可以检验同一现象能否在其他方法上出现。

通用时序模型也有现成对照。T1 在 Kelmarsh 2021 验证段对 TSMixer、iTransformer、TimeXer 各搜索 60 次，冻结后按五种子平均预测评分。两个风场的调参后 MAE 如下（kW，依次为 TSMixer、iTransformer、TimeXer）：Kelmarsh 2022 为 157.12、183.56、179.05；Penmanshiel 2021 为 177.54、201.05、200.92。此比较采用本场和上游观测、共同窗口，属于回顾性实验。当前完整版另含 GFS，三场同输入比较仍缺这一行。[正式结果](../../output/t1_tuning/RESULT_SUMMARY.md) · [调参协议](../../configs/t1_tuning_protocol.json)

E1/A2 另做过有无上游的配对消融。Kelmarsh 的 iTransformer 加上游后 MAE 降低 7.26 kW（95% CI 4.35–10.21），Penmanshiel 两折的区间均跨零。若主张上游收益能推广到 Transformer，应保留这个跨场差异；若主张全输入 LightGBM 在三场优于 iTransformer，需要相同 GFS 可用周期、相同起报窗口和同等选参机会的直接比较。[上游消融记录](../../output/e1_baselines/a2/RESULT_SUMMARY.md)

## 已实测的 Applied Energy 近邻

Tang 等（2024）使用按分布差异确定的时间域、TCN-GRU 和域对抗训练。项目的 DATCNGRU 在拟合期按时间等长划域（见 `models/LegacyBases.py` 的 `time_domains`），保留 TCN-GRU 和域对抗学习。论文应写成“参照 Tang 等的域对抗 TCN-GRU 改编”，分域差异影响对原论文的归因。[论文](https://doi.org/10.1016/j.apenergy.2023.122266)

复算使用 `output/x2/x2_0f24875/` 的正式 A/B 注册表和 V2 预测，逐场验过启动身份、注册表及所用文件 SHA，核对机组、起报时间、目标时间、真值和功率锚点。每臂三个 seed 先平均绝对误差，四个步长等权。按起报时刻成簇、保留窗口权重，移动块长 8，5,000 次、随机种子 2024。以下为逐场点对点比较的 95% 区间，无跨场多重校正。

- Kelmarsh，8,249 窗口。LightGBM 132.05，域对抗 139.97，MAE 降低 7.92 kW，95% CI [4.70, 10.95]；相对降低 5.66%。
- Penmanshiel，16,176 窗口。LightGBM 176.28，域对抗 193.10，降低 16.82 kW，95% CI [11.94, 21.72]；相对降低 8.71%。
- Hill of Towie，15,103 窗口。LightGBM 186.39，域对抗 215.31，降低 28.91 kW，95% CI [23.89, 34.56]；相对降低 13.43%。

V2 是已经使用过的开发段。以上为重新计算的同窗口比较，未增加拟合。三场 MAE 均已对回 `report/dev_decisions.json`。同次 X2 的四个已暴露风场年，按原 `report/results.json` 的 seed 平均损失，完整输入 LightGBM 为 145.34 / 160.84 / 194.10 / 186.73 kW，域对抗改编版为 152.77 / 177.76 / 220.94 / 210.97 kW。顺序为 Kelmarsh 2022、Penmanshiel 2021、HoT 2024 和 2025；这里只读取汇总。

## 建议 Claude 登记的对照

R1-P1【提案】Bouche 等，Renewable Energy（2023）的直接功率 Nyström 核岭回归。原文把当地观测与 ECMWF 结合，逐 10 分钟预测至 4 h，直接预测功率的非线性方案最贴近本题。附录 B 给出高斯核、300 个采样点和核宽度、正则项范围。英国逐机组、逐小时、GFS 的改编须写明。[论文](https://www.sciencedirect.com/science/article/abs/pii/S0960148123006201) · [作者全文与附录 B](https://bouchedimitri.github.io/files/pdf/nowcast_to_short_term.pdf)
建议先在三场按 GFS 延迟 6/18 h × 有/无上游配对，四种条件均匹配相应延迟的拟合与评估，复用 X5 同条件 LightGBM 基准。主对比是各条件的 MAE 和上游使 MAE 降低多少，另算两种延迟之间增益之差的配对区间。
核中心和标准化只从拟合段确定，核参数用 V1 选；Claude 在运行前固定种子、采样、超参搜索与时间预算，记录与原文 30×30 网格的差异，开发 V2 全量报告。一个开发窗口的四步风速预报只来自起报时按规则已可用的周期，两个模型共用相同缺测与时效标记。

R1-P2【提案】Wang 等，Energy（2026）的 DeepWind-Base，风电专用预训练模型。原文评估包括 1–12 h，作者发布推理代码和 890M 基础版权重。拟固定一个版本、三场零微调，用预测中位数作 MAE 点预测，与同可见历史的对照配对，另列完整输入 LightGBM 的业务精度。[论文](https://doi.org/10.1016/j.energy.2026.141793) · [作者代码与输入约定](https://github.com/Hexian-2001/DeepWind) · [模型卡](https://huggingface.co/Hexian-2001/DeepWind1.0-890M)
预检记录代码/权重版本、预训练对三场的覆盖、小时步长、真实上下文、风速高度、功率单位与分位数顺序。作者示例是 5 分钟；优先使用接口能正确表达的逐小时历史，禁止把逐小时值复制成 5 分钟。原预训练场站身份未查清时，只声称“未在本站微调”，单列重合风险。
输入可先选严格匹配的功率历史，或明确可映射的本地变量；同键同历史长度的轻量对照必须具有相同可用输入。长上下文如需另测，应固定长度并向比较方提供相同历史。原接口的机舱高度风速与 GFS 10 米风速分列，不把 MIDAS 手工特征放入异义通道。

R1-P3【提案】首批与 R1-P1 并列纳入 Cabin 及 IEEE TSG 缺测自适应方法，R1-P2 的 DeepWind 作为后续独立选项。Cabin 比较同样 24 小时历史下的绝对精度，TSG 比较原 X5 混合延迟条件下的上游收益。最小验证分别检查历史通道对齐、真实零值与缺测零值的区分。正式预算及文件范围见[操作稿](../../output/thesis_materials/wind_baseline_experiment_plan.md)。

## 文献筛选与交付口径

- Dowell 与 Pinson 的稀疏向量自回归发表于 IEEE Transactions on Smart Grid（2016）。原任务是在澳大利亚 22 个风场做 5 分钟超短期概率预测。若论文转向多机组空间建模，可另立对照；当前由多风场改到同一风场内多机组，还需对等提供邻机历史。[作者全文](https://backend.orbit.dtu.dk/ws/portalfiles/portal/110321945/dowellpinson2015_vstsparse_final.pdf)
- Energy（2026）的 CGForecast 引入实时雷达和区域 NWP 场。现有三场输入缺少对应雷达资料，适合作为不同信息条件下的相关工作。[论文](https://doi.org/10.1016/j.energy.2026.141367)
- 两组新增对照均待 Claude 决议。正式输出逐场、逐步和汇总 MAE/RMSE/R²、样本数、时间范围、上下文和输入、耗时与复算身份，主指标为 MAE。沿用原始标签、缺测规则和冻结时间键；V2 结果定位为开发期比较，新对照不得据已暴露评估年选模。正式执行沿现有分工由 Cursor 运行。

## 11 月 23 日补充检索：一区和领域顶刊的候选模型

武汉大学图书馆公开目录将以下五种期刊均列为中科院工程技术大类一区。该目录另列 JCR 和小类分区，学位认定以学校采用的年度版本为准。[Renewable Energy](https://topj.lib.whu.edu.cn/show.asp?cat=sci&id=8240) · [Energy](https://topj.lib.whu.edu.cn/show.asp?cat=sci&id=218) · [Applied Energy](https://topj.lib.whu.edu.cn/show.asp?cat=esi&id=944) · [IEEE Transactions on Smart Grid](https://topj.lib.whu.edu.cn/show.asp?cat=sci&id=6116) · [Advanced Engineering Informatics](https://topj.lib.whu.edu.cn/show.asp?cat=esi&id=269)

**首选的三项对照**是 Bouche 等的直接功率核岭回归、Cabin 和 Stratigakos–Andrianesis 自适应鲁棒模型。DeepWind 单列为风电专用基础模型。以下是供 Claude 审阅的文献和适配依据。

- **Cabin，Wu 等，Energy，2025。** 输入历史功率及环境变量，用环境表示模块和协同调整模块融合。作者提供模型、训练和使用示例；逐机组过去 24 h 到未来 1–4 h 可直接按同样本键设计。首选双方都只使用相同本场及滞后上游历史；若要接入起报时已知的未来 GFS，先把有效时刻和通道表示写清，作为额外的同信息适配。上游站的未来实测不可作模型输入。[论文](https://doi.org/10.1016/j.energy.2025.137753) · [作者实现](https://github.com/xiufengliu/cabin-wind-forecasting)
- **按可用特征调整的鲁棒模型，Stratigakos 与 Andrianesis，IEEE Transactions on Smart Grid，2026 卷期。** 原文预测提前 15 分钟到 4 小时的风电功率，并公开线性和神经网络版本及学习缺测模式分区的实现。适合按 X5 的 6/18 h 两端和有/无上游比较。对时效越界导致的缺失传真实掩码，对仍存在的旧预报保留数值和实际年龄；两模型使用同一份输入。原文证据是输入缺失，GFS 延迟是本项目迁移。混合延迟训练和每个延迟单独训练的比较须分别对齐。[期刊论文](https://doi.org/10.1109/TSG.2025.3620957) · [作者代码](https://github.com/akylasstrat/wind-forecast-missing-data)
- **KSC-TFT，Energy，2025。** 按历史功率形状聚类风机，用 TFT 对各簇建模，可结合 NWP。原任务为 15 分钟分辨率的 0–4 h 预测，还会汇总簇预测得到全场总功率。适合第三方查问机组异质性时追加。聚类只用拟合时段，并先核对输出粒度；如果只能得到簇级或场级预测，比较双方聚合后的同一批机组。作者论文已定位，本次未确认作者实现。[论文](https://doi.org/10.1016/j.energy.2025.137995)
- **Wang 等的时空图方法，Advanced Engineering Informatics，2025。** 原文结合异常检测和校正、谱聚类、两层自适应图卷积循环网络，使用 134 和 200 台风机的数据并汇总区域功率。对当前 6–21 台风机，更适合作为场级敏感性比较。保留原始标签、给对手相同的邻机历史；省略原文校正步骤应列为改编。本次未确认作者实现。[论文](https://doi.org/10.1016/j.aei.2025.103137)
- **CPLLM-WPF，Applied Energy，2025。** 用多尺度分解、序列重编程、任务提示和 LoRA 做风功率预测。适合作为大模型补充线。复现时分解须限制在各起报时刻已经观测到的序列，逐窗口的结果不可借用整段测试序列作预处理。本次核实了期刊论文，未确认对应的作者代码。[论文](https://doi.org/10.1016/j.apenergy.2025.126912)

三场新比较统一保留单机组原始 kW、相同起报键、1–4 h 分步及汇总 MAE/RMSE/R²。对于输入或目标不同的研究，配同输入对照或同粒度对照。计算 18 h 相对 6 h 的上游净增益时，四组预测同窗配对、按起报时间成簇，报告区间；已暴露的年份标为探索性复核。新增候选均未在本项目运行。
