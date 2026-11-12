# G1 独立创新评估与文献证据

评估日期：2026-11-12。对象：用户本轮给出的 G1 描述。结论与实验建议供 Claude 讨论采用，不构成修改已冻结协议或启动实验的批准。

## 1. 结论

**建议：收窄创新主张、修订识别方法后，开展小规模验证。问题有价值；“新的通用预测机制”尚不成立；“可辨识的上游相位修正转移”有研究空间。**

当前最有价值的假设是：某些移动天气结构的 NWP 相位误差在空间上连贯，已到达上游站的观测能够比风场自身观测更早暴露下游的相位误差。需要证明提前量和可转移性，不能仅凭最终功率 MAE 改善解释为该机制成立。

用户自评“通用思路不新、属于组合创新”方向正确，但遗漏了直接的跨站预测误差传播文献。公平实验是贡献可信度的来源，本身不构成新的算法机制。softmin、少参数和同次预报也分别属于已有计算工具、结构约束和数据一致性要求。

按六维概念量表作诊断：问题价值 4/5；相对近邻的新颖性 2.5/5；核心洞察 3.5/5；机制自洽性 3/5；结构简洁性 4/5；能源领域适配性 4/5。权重分别为 20%、25%、20%、20%、10%、5%，加权 3.325/5；覆盖 100%，其中新颖性结论置信度中等。这个分数不评实验完成度，也不代表录用概率。

单独讨论新颖性，约 5/10 是可接受的主观定位；5–5.5/10 只有在“显式相位转移确实区别于通用残差转移”得到支持时才有依据。没有可校准依据支持 Q1 60–70%、Q3+Q4 10–20%、最终 10% 等概率，不建议用于决策。

背景中的 MAE、110 个程序及既有负结果均为用户提供，本次未复算。有限编码器搜索打平，只说明已试方法没有找到稳定增益，不能证明编码形式达到信息论上限；真实未来风速实验说明未来风信息有价值，不能给出 GFS 的可达收益。

## 2. 最接近的文献与真实差异

**S1｜Liu et al., 2023, Applied Energy 336:120815，方法论文。** [出版社页面](https://www.sciencedirect.com/science/article/pii/S0306261923001794)，DOI：10.1016/j.apenergy.2023.120815。
- 已核查出版社摘要、贡献描述与章节摘录：TLA 使用实测风速与 NWP 序列相似度识别可能的 lag，再融合 lag 修正后的 NWP；还研究了短时滞后的惯性。
- 覆盖 G1 的“序列匹配 → lag 权重 → 修正 NWP → 下游预测”核心结构。剩余差异主要在异地上游证据及其提前转移。
- M_self 只能称为自站消融或受 TLA 启发的对照，不能自动当作原论文复现。全文公式和训练配置尚未完整获取。

**S2｜Tastu et al., 2011, Wind Energy 14(1):43–60，方法论文。** [DTU 作者稿](https://backend.orbit.dtu.dk/ws/files/6380046/tastuetal09_rev.pdf)，DOI：10.1002/we.401。
- 已阅读作者稿摘要、引言及 CP-TARX 章节：通过邻近地区历史预测误差、风向分区与风速条件化系数建模误差的时空关系。
- 直接覆盖“异地已发生误差可提供目标地点未来误差信息”。其预测对象是功率误差，未据此确认其估计了 G1 所定义的 NWP 时间平移量。
- 因此必须加入跨站残差回归/传播对照；仅与原始数据拼接或小 MLP 比较不够。

**S3｜Girard & Allard, 2013（2012 在线），Wind Energy 16:999–1012，分析方法论文。** [原文 DOI](https://doi.org/10.1002/we.1527)。
- 核查可检索的原文摘要与引言：利用相关结构局部极大值和平面波建模分析功率预测误差的传播方向和速度，并讨论对即将出现的锋面或相位错误的预警价值。
- 它削弱“首次想到误差从上游传播”的主张；没有据此确认其执行了跨站 NWP 轨迹重定时。

**S4｜Tastu et al., 2014, IEEE Transactions on Smart Grid 5(1):480–489，方法论文。** [DTU 正式书目](https://orbit.dtu.dk/en/publications/probabilistic-forecasts-of-wind-power-generation-accounting-for-g/)、[作者全文](https://www.pierrepinson.com/docs/tastuetal13_probfore.pdf)，DOI：10.1109/TSG.2013.2277585。
- 已核查作者全文的时空修正定义：利用其他风场已记录的误差调整目标场预测，并扩展为概率预测。
- 这不是仅提出传播直觉的文献，而是已有校正应用；G1 要证明“校正相位”优于“直接预测校正量”。

**S5｜Peduto et al., 2026, Journal of Geophysical Research: Machine Learning and Computation，ORCA，方法论文。** [开放全文](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JH001174)。正式首次发表 2026-07-07，DOI 年份为 2025，不能混写。
- 已核查摘要、数据、方法及实验部分：以分散位置的近期观测—GFS 配对为上下文，通过注意力修正任意目标坐标的海面风预报。
- 与 G1 的信息流高度接近；其方法不是显式估计并转移时间平移。研究场景、观测网络和校正对象也不同。
- 应列为现代近邻；小型注意力适配可以检验通用跨站学习能力，但不应把自行简化版冒充 ORCA 原方法。

**S6｜Yang et al., 2025, Expert Systems with Applications 263:125724，方法论文。** [出版社方法摘录](https://www.sciencedirect.com/science/article/pii/S0957417424025910)，DOI：10.1016/j.eswa.2024.125724。
- 题名为 A day-ahead wind speed correction method: Enhancing wind speed forecasting accuracy using a strategy combining dynamic feature weighting with multi-source information and dynamic matching with improved similarity function。
- 已核查摘要、引言和方法摘录：通过多风场动态相关性加权、分解与动态匹配校正日前风速。多站动态加权结合 NWP 修正已经存在；摘要不能证明它与 G1 的在线相位识别完全相同。

**S7｜Schuhen, Thorarinsdottir & Lenkoski，RAFT，2019 在线/2020 卷期，QJRMS，方法论文。** [作者全文](https://arxiv.org/html/1910.05101)，DOI：10.1002/qj.3718。
- 已阅读 §3.2 与式 (9)：用同条已发布预报轨迹中已经实现的误差，回归预测后续误差；显式计入观测处理延迟。
- 可借鉴实时残差更新及延迟审计。原应用是本站温度，不能描述成原文已经做上游风速时间平移。

**S8｜Ravela, Emanuel & McLaughlin, 2007, Physica D 230:127–145，方法论文。** [MIT 作者全文](https://texmex.mit.edu/ftp/pub/emanuel/PAPERS/physD-Final.pdf)，DOI：10.1016/j.physd.2006.09.035。
- 已核查摘要、方法描述和讨论：先调整位移使场对齐，再做幅度校正；包含对位移场的约束。
- 可借鉴“位移与幅度分开估计”和低信息情况下的约束；空间资料同化不等同于本任务的站点时间平移，也不证明稀疏三点序列足以识别位移。

**S9｜Jin et al., 2024, Meteorological Applications 31:e70007，评估研究。** [开放全文 §3.2、§4.3](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/met.70007)。
- 已核查误差分解公式和“约 70%”原文。SEQU 来自原序列误差与分别排序后的误差之差，覆盖广义时序不一致。
- 排序允许任意重排，远强于 −2…+3 h 的单一平移；因此“七成误差可通过平移修复”不成立。中国轮毂高度多模型统计也不能直接外推英国两场的 GFS 10 m 风。

**S10｜Le Guen & Thome, 2019, NeurIPS，DILATE，方法论文。** [官方论文页](https://papers.nips.cc/paper_files/paper/2019/hash/466accbac9a66b805ba50e42ad715740-Abstract.html)。
- 已核查摘要：在预测目标中分开处理形状和时间失真。适合作为机制评估的参考；本轮不建议引入新损失替代功率 MAE，避免同时改变多个变量。

**S11｜Cuturi & Blondel, 2017, ICML，Soft-DTW，方法论文。** [PMLR 原文页](https://proceedings.mlr.press/v70/cuturi17a.html)。
- 已核查方法摘要：对动态规划对齐路径进行可微软最小化。G1 的六个刚性平移候选 softmin 不是完整 Soft-DTW，也不因使用 softmin 获得算法新颖性。

**S12｜WindLogics / Xcel Energy, 2008，RD-57 官方技术报告，系统研究。** [原始报告 §3.1，PDF 第 25–28 页](https://www.xcelenergy.com/staticfiles/xe/Corporate/Renewable%20Energy%20Grants/Windenergyforecasting-Final-Report.pdf)。
- 已核查该节：明确讨论 NWP 锋面到达时间错误以及上游观测的提前预警价值；所展示的对应实验是塔站观测驱动的短时 SVM 预测。
- 因而它是动机近邻，不能据此宣称已实现与 G1 相同的相位修正算法。报告年份是 2008，不能采用搜索引擎抓取时间。

**S13｜Monbet, Ailliot & Cuzol, A State Space Model for Wind Forecast Correction，约 2008 作者稿，待全文复核候选。** [作者 PDF](https://perso.univ-rennes1.fr/valerie.monbet/doc/papiers_pdf/monbet_ailliot_cuzol08.pdf)。
- 可检索作者稿说明把偏差和相位作为隐藏状态；直接 PDF 获取超时，正式出版状态与全部公式未完成核验。
- 是“偏差 + 相位联合隐状态”的高风险遗漏线索，列入 Claude 获取全文清单，不把它当作已核实的跨站同方法证据。

其他核查线索：Jiang et al. [EGU 2009 官方摘要](https://meetingorganizer.copernicus.org/EGU2009/EGU2009-9192-2.pdf) 已讨论 bias/phase/Cressman 修正；用户提及的 2025 DTW 工作可定位到 [SSRN 5153521](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5153521)，目前只核实预印本摘要，不当作已同行评审、已验证的在线对照。

检索结论：尚未核实一篇同时满足“上游气象站、同次 NWP 残差、显式时间错位、到达下游前转移、机组级功率验证”的完全同构工作；这不等于证明首次。已有文献足以否定宽泛的首次跨站误差修正主张。

## 3. 五个主要问题

**G1-C1【确定的推论缺陷】把 SEQU 份额解释为可平移恢复份额。** 最小修正是改写动机，并在开发数据上实测受限 shift-only oracle 的收益；不能直接沿用 70%。

**G1-C2【确定的条件性数学缺陷】三点局部线性序列下，时间平移与自由幅度偏差不可识别。** 若 f(s)=a+vs，则 f(s+δ)+b=a+vs+(vδ+b)；任何保持 vδ+b 不变的组合都等价。无噪声也不能区分，联合自由拟合或去均值后相关匹配均不能保证恢复真实 δ。

**G1-C3【不确定的关键假设】上游相位误差可以在 1–4 h 下游预测窗口内转移。** 上风向不意味着气象结构相同；天气结构移动速度不必等于 10 m 风速。保留假设，预先固定传播方向/距离分层及负对照，用实验判断。

**G1-C4【确定的比较解释缺口】现有对照不足以把改进归因于相位转移。** 缺少直接跨站残差修正，以及 δ=0 的幅度修正；M_self 也不能替代原 TLA 对照。599 参数大于 6 不足以证明对照强度，且 6 仅是新增结构参数而非整个模型参数。

**G1-C5【待定义、不能直接判错】可用时钟、推断单位与成功判据未完全明确。** 要写清每条观测发布时间、GFS cycle/lead 支撑、两场合取检验、实用改善门槛、seed 汇总方式及 Hill of Towie 的训练/适配权限，详见实验建议。

另一个应检查的边界：δ 候选 −2,−1,0,1,2,3 在均匀 softmin 下的均值为 +0.5 h。只有在置信度门控或先验明确保证退回 δ=0 时才不会产生无证据平移；用户描述尚不足以判断实现是否已有该保护。

## 4. 统计与投稿判断

原 Holm 族 {Q1,Q3,Q4} 可以成立：每个问题先取两场单侧 p 值的最大值，再对三个问题做 Holm。该构造检验的是“两场均有效”，不要求两场独立。若要分别宣称每场、每时距均显著，必须另行处理这些声明的多重性。

Q3 的“点估计至少 1.5% 且对零显著”是一种合法的工程决策门槛，却不能表述成“已统计证明真实改善超过 1.5%”。后者应检验 E[0.985×loss_I − loss_M]>0。新方案采用哪一种须事先固定，不追溯改变已登记定义。

只通过原五/六臂实验，适合定位为有应用价值的增量方法。加上强残差对照、时间识别证据、完整季节覆盖和独立场确认后，可考虑 Renewable Energy、Energy、ESWA/EAAI 等方向；Applied Energy 或 IEEE Transactions on Sustainable Energy 属于需要更强机制与工程收益证据的冲刺目标。这是内容适配判断，不是录用承诺。

CCF 官方人工智能目录将 [ESWA、EAAI 列为 C 类](https://www.ccf.org.cn/Academic_Evaluation/AI/)。当前机制与三个风场证据不足以把 CCF A/B 通用机器学习会议设为自然目标。能源期刊并不按 CCF 排名来衡量领域价值。

中科院可把“一区/二区候选期刊范围”作为选刊目标，而不能由新意分数直接换算。此次无法登录 [中科院官方分区表](https://www.fenqubiao.com/Manager/OrderManager.aspx?id=59950) 核验目标年度大类/小类条目，所以不把具体期刊标成已核实的 2026 分区；正式选刊需按单位认可年份查询，也不能混用 JCR Q1。

## 5. 检索覆盖与交付边界

覆盖了：NWP temporal lag/phase correction、空间预测误差传播、实时残差后处理、观测条件化 GFS 校正、field alignment、DTW/形状与时间失真，以及官方 GFS 产品和 CCF 目录。采用公开关键词、论文题名和作者检索，未上传私人研究材料。

最关键的 S1–S9 已至少核查原始作者文本或出版社实质摘要/方法摘录；S1、S3、S6 尚非完整全文复现级核查。S13、2010 原始时间修正工作及 2025 DTW 预印本仍有全文覆盖缺口。水文/风暴潮没有取得比上述文献更直接、且足够核实的新机制证据，故不强行添加类比。

本次不检查或读取封存年份，不运行训练，不改原始数据、标签、冻结配置或 STAGE_PLAN。仓库开工检查发现既有 .gitignore 与三个 GFS shell 文件不在旧交接允许范围；该问题不影响本轮文献判断，未替作者修改。

可交给 Claude 的具体步骤见同目录 `experiment-plan.md`。其中所有新增阈值、基线和方法变体均是待审核建议，不能伪称原先已预注册。
