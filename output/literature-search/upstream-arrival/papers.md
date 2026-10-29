# 站点观测作用时间：方案用文献依据

检索日期：2026-10-29。范围：为下一轮创新方向和实验设计定位近邻；不是系统综述或“首次”证明。新主题与已有 wind-wiki 检索分开保留，后者不被覆盖。

查询使用公开通用关键词：wind power forecasting / upstream observations / advection / directional delays / exogenous forecasting / missing values / observation latency；未发送私有结果、代码或方案文字。来源优先原作者、论文主页和正式出版页；政策排除来源不列入依据。候选按标题去重，以机制覆盖停止扩搜，没有按引用量打分。

## 已核查的核心来源

| 论文及稳定来源 | 年份、状态、类型 | 已读范围与直接支持 | 对本方案的约束/用途 |
|---|---|---|---|
| [TSMixer: An All-MLP Architecture for Time Series Forecasting](https://arxiv.org/abs/2303.06053v5) | TMLR 2023；pure method | 作者v5摘要及期刊登记；时间/特征维MLP混合、辅助信息 | 当前底座来自既有方法；不能把混合器或外部通道当新贡献 |
| [TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables](https://arxiv.org/html/2402.19072v2) | NeurIPS 2024；pure method；[正式论文页链接](https://proceedings.neurips.cc/paper_files/paper/2024/file/0113ef4642264adc2e6924a3cbbdf532-Paper-Conference.pdf) | HTML摘要、引言、结构与缺测实验索引；内/外生信息分工、patch自注意力和变量交叉注意力；明确讨论时差和不规则外生序列 | “外生注意力＋错位/缺测”已被覆盖；必须有同信息量通用融合对照，后续需完整复现比较 |
| [Spatio-temporal analysis and modeling of short-term wind power forecast errors](https://orbit.dtu.dk/en/publications/spatiotemporal-analysis-and-modeling-of-shortterm-wind-power-fore/) | Wind Energy 2011，14(1):43–60，doi:10.1002/we.401；method + empirical analysis | DTU出版登记、作者稿相关性/方向/时延与条件模型部分；[作者稿](https://pierrepinson.com/docs/tastuetal09_rev.pdf) | 风向条件化和空间时延不是新发现；原文还指出天气特征传播速度未必等于地面风速，直接限制d/v解释 |
| [Enhancing Short-Term Wind Speed Forecasting using Graph Attention and Frequency-Enhanced Mechanisms](https://arxiv.org/html/2305.11526v2) | 2023作者预印本v2，未独立核实最终会议/期刊；pure method | 摘要、引言、III-A：复数邻接的实部相关/虚部时间差，先移位再图聚合 | 动态图＋时差已有；目标是风速且提前量不同，不能照抄公共指标比较本项目功率结果 |
| [Recurrent Neural Networks for Multivariate Time Series with Missing Values](https://dspace.mit.edu/entities/publication/ae7fcc3e-edac-4bd3-a684-c4f44a18fb7d) | Scientific Reports 2018；pure method；doi:10.1038/s41598-018-24271-9 | MIT正式版本记录与摘要；GRU-D使用掩码、时间间隔表示缺测 | 年龄/掩码编码是既有手段；临床分类证据不证明风电时延处理有效 |
| [ExoTST: Exogenous-Aware Temporal Sequence Transformer for Time Series Prediction](https://arxiv.org/abs/2410.12184) | ICDM 2024（作者页注明接受并列正式doi:10.1109/ICDM59182.2024.00105）；pure method | 作者摘要、元数据；融合过去与当前外生序列，报告缺测/噪声鲁棒性 | 可用时间不同不能被简单混作同时信息；本案无未来气象，不能为凑输入偷偷提供未来外生驱动 |
| [UniWind: Day-Ahead Wind Power Forecasting via Physics-Informed State Routing](https://arxiv.org/abs/2607.01670v2) | 2026预印本v2，更新2026-10-15；pure method；未核实正式接受 | 摘要/版本历史；物理发电先验、历史偏差、未来气象上下文及状态修正 | 物理先验＋状态专家/有界修正也已有；它的日前未来气象设定不同，不能直接支持本案只用历史观测 |
| [TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a7ac8a21e5a27e7ab31a5f42a0117bdb-Abstract-Conference.html) | ICLR 2024；pure method | 官方摘要；分解、多尺度混合和多预测器 | 仅加多尺度MLP是拥挤备选，不是当前具体站点问题的必要机制；TSMixer和TimeMixer不能混称 |

这些来源只支撑表中的局部描述，不代表本轮逐篇完整复现。引用既有工作作为先验来源，不把论文的百分比成绩移植到本项目。

## 高重叠候选：证据不足，必须补核

- **Physics-enhanced dynamic graph network for wind power forecasting with directional delays and aerodynamic interference (DPGNet)**：[出版方链接](https://www.sciencedirect.com/science/article/pii/S0378779626012393)，检索页列doi:10.1016/j.epsr.2026.113951。检索可见摘要声称方向时延、MIC和尾流；直接页面403，搜索元数据显示卷期为2027年4月，与当前日期不一致。未证实首次上线日期及全文，**不作为截至今日已完整核验的发表文献**，但不能忽略潜在重叠。
- **Considering Meteorological Spatio-temporal Propagation and Multi-Factor Coupling Short-Term Wind Power Forecasting**：[出版方记录](https://www.sciopen.com/article/10.26599/PEF.2026.9650019)。检索记录显示2026年10月上线、时延注意力和动态物理图；两次直接访问超时，未核验全文。列为待查候选，不据摘要片段判定具体模块缺失。
- **Physics-informed LLM-guided dynamic spatio-temporal graph deep network for short-term wind farm power forecasting**：[出版方记录](https://www.sciencedirect.com/science/article/pii/S019689042600943X)。搜索记录卷期为2026年11月，早期上线日期待核；仅用于提醒LLM动态风向图路线已拥挤，未作为建议实现依据。
- 其余检索命中包括PI-TFN、PSTG-Net、物理尾流网络和光伏图模型：与拟议“历史站点年龄×目标小时”差异较大或全文证据不足，不据此作性能判断，不扩成无关模块清单。

## 覆盖结论

已覆盖：底座来源、外生融合近邻、风方向/时差近邻、缺测年龄原语、近期风电状态路由、常见多尺度替代。最有风险的重叠是方向时延图和TimeXer已有的错位处理。尚未证明的差异是“观测可用性与小时目标的联合作用时间匹配”在同数据/容量下的必要性。

下一步补查应针对上述高重叠全文、公式和可用性假设，而不是继续收集泛泛风电论文。即使未找到完全同构，也不等于已经证明创新。

## 2026-10-29 补充检索：创新评分所用的新增近邻

本轮用途：更新既有概念判断并发展候选，不作论文录用预测。公开查询覆盖 wind power / observation latency / advection / conditional information / time-series deferral / LLM rules；未发送本项目私有指标或未发表方案原句。按标题与arXiv编号去重；排除来源政策仍适用。未对未读全文作论文质量分数。本轮没有增加数据源或下载实验数据。

以下是新增决定性来源，旧表中TimeXer也再次核查了引言与外生变量表示部分。日期以版本记录为准，不采用搜索引擎的相对时间标记。

1. **Theuer等，Enhancing minute-scale lidar-based power forecasts of offshore wind farms towards an operational use**：[WES讨论稿](https://wes.copernicus.org/preprints/wes-2024-141/)，2024-12-10；method + empirical analysis。页面说明修订版已接受，正式发表条目未另核；本次读摘要与状态，PDF超时。直接支持：观测年龄、传播时长与预报技巧有关。分钟级海上lidar不能直接等同于小时级公共地面站，但“观测年龄＋传播时间”不再可视作空白。
2. **Tan等，Are Language Models Actually Useful for Time Series Forecasting?**：[NeurIPS 2024官方页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract-Conference.html)；empirical analysis。读官方摘要：对其研究的三种方法，移除／替换LLM未降低效果，部分改善。这限定于被研究的方法，不证明所有LLM或本项目路由都无效；要求明确LLM不可替代的作用。
3. **Jiang等，ConceptTS**：[arXiv:2608.21277v1](https://arxiv.org/html/2608.21277v1)，2026-08-21预印本，未核实正式接受；pure method。读摘要、III-A/B、相关推理边界。已用离线LLM提出概念与可执行规则，推理时无需LLM；因此“LLM写规则后部署小模型”本身不新。剩余差异需落到气象规则的时效与收益有效性，而不是规则生成。
4. **Lai等，ReLaMix**：[arXiv:2603.20869v1](https://arxiv.org/html/2603.20869v1)，2026-03-21预印本；pure method。读摘要、III-A/B/C：在金融序列模拟ZOH延迟，用轻量瓶颈／残差混合处理陈旧值。“轻量mixer＋延迟鲁棒性”已有近邻；与物理空间传播不同，但不能仅换风电数据声称新方法。
5. **Mao等，Regression with Multi-Expert Deferral**：[ICML 2024官方页](https://proceedings.mlr.press/v235/mao24d.html)；method + theory。读官方摘要：回归多专家选择、预训练预测器后学习选择器、一阶段与两阶段均已建模。支持把通用回退／路由视作已有原语，不自动支持本项目统计保证。
6. **Montreuil等，Learning-to-Defer in Non-Stationary Time Series via Switching State-Space Models**：[arXiv:2601.22538v4](https://arxiv.org/html/2601.22538v4)，2026-10-05修订；预印本，pure method + theory。读摘要、贡献与4.4相关定位：非平稳序列、咨询成本与专家可用性已进入路由研究。其定理依赖预测准确性与可辨识等条件，不能迁移成风电“保证不退化”。
7. **Angelopoulos等，Conformal Risk Control**：[作者稿](https://people.eecs.berkeley.edu/~angelopoulos/publications/downloads/conformal-risk.pdf)，ICLR 2024（作者履历与书目登记交叉核对）；theory + method。读摘要／引言：经典构造处理有界单调损失。时序相关、漂移以及有正有负的相对损失不能不加条件直接套用。另有2026非单调扩展，本轮未充分审查，不能据旧条件断言该领域完全不能处理非单调风险。
8. **Causality by Abstraction: Symbolic Rule Learning in Multivariate Timeseries with Large Language Models**：[arXiv:2602.17829v1](https://arxiv.org/html/2602.17829v1)，2026-02-19预印本；pure method。读摘要、模拟器适用条件及附录说明。已含时间／延迟语义、规则验证与反事实输入；但依赖可调用的原则模型／模拟器。本项目只有观测历史，不能把规则和方向打乱试验升级为因果识别。

补查失败项：LLXSTG出版页仍403，首次上线时间未核实，保留在“高重叠候选”而不据此作定论；LACGNN的SSRN全文403，未纳入已核查核心。2015年条件互信息风电特征选择命中IEEE官方索引，但直接页418、作者PDF跳转失败，记作后续全文核查入口：[doi:10.1109/TSTE.2015.2441747](https://doi.org/10.1109/TSTE.2015.2441747)。

机会判断（本段是推断）：方向／年龄／规则／路由各自都有先例。可以继续研究“观测可用时间、到达目标小时的软支持、相对已有底座的增量价值”的统一选择问题；尚未证明该组合在最近工作之外构成实质方法差异。具体概念评分及改判条件见 `docs/UPSTREAM_INNOVATION_PROPOSAL.md` 第13节。

第4次交接更正：Claude明确LLXSTG也只有摘要级信息，双方均未读到全文；此前“Claude全文判断”记载有误，现撤回。schema校验/回退细节仍未全文核实，否决A-L主要依据R4及信息优势不足。PMC11511355访问遇验证页；Mao Yang等的SciOpen索引显示2026-10-10发表和时延注意力／物理扩散图摘要，仍未取得全文；IJEPES S0142061526002115尚无可靠正文。以上列入A0定向核查，不谎报已完成。

## A0定向全文核对结果（2026-10-29；更新上述访问状态）

七项中五项取得全文并定向核对机制；两项仍不完整。此处“全文取得”不表示复现、逐式证明或全面质量审查。只查公开论文，没有传送本项目原始数据、私有结果或代码。PMC11511355为用户本轮点名的MDPI来源，作为指定近邻单列核验；不据此扩展推荐来源或给论文质量打分。

| 指定近邻 | 本次实际证据与方法位置 | 对统一方案的直接约束 |
|---|---|---|
| Theuer，WES讨论稿wes-2024-141 | [27页作者PDF](https://wes.copernicus.org/preprints/wes-2024-141/wes-2024-141.pdf)已取得；核对§2.2、2.5、3.3–3.4、4.2 | 用观测局地风矢量平流，考虑多个旧扫描和扫描内观测时刻；预测目标附近±30秒的到达窗口，并以风矢量年龄和预测提前量分析传播时长与技巧。“历史年龄＋目标时段到达匹配”已有直接先例；本案差异只能落到小时公共站、模拟可用性、软先验和给定底座的对照实证，不能把原语宣称为首次。分钟lidar结果不等于本案效果。 |
| TimeXer | [作者v2全文](https://arxiv.org/html/2402.19072v2)，§3、§4.3.1及附录B.3 | 内生patch/global token和外生变量表示，经自注意力/交叉注意力融合；检查了不同长度与缺测设定。所查结构没有显式站点距离传播到目标小时的几何项，但通用外生融合及错位输入并非空白，B2不可省。 |
| GFST-WSF | [作者v2全文](https://arxiv.org/html/2305.11526v2)，§II–III、式2–7、§IV B–D | 复数邻接编码相关与时间差，经时间移位再GAT/FEDformer；涉及距离/方向信息。不能把“先对齐时延再融合”本身当新机制；其风速任务和数据提前量与本项目不同。所核公式未建立本站观测历史接收日志，不能反向断言全文完全没有可用性讨论。 |
| DPGNet | [出版页](https://www.sciencedirect.com/science/article/abs/pii/S0378779626012393)，doi:10.1016/j.epsr.2026.113951；仅取得摘要、引言及部分章节片段，正文访问403 | 片段已明确风向筛上游、MIC选时延以及距离/相关/尾流边权。高重叠仍成立，但全文、完整公式和首次上线日期未核实；卷期标2027年4月，不能混作已核实的2026正式出版。请Claude补作者稿/PDF，不凭片段声明有或没有具体可用性约束。 |
| Mao Yang 2026，PE-STGNN | [SciOpen正式页](https://www.sciopen.com/article/10.26599/PEF.2026.9650019)，2026-10-10；PDF入口403，但通过该页公开全文接口及其正文资源取得全文。核对§III、IV A–B式12–23、VI A | 式12是距离衰减乘两场气象影响特征均值的**对称**图亲和权；原文明确它不独立识别传播方向/速度。时延模块为GRU、位置编码及气象衰减调整的点积注意力。由此不能把题名“传播”直接等同于本案d_parallel/v与年龄的目标小时积分，但“物理图＋时延注意力”已存在。数据为内蒙古100场、2020年15分钟记录，与本案输入信息不同。 |
| PMC11511355 | [PMC全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11511355/)，Ultra-Short-Term Wind Farm Power Prediction Considering Correlation of Wind Power Fluctuation；Sensors 2024,24(20),6538，doi:10.3390/s24206538；直接抓取验证页，检索工具取得全文，核对§2–5、式1–2 | 按风向选择上游场，式2为T=d/[v·cos(θp−θw)]，再区分具有上游提前信号的预测时段与其余时段。注意它是**除以cos**，不能抄成我们的d·cos/v。这是“方向＋到达时延＋分目标时段”的直接近邻；对象为邻场功率、15分钟分辨率，与公共地面气象站不同。 |
| IJEPES S0142061526002115 | 出版社直链/abs入口403，多次精确PII检索未定位可靠题名和正文 | **身份及全文未核验**；请Claude提供完整题名、DOI或合法作者稿。不得替换成相似PII，更不能虚构方法差异。 |

由证据作出的判断：A可以作为受约束的应用机制假设继续设计，但仅“上风向＋年龄＋传播匹配”不足以支撑强新颖性。Theuer与PMC近邻比泛泛图网络更直接；Mao全文则显示题名相似不等于公式同构。统一方案应把可检验贡献写成：同信息、同容量下，对小时公共站历史施加可用性与目标小时软支持是否比B2更有用，以及何时失效。实际信息增益、创新评分和有效性仍待对照，不能由A0覆盖率给出。

A0输入统计及时间语义见 `output/upstream_observation_review/station_history_v33/README.md`；历史归档的名义时间不等于真实发布时刻。尚未核完的两篇使“七篇全文核对完成”这一说法不成立。
