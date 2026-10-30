# 两项贡献的定向近邻核查

## 第8次交接补核（2026-10-30）

当前审查对象为proposal §14.12；以下第7次及更早账本保留其原阅读层级，不代表本轮重新阅读全文或执行实验。仅以公开题名/URL检索，未上传未发表方案或私有数据。

**R19 — ChronosX: Adapting Pretrained Time Series Models with Exogenous Variables.** [PMLR正式发表页](https://proceedings.mlr.press/v258/arango25a.html)，AISTATS 2025，PMLR 258:2242–2250。类型：方法；优先级Risk。本轮核到官方题名、作者、卷页及摘要：通过模块注入过去/未来协变量，可不修改预训练主体。足以否定“给预训练时序模型接入协变量本身新颖”，不证明与本项目物理到达代理同构。PMLR所链GitHub PDF未被浏览工具解析为PDF，Amazon托管PDF也未返回可读正文，故**阅读深度仅为正式页/摘要**，不宣称已审计冻结策略、完整训练机制或复现指标。

**R20 — Chronos-2官方接口契约（软件来源，不是新论文）.** [官方pipeline.py](https://raw.githubusercontent.com/amazon-science/chronos-forecasting/main/src/chronos/chronos2/pipeline.py)，2026-10-30读取主分支的predict说明第455–466行：future_covariates键须属于past_covariates，二者分别匹配预测长度和历史长度；批内schema须一致。这要求阶段5定义预测代理及其历史/缺失标记通道。该可变分支仅作本轮接口诊断证据，不是已冻结的运行revision；实际权重、软件版本、许可证与信息预算仍另审。未安装或运行模型。

**指定近邻的可读全文仍未取得：** Chasing the Wind的[PDF入口](https://openreview.net/pdf?id=7WG3XbHV3V)、DA-Transformer的[PDF入口](https://openreview.net/pdf?id=PLO1gjCMk5)本轮仍返回验证页；针对Chasing的公开作者稿/GitHub检索没有获得可核验全文。DPGNet的[DOI入口](https://doi.org/10.1016/j.epsr.2026.113951)本轮工具不可访问。维持“全文未闭合”，不把访问失败解释为不存在重叠，也不绕过访问控制。

AirDDE方法、UniWind v2跨场段落承接R16/R17已有定向核查。AirDDE的风/距离相关延迟与本案学习标量κ应区分；UniWind涉及Penmanshiel不等于其未来NWP输入与本案历史观测输入一致。此次没有把它们的实验数字当成本项目证据。

---

## 第7次交接补核（检索2026-10-29，记录完成2026-10-30）

第7次审查对象为proposal §14.11的C1/C2；原A*/K*已放弃，以下历史账本不代表执行授权。

**R16 — AirDDE: Multifactor Neural Delay Differential Equations for Air Quality Forecasting.** [AAAI正式页面](https://ojs.aaai.org/index.php/AAAI/article/view/38627)，2026-03-14，40(19):15941–15949，DOI 10.1609/aaai.v40i19.38627；[正文](https://ojs.aaai.org/index.php/AAAI/article/view/38627/42589)。类型：方法；优先级Risk。定向阅读正文方法页15942–15944，公式(4)–(10)：风向/速度/距离构造延迟邻接，历史邻居注意力，延迟平流演化。不能泛称这些组合是本项目首创；尚不证明其等同本项目的小时功率剩余到达窗与κ参数。

官方[代码](https://github.com/w2obin/airdde-aaai)解析为`9277a56db4bcc0e5e6a3a78bc9aa2b8df09d9774`（2026-07-04），API未识别许可证，未安装运行。定向读该版本`model.py::LocalMemoryModule`：邻居按最后风变量的距离top-k取出，窗口tau固定参数；该路径与论文地理传播邻接的描述不能直接视为相同，故不把代码当论文全部机制已实现的证明。

**R17 — UniWind: Day-Ahead Wind Power Forecasting via Physics-Informed State Routing.** [作者正文v2](https://arxiv.org/html/2607.01670v2)，预印本，未确认正式接受。类型：方法；优先级Risk。本轮从先前摘要层升级为定向阅读§4.1.5、§4.2.2和§4.5.2：跨场/跨区域零样本、Penmanshiel、Chronos-2等对照及站点校准消融已经出现。它使用未来NWP；不能直接否定本项目仅历史地面观测的设定，但“物理先验＋跨场＋Penmanshiel确认”不足以独立支撑7分。没有复现其结果或完整审计数据泄漏，不引用其数字作为本项目证据。

**R18 — Benchmarking Time Series Foundation Models for Load Forecasting Under Covariate Uncertainty.** [作者正文v1](https://arxiv.org/html/2610.07232)，2026-10-05预印本。类型：benchmark；定向读摘要、实验目录及§II-F，非全部实验复核。已区分未来协变量质量并讨论预训练数据污染；支持本项目报告预测协变量误差和权重训练数据披露，不能据此断言本项目所选权重无污染。这里不确认正式接受状态。

**仍未闭合，不能写“全文已核验”：**

- Chasing the Wind：[正文入口](https://openreview.net/pdf?id=7WG3XbHV3V)及[哈希PDF入口](https://openreview.net/pdf/599b9f556f61eedf99ccef3261ac1058e364c38b.pdf)本轮均被OpenReview浏览器验证页阻挡；题名/MoP/GitHub定向搜索未找到可核验的作者公开镜像。搜索索引只能支持存在该候选，不能证明是否同构、是否录用。待可访问作者稿或用户提供PDF再完成方法全文比对。
- [DA-Transformer](https://openreview.net/forum?id=PLO1gjCMk5)同样返回验证页；[DPGNet DOI](https://doi.org/10.1016/j.epsr.2026.113951)和[条件互信息工作](https://doi.org/10.1109/TSTE.2015.2441747)本轮正文访问仍失败。既有日期/机制未核实标记保留。

以上是完成了可访问材料的补核，并非所有近邻已闭合。未绕过访问控制，未因无法访问而给“无重叠”结论。

---

检索日期：2026-10-29。用途：为 `two-contribution-review.md` 的概念判断提供证据，不是系统综述、论文质量排名或“首次”证明。与同目录 `papers.md`、`idea-grounding.md` 的历史检索互补，不覆盖旧结论。

本轮检索覆盖四组公开关键词：upstream wind / advection / arrival / observation latency；residual correction / executable features；model upgrade / calibration transfer / memory portability；官方 GitHub 实现。先检索近邻，再检查方法段和代码，不以题名相似直接判定同构。没有向搜索服务上传私有数据、结果文件、代码或完整未发表方案。MDPI 泛检索命中未作为新增依据；此前用户点名核验的来源仍保留在旧报告中。

以下精选15项会改变设计或评分的来源。阅读深度分别标注；未逐篇复现，未独立证明其定理，未核实的会议接受状态不补猜。作者报告的实验成绩不移植给本项目。

## 上游观测与传播：R1–R7

**R1 — Baïle & Muzy, Leveraging data from nearby stations to improve short-term wind speed forecasts.** [作者全文 v2](https://arxiv.org/html/2207.14679v2)，2022预印本版本；本轮不依赖未重新核实的期刊登记。类型：实证分析。读摘要、模型说明、§4.1及邻站/提前量讨论。邻站观测与上游传播解释均已有；这不是本案的新发现。本轮未确认官方代码。不能从某区域估计的等效传播速度推出本项目 κ 的物理可辨识性。

**R2 — Theuer 等, Enhancing minute-scale lidar-based power forecasts of offshore wind farms towards an operational use.** [WES讨论页](https://wes.copernicus.org/preprints/wes-2024-141/)，[作者PDF](https://wes.copernicus.org/preprints/wes-2024-141/wes-2024-141.pdf)，2024讨论稿。类型：方法＋实证。全文定向核对承接本项目A0检索记录：§2.2、2.5、3.3–3.4、4.2，包含观测年龄、平流及目标到达窗。页面的后续审稿状态不能当成已核验的正式卷期。不能把“年龄＋到达时间”独立算新贡献；分钟级lidar与小时级地面站仍有输入条件差别。

**R3 — Ye 等, AIRU-WRF.** [作者全文 v2](https://arxiv.org/html/2303.02246v2)，2023预印本版本。类型：方法。读摘要和§3.3的随机平流核：空间偏移为 γ−Θw，Θ有均值与协方差，结合扩散；风场量来自NWP。直接约束：随机传播、传播不确定性及物理统计融合并非空白。NWP条件与本案历史观测不同；区别不等于本案必然更优。本轮未确认官方代码。

**R4 — Hettige 等, AirPhyNet.** [作者全文](https://arxiv.org/html/2402.03784)，2024预印本版本；[官方项目仓库](https://github.com/kethmih/AirPhyNet)。类型：方法。读§3.2–3.4，并定向读 `airphynet_model.py`、`ode_func.py`：风变量经可学习映射参与平流，扩散/平流表示经学习门控融合。它研究空气质量，不是本案功率任务，但足以否定“把可学习物理传播放进网络本身很新”。

**R5 — Advection-diffusion spatiotemporal recurrent network for regional wind speed prediction（ADNet）.** [出版方记录](https://www.sciencedirect.com/science/article/pii/S0031320325009434)，[作者仓库](https://github.com/ShidongC/ADNet)。类型：方法。出版页直接403，论文仅核到索引摘要/方法片段，卷期日期不据PII推断。代码取得：`modules/adnet_modules.py` 中 `conv_uvx` 学习两个方向场，平流梯度与扩散二阶项进入状态更新，`MSCA` 学习调制。代码证明学习型平流原语存在；不能据此声称它已实现本案标量 κ 注意力或公共站网投影。

**R6 — Wang 等, TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables.** [作者全文 v2](https://arxiv.org/html/2402.19072v2)，[NeurIPS 2024正式论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/0113ef4642264adc2e6924a3cbbdf532-Paper-Conference.pdf)，[仓库](https://github.com/thuml/TimeXer)。类型：方法。承接既有方法阅读，本轮读 `models/TimeXer.py`：内生全局token查询外生表示。通用外生融合是必要对照；额外气象输入或cross-attention本身不能算创新。

**R7 — Lai 等, ReLaMix: Residual Latency-Aware Mixing for Delay-Robust Financial Time-Series Forecasting.** [作者全文 v1](https://arxiv.org/html/2603.20869v1)，2026-03-21预印本。类型：方法。读摘要、问题设定，方法段阅读承接已有检索：模拟ZOH陈旧值、瓶颈压缩和残差混合。简单“延迟增强＋轻量mixer”已有近邻。本轮未确认该论文官方仓库；GitHub上同名视频动作识别项目不是它，不能误配代码。

## 规则与底座更替：R8–R15

**R8 — Wang 等, CRAFTER: When Do Corrective Features Help? An Agent for Corrective Feature Discovery on Black-Box Forecasters.** [作者全文 v1](https://arxiv.org/html/2608.05207v1)，2026-08-05预印本。类型：方法。读方法和附录L：可执行特征、残差修正、来源无关验证，以及跨时间保留特征后重拟合修正器。直接约束：规则复用、轻量修正和目标侧再拟合均已有。未确认官方代码。它是新点II必须正面对照的近邻，不能只比较旧Wiki。

**R9 — Dong 等, Guarded Agent-in-the-Loop Residual Adaptation for Evolving Wind Power Regimes.** [作者全文 v1](https://arxiv.org/html/2610.34968v1)，2026-10-26预印本。类型：方法。读§3.1–3.3：冻结底座、情景残差专家、成熟反馈驱动的接受门控。风电＋Agent＋残差＋门控已有直接近邻；不再以这四项组合主张独创。未确认官方代码；不推断作者与本项目的关系。

**R10 — Demaeyer & Vannitsem, Correcting for model changes in statistical postprocessing – an approach based on response theory.** [NPG正式全文](https://npg.copernicus.org/articles/27/307/2020/)，2020，27:307–327。类型：方法＋理论。读摘要、引言与后处理问题说明：模型变化会使后处理失配，利用响应理论修正。没有逐式审计全文证明。“模型升级后修正器过时”不是新问题；本案只能在黑盒规则效用重估的具体机制上争取区别。

**R11 — Rinaldi 等, Update Your Transformer to the Latest Release: Re-Basin of Task Vectors.** [ICML 2025/PMLR官方页](https://proceedings.mlr.press/v267/rinaldi25a.html)，[TransFusion仓库](https://github.com/aimagelab/TransFusion)。类型：方法。读官方摘要/方法概述并核 `permutations/weights_matcher.py` 的权重匹配/指派实现。利用参数重排迁移任务向量；它要求的白盒权重与本案外部修正规则不同。泛称“底座升级后知识迁移”仍然不新。

**R12 — Doula, Conformal Calibration Transfer（TCC）.** [作者全文 v1](https://arxiv.org/html/2609.10737v1)，2026-09-09预印本。类型：方法＋理论。读§3.1–3.5及限制：利用配对无标签视图、校准运输与不确定性代理；覆盖保证依赖任务不变、代理控制与独立性条件。对象是预测集合校准，不是旧点修正相对新底座的收益。不能宣称首次无标签校准迁移，也不能照搬其保证。未确认官方代码。

**R13 — Balachandran, Pay Only for Disagreement: Certified No-Regression Verdicts for Model Updates with Matching Label-Complexity Bounds（DISCERN）.** [作者全文 v1](https://arxiv.org/html/2609.17560v1)，[摘要记录](https://arxiv.org/abs/2609.17560)，2026预印本。类型：方法＋理论。读更新审计设定和§5.6：配对模型风险差、无标签分歧、序贯审计，也讨论连续分数/Lipschitz损失。不能把本案称为首次连续回归更新审计。arXiv编号年月与页面所列2026-07-22日期不一致，本报告保留该元数据异常，不据其精确时间作优先权结论；定理未独立验算。

**R14 — Goyal & Ray, Does Your Agent’s Memory Survive a Model Upgrade? A Controlled Study of Memory Portability.** [作者全文 v1](https://arxiv.org/html/2609.05339v1)，2026-09-04预印本。类型：实证分析。读摘要及研究结构：比较不同记忆格式在模型更换后的保持/修复。直接约束：固定schema与模型升级后记忆可迁移性已经被研究。本案的区别应是可执行气象修正规则的数值效用，而非把Wiki改成结构化记忆。未确认官方代码。

**R15 — Zhang 等, Prediction-Powered Risk Monitoring of Deployed Models for Detecting Harmful Distribution Shifts.** [作者全文 v1](https://arxiv.org/html/2602.02229v1)，2026-02-02预印本。类型：方法＋理论。本轮读摘要和引言：少量真实反馈与预测辅助的风险监测。尚未完整核验其序贯推导；只据此排除“用少量标签＋无标签预测监控风险”这一泛化首创说法。未确认官方代码。

## GitHub实际核查范围

2026-10-29经GitHub公开API解析默认分支到以下commit，再读指定源文件。只读，没有clone、安装、运行、下载权重或数据。许可证列的“未识别”不是可任意复用的许可。

- **ADNet**：`4e5bbc67e3d9a2f942d8437b40e5171636768886`，2025-08-19；API未识别许可证。读完整 `modules/adnet_modules.py` 与 `modules/constraint_modules.py`。仓库README很简略，未验证训练入口/数据是否足以复现。固定源码：[平流模块](https://github.com/ShidongC/ADNet/blob/4e5bbc67e3d9a2f942d8437b40e5171636768886/modules/adnet_modules.py)。
- **AirPhyNet**：`e77576cfea777e8cd07f2ae198c560a8790f4b91`，2024-02-24；API标MIT。定向读 `airphynet_model.py`、`ode_func.py` 的风场与门控路径，未执行。固定源码：[ODE模块](https://github.com/kethmih/AirPhyNet/blob/e77576cfea777e8cd07f2ae198c560a8790f4b91/ode_func.py)。
- **TransFusion**：`40aaf0a82ee6d3715dddcf587923d770619ea5c8`，2025-07-30；API未识别许可证。读目录、README线索和权重匹配代码片段。固定源码：[权重匹配](https://github.com/aimagelab/TransFusion/blob/40aaf0a82ee6d3715dddcf587923d770619ea5c8/permutations/weights_matcher.py)。
- **TimeXer**：`76011909357972bd55a27adba2e1be994d81b327`，2024-11-27；API未识别许可证。读完整 `models/TimeXer.py`，未复现其论文成绩。固定源码：[模型](https://github.com/thuml/TimeXer/blob/76011909357972bd55a27adba2e1be994d81b327/models/TimeXer.py)。
- **Chronos-2**：[官方pipeline](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py)，本轮在线读可变main的接口说明，没有锁运行版本：`target`为必需历史输入，`future_covariates`与过去协变量有schema约束。零样本免微调不等于无需任何本地历史；把预测出的天气送进该接口，也不会把它变成未来实测真值。
- **ppi_py**：[官方仓库](https://github.com/aangelopoulos/ppi_py)的README检索可见；本轮API遭SSL失败，未取得固定commit/定向源代码，不列为已完成代码核查。

## 未闭合的高风险近邻

- **Diffusion-Advection Transformer for Air Quality Prediction**：[OpenReview](https://openreview.net/forum?id=PLO1gjCMk5)。索引PDF片段含学习型物理算子和注意力；网页/PDF/API直接访问受阻。只作为重叠警报，不能写“已全文证明它有/没有本案κ”。
- **Chasing the Wind: Background Flow Tracing for Wind Speed Forecasting**：[OpenReview PDF](https://openreview.net/pdf?id=7WG3XbHV3V)。索引片段含上游区域、近似固定时延、残差；直接访问/API403。接受状态与完整机制未核实，不能用“没看到”排除重叠。
- **DPGNet**：见 `papers.md` 的出版日期/访问异常；尚未闭合。
- 条件互信息风电特征选择：[IEEE DOI入口](https://doi.org/10.1109/TSTE.2015.2441747)，承接旧报告的全文访问障碍。A*必须比较其“条件新增信息”主张，不能因年代较早忽略。

## 对Claude §14.9备选的补充来源

[Benchmarking Time Series Foundation Models for Load Forecasting Under Covariate Uncertainty](https://arxiv.org/abs/2610.07232)，2026-10-05预印本，摘要标明将参加2027会议（未独立核实接受）。本轮读摘要：明确区分可用/预测准确/噪声协变量，研究Chronos-2等。只支持“预测协变量有误差且应单独评价”的要求，不证明本案跨场方案无效。

没有查到完全相同实现，只能支持继续研究；它既不是无先例证明，也不构成创新性7分的下限保证。
