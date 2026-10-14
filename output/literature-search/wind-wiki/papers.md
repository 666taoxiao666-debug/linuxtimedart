# Wind forecasting and executable Wiki: source notes

Search/read date: 2026-10-14. Scope: short-term SCADA forecasting, temporal/spatial
adaptation, evidence-grounded knowledge organization and retrieval. Seventeen
distinct candidates were screened. Eight focal papers below were inspected at
method and experiment level; reading ranges are explicit. No journal ranking is
inferred from a search snippet. Applied Energy and IEEE TSTE are the selected
mainstream high-level energy venues. Conference memory papers are not wind papers.

## Wind forecasting

### Temporal collaborative attention for wind power forecasting

Yue Hu, Hanjing Liu, Senzhen Wu, Yuan Zhao, Zhijin Wang, Xiufeng Liu.
Applied Energy 357 (2024), 122502. [Publisher](https://doi.org/10.1016/j.apenergy.2023.122502).
[Open publisher-version copy](https://backend.orbit.dtu.dk/ws/portalfiles/portal/346090239/1-s2.0-S0306261923018664-main.pdf).
Type: method. Priority: B, useful architecture control, not a transfer guarantee.
Read: sections 3–5.1, algorithm 1/2, equations 17–27; scanned comparison tables.

The mechanism combines a GRU long-range path, collaborative attention and a local
short-range residual. Experiments distinguish component combinations and use two
datasets. Importantly, its main experiment aggregates hourly observations to daily
averages. Its instance attention normalizes along the batch dimension (equation
17), so this operation should not be imported into an independently issued online
forecast without checking batch-composition dependence. WSTD2 missing values are
zero-filled; this is not an acceptable default for our retained missing-data
semantics. Adopt the explicit local/global comparison logic, not the whole
architecture or its reported accuracy as a comparable number.

### An Adaptive Approach for Probabilistic Wind Power Forecasting Based on Meta-Learning

Zichao Meng, Ye Guo, Hongbin Sun. IEEE Transactions on Sustainable Energy 15(3)
(2024), 1814–1833. [Published record](https://ieeexplore.ieee.org/document/10476768/),
[author preprint](https://arxiv.org/abs/2308.07980).
Type: method. Priority: A for adaptation mechanism. Read author preprint sections
II–IV, especially III.B/C and equations 17/18; application/benchmark settings.

Offline task adaptation and online incremental updates are different operations.
The latter uses only recently available outcomes, with horizon-dependent label
delay and a forgetting factor. Tasks differ by location or lead time. Its
chronological 40/20/40 split and identical online treatment of comparison methods
are important protocol anchors. Adopt explicit availability time and adaptation
scope. Do not claim that adding timestamps implements meta-learning; our prototype
does not implement MAML or online gradient updates. Its pinball/reliability scores
are also not our Persistence-relative MAE Skill.

### A novel day-ahead regional and probabilistic wind power forecasting framework using deep CNNs and conformalized regression forests

Jef Jonkers, Diego Nieves Avendano, Glenn Van Wallendael, Sofie Van Hoecke.
Applied Energy 361 (2024), 122900.
[Institutional record/full text](https://biblio.ugent.be/publication/01HQ8SA9C23K1EA9A2QJRD96N7).
Type: method. Priority: A for uncertainty/evidence separation.
Read: sections 3–5.3, algorithms 2/3, equations 2–6, reliability discussion.

The pipeline separates deterministic power prediction, conditional uncertainty and
calibration. Trees define local regions while separate calibration examples supply
their residual distributions. Evaluation includes pinball loss, CRPS and quantile
coverage, rather than treating sharp distributions as automatically reliable.
The article explicitly leaves coverage proofs for its new adaptive forest to
future work. Adopt separate fitting/evidence stages and evaluate uncertainty
reliability. Do not transfer its logit residual transform to raw Kelmarsh power:
negative and above-capacity readings are retained here. Its regional NWP-dependent
day-ahead task is not our history-only two-hour SCADA task.

## Knowledge/Wiki and memory

### Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models (STORM)

Yijia Shao, Yucheng Jiang, Theodore Kanell, Peter Xu, Omar Khattab, Monica Lam.
NAACL 2024, 6252–6278. [Official record](https://aclanthology.org/2024.naacl-long.347/).
Type: system/method. Priority: B for source-grounded knowledge curation.
Read: sections 2–5, citation analysis and editor evaluation.

STORM separates research, outline construction and grounded writing. Perspective-
guided conversations expand sources, but expert evaluation still finds unsupported
inferences and over-association. Adopt source-linked knowledge records and explicit
unsupported relations. Do not insert its simulated conversations into the numeric
forecasting loop or interpret a well-written event explanation as measured gain.
Its Wikipedia quality metrics do not measure forecast accuracy.

### Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

Akari Asai, Zeqi Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi. ICLR 2024.
[Official paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/25f7be9694d7b32d5cc670927b8091e1-Paper-Conference.pdf),
[full author HTML](https://arxiv.org/html/2310.11511).
Type: method. Priority: A for separating retrieval from usefulness.
Read: sections 3.1–3.3 and 5.2, critic supervision and inference ablations.

The system distinguishes whether retrieval is useful, whether a passage is relevant,
whether generation is supported, and overall output utility. Critic supervision is
distilled from GPT-4 rather than assumed free of supervision. Adopt the separation
of physical relevance, data support and realized correction benefit. Replace text
judgment with chronological numeric errors for the forecasting task. Do not call
the existing numerical utility table a reproduction of Self-RAG.

### HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models

Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su.
NeurIPS 2024. [Official record](https://proceedings.nips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html).
Type: method. Priority: B for structured relations, not a numeric intervention rule.
Read: sections 2.2/2.3, 3, retrieval/QA results and cost discussion.

OpenIE builds an entity-relation graph; query-linked nodes seed Personalized
PageRank and passage ranking. This makes multi-document associations retrievable
without repeated reader calls. Adopt explicit links among event, observable
predicate, expert version and evidence record. Graph connectivity does not prove
causality or independent statistical support. With only four event families here,
LLM-built graph extraction adds complexity without a demonstrated retrieval need.
We therefore do not add PageRank or a graph neural network.

### A-Mem: Agentic Memory for LLM Agents

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang.
NeurIPS 2025. [Official record](https://proceedings.neurips.cc/paper_files/paper/2025/hash/19909c36f51abc4856b4560aff3d36d6-Abstract-Conference.html).
Type: method/system. Priority: A for atomic memory and update semantics.
Read: sections 3–4.6 of conference PDF, note construction, linking, evolution,
LoCoMo/DialSim evaluation and link/evolution ablations.

Memory notes retain timestamps, content, tags, context and links. New notes can
alter old descriptions; ablations separate linking from evolution. Adopt atomic
source-preserving cases, but distinguish semantic edits from changes to executable
outcomes, readout or base. Any executable change requires new benefit evidence.
Do not import conversational-memory QA gains or reported microsecond retrieval
numbers as evidence of wind forecasting improvement. Its LLM-driven memory update
is not implemented by our timestamp ledger alone.

### A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces

Mingxuan Du, Benfeng Xu, Chiwei Zhu, Shaohan Wang, Pengyu Wang, Xiaorui Wang,
Zhendong Mao. 2026 arXiv preprint, not verified as a peer-reviewed top-venue paper.
[Primary source](https://arxiv.org/abs/2602.03442).
Type: method/system. Priority: A for direct overlap, venue status kept explicit.
Read: sections 3–4, tool-removal ablations, limitations and failure analysis.

Keyword search, semantic search and full chunk reading expose different information
granularities to an agent. Its ablations and retrieval-token accounting test these
interfaces separately. Adopt progressive evidence access when needed, but do not
identify text chunk granularity with event/power-condition partitioning: those are
different operations. Hierarchical retrieval by itself is already established.
The paper's QA benefits do not establish transport of event correction gains.

## Additional screened work and reading boundary

- [Spatio-temporal short-term wind forecast: a calibrated regime-switching method](https://pmc.ncbi.nlm.nih.gov/articles/PMC7422960/), Annals of Applied Statistics 2019: inspected method/rolling procedure. Regime duration and rolling calibration support a time-aware hypothesis; not counted as a new energy-venue paper.
- [Nearest Neighbor Multivariate Time Series Forecasting](https://ieeexplore.ieee.org/document/10752995/), IEEE TNNLS 36(7), 2025, DOI 10.1109/TNNLS.2024.3490603: verified published status and abstract; full-method comparison still needed. Corrects our earlier preprint-only label.
- [GNA](https://arxiv.org/abs/2609.36281): abstract-level overlap flag for granular retrieval/trust gating; preprint.
- [HippoRAG 2](https://arxiv.org/abs/2502.14802): metadata/abstract screening, not a deep-read claim here.
- [Physics-informed reinforcement learning under extreme events](https://doi.org/10.1016/j.apenergy.2024.124068), Applied Energy 2024: publisher abstract/highlights only; full text unavailable through the fetched publisher route. Not used to specify an RL implementation.
- [Real-time Error Compensation Transfer Learning with Echo State Networks](https://doi.org/10.1016/j.apenergy.2024.124893), Applied Energy 2025: screened abstract; unusually large reported gains require protocol audit before borrowing.
- [CPLLM-WPF](https://doi.org/10.1016/j.apenergy.2025.126912): publisher dataset/conclusion inspected. It uses NWP, numerical reprogramming and LoRA; not directly reproduced with our current inputs.
- [Time-series prompts for foundation wind models](https://doi.org/10.1016/j.apenergy.2025.126535): abstract-level overlap flag, no independent novelty in adding trend prompts.
- [Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html): primary abstract inspected; no conformal guarantee claimed for our readout.

## Selective adoption

Implemented now: timestamp-bound executable cases; outcome-maturity checks;
weather-block then turbine probability mass; fresh utility evidence after the
readout changes; exact baseline fallback. No LLM-created gain labels.

Rejected now: more text for its own sake, semantic links treated as causal effects,
adding all popular modules, importing future NWP not present in the dataset,
calling four structured event tables a proven self-evolving agent.

The research hypothesis is our inference from sources plus training diagnostics:
balancing donor turbines does not balance shared weather episodes. A valid expert
identity is necessary for evidence reuse, but insufficient for temporal transfer.
Whether the new weighting improves future forecasts remains an experimental
question, not a claim inherited from any cited paper.

## Completed mechanism probe

Server training-only probe completed on all three folds at fixed seed2024.
Same frozen bases and memory cases, daily block length24, unchanged rules and
bounds, no outer-validation/test inference and no neural retraining.
Episode weighting worsened both early/late training MAE and RMSE in fold0,
improved both in fold1, and produced no actions in fold2. This does not support
unconditional deployment or expanding only fold1 after inspecting the result.
The new expert stays opt-in. Artifact:
`output/kelmarsh_hourly/model_review/episode_probe/audit.json`.
