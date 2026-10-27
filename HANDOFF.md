# HANDOFF — Claude ⇄ Codex 唯一交接文件

规则见 AGENTS.md：两节各由一方整节覆盖、不追加，每节不超过 40 行；历史用 `git log -p HANDOFF.md` 查。

## Claude → Codex
<!-- 由 Claude 覆盖。最后更新：2026-10-27 第 2 次（追加第 6 条） -->

**已读 v29（c6e1b72）及瓶颈诊断**（2021、2022 自此都是开发数据；复现：`python output/kelmarsh_hourly/operating_state_v29_review/claude_v29_bottleneck_diagnosis.py`，未跟踪的复核目录）
- 模型本身没有发现缺陷：不过拟合（带 dropout 的训练损失 0.296 高于验证等价损失 0.263），约 10 个 epoch 后就平台；各臂差距在 2% 以内；校准斜率 0.91–0.97；3 seed 平均只再降 0.9 kW。
- 误差与持续法误差相关 0.90–0.93，全场共同部分占 73–80%。2022 年线性叠加 last10/上一小时最多降 RMSE 1.4%（样本内）。
- 对 2021 年残差做半年拟合、另半年检验：10 分钟级趋势、日历、昨日同时刻变化都 ≈0 或为负。全场其他风机信息仅在第 1 小时有 1–2.6%，2–4 小时 ≤1.4%。结论：风电场内部信息基本用尽，瓶颈是未来天气信息。
- 次要设计问题，留给 v30 一并处理：训练偏 MAE（0.8/0.2）导致平均误差为负（−9 到 −27 kW）而中位数约为 0，MAE 与 RMSE 的排名因此不一致；训练窗口步长 3 只用了 1/3 的窗口。

**用户决定（2026-10-27）**：先筛选上游气象站观测（MIDAS Open），有信号才写 v30；论文主指标 MAE 与 RMSE/R² 都要，v30 的规则须两者同时成立。

**本轮改动（工作区，未提交）**
- 新增 `scripts/screen_kelmarsh_upstream_observations.py`：只读、不依赖 torch、不训练。读取 BADC-CSV，只打开 2016–2022 的文件。站点规则事先固定：Kelmarsh（52.4013, −0.9428）150 km 内、2016–2022 每年风速完整率 ≥85%。特征只用发出时刻前 1 h 的观测（0 h 作敏感性）：上风向风速及 3 h 变化、上风向减本地、区域风速、气压 3 h 倾向、区域气压梯度。
- 检验方法：对 v28 TSMixer_last10（2021）与 v29 Guard（2022）的 seed 平均残差做线性修正，上下半年互拟互测；L2 计 RMSE、L1 计 MAE，都相对"只修偏差"计增益；风电场内部信息作对照组。
- 通过规则（PASS_RULE）：两年中第 2–4 小时平均 RMSE 增益与 MAE 增益都 ≥2%。
- 本地用合成站点数据做了正负对照：泄漏未来真值的阳性对照，第 4 小时增益 +33–37%，判定通过；纯噪声的阴性对照增益 ≤0，判定不通过。py_compile 通过。

**请 Codex 做**（只筛选，不训练、不写协议）
1. 审查脚本；有问题直接修，通过后提交、推送。
2. 一次性下载全部数据，免得网络不稳还要再跑一轮。CEDA 账号由用户本人注册，你不要代为注册或保存密码。
   - 令牌：用户把 CEDA 访问令牌放在服务器 `~/.ceda_token`（权限 600，有效期 3 天）。你只用 `Authorization: Bearer` 请求头读取它，不打印、不写进仓库、日志或提交；过期就请用户重新生成。
   - 站点数据：MIDAS Open "UK hourly weather observation data" 最新版（预计 v202507）的站点元数据，以及半径内站点 2016–2024 的年文件。
   - Kelmarsh 数据：2023、2024 年 SCADA ZIP，按官方 MD5 校验。
   - 封存规则：2023–2024 的文件放到单独的 sealed 目录（不放在 `--midas-root` 下），只校验大小、MD5/SHA256，不解压、不解析、不画图。它们要等 v30 冻结后才能准备 2023，2024 继续封存。
   - 记录版本、DOI、文件清单与哈希。
3. 运行：`python scripts/screen_kelmarsh_upstream_observations.py --midas-root <下载目录> --metadata <站点元数据> --eval-2021 outputs/logs/Kelmarsh/20261026/multiyear_benchmark_v28_Co2TSf/evaluation --eval-2022 outputs/logs/Kelmarsh/20261027/operating_state_v29_vEjzoZ/evaluation --output outputs/audits/upstream_screen/screen.json`
4. 报告：合格站点数、覆盖率、各年各 lead 的增益、passed。通过则由 Claude 写 v30（已封存的 2023 年 Kelmarsh 与站点数据在冻结后才解析）；不通过就如实记录。
5. 结论写进提交正文和「Codex → Claude」。
6. 追加下载（用户 2026-10-27 决定：Kelmarsh 为主，SDWPF 完整版与 Penmanshiel 作外部验证）。排在当前下载之后，同样一次下完。
   - SDWPF 完整版：figshare 24798654；Penmanshiel：Zenodo 5946808 的最新版本。都下载全部文件，按官方校验值或 SHA256 核对。
   - 封存：两者整体放入 sealed 目录，只记录文件清单、大小、哈希和许可，可以读 README 与列名，不读数据值、不画图、不统计。
   - 等 Kelmarsh 上的方案定型、外部验证协议冻结后才解析。用哪个、怎么报告由用户确认（Claude 建议两个都跑、都报告，不按效果挑选）。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-10-27，外部队列已等待；Kelmarsh真实网络中断已定位 -->

**本轮结论与修改（只下载和原MIDAS开发筛选）**
- 已读第6条；Kelmarsh主数据集、SDWPF完整版和Penmanshiel外部验证的下载安排已接受；如何使用/报告尚未决定，本轮不训练、不写v30、不选外部协议。
- 新增`scripts/acquire_sealed_external_validation.py`及`tests/test_sealed_external_validation_acquisition.py`；原MIDAS/Kelmarsh运行源码不改、不停止或重复启动。
- 外部队列只在MIDAS=`DOWNLOADED_VERIFIED`且Kelmarsh=`SEALED_VERIFIED`、两项锁释放并逐文件大小/MD5/SHA256复核后启动；无论筛选passed真假，都不据此挑选外部数据。
- 外部文件只做原始字节传输和校验，不解压，不调用CSV/Parquet解析，不读数据值、画图或统计；本轮连允许的README/列名也无需解析。
- 单队列锁、官方版本/文件清单固定、同名文件按官方ID分目录、断点复用、网络失败保留字节；校验或身份错误停止，已完成产物核验复用不重下。

**官方来源及固定清单**
- SDWPF：Figshare `24798654.v2`，DOI`10.6084/m9.figshare.24798654.v2`，CC BY 4.0；全部11文件、3,253,402,097字节（含完整版CSV/Parquet、竞赛包和附属文件）。
- SDWPF有三个同名`.DS_Store`及一个交换文件，遵照“全部下载”保留，各按file ID隔离；不打开其内容。
- Penmanshiel：从`5946808/versions/latest`官方解析为`16807304`，parent=`5946807`，页面标v3；DOI`10.5281/zenodo.16807304`，CC BY 4.0；26文件、7,489,967,756字节。
- 合计37文件、10,743,369,853字节；所有文件均有官方MD5，另记录本地SHA256；只承诺下载该版本全部已发布文件，不凭标题假定每台风机逐年完整。
- Figshare未限定版本API本次403，正式`/v2/articles/24798654/versions/2`公开API成功；未绕过权限、未关闭TLS；元数据SHA`e931813317363fe92f04df501a2d1431eb566f107920500dbeb37dd4c6825b93`。
- Penmanshiel官方元数据SHA`ff57895a9206dc0b3cb94fd4a461983e3381f2aa62529c9193e519b88fda6be5`；原始元数据与inventory保存于`outputs/audits/external_validation_download/`。
- 外部原始文件根`datasets/external_validation/sealed/{sdwpf_24798654,penmanshiel_16807304}/<file_id>/`；整套封存，等待未来冻结协议。

**验证与现有进度**
- 新16项测试服务器全部通过（本地15通过、1项Linux测试跳过）；原24+31+7项本地回归通过（仅POSIX权限项跳过）；py_compile/diff检查通过。
- 原62项服务器测试已于上一轮全部通过，CEDA令牌只在服务器私有文件和Authorization头中使用，不进入本轮配置或日志。
- 最新实查MIDAS PID994152正常：计划118站共946文件、2,307,301,313字节，已验收13文件；335为半径内历史站数，不是合格站数。
- MIDAS进度/结果仍在`outputs/audits/upstream_screen/midas_v202607/`；先2016–22开发筛选，再下载2023–24 sealed字节；未产生合格站点/增益/passed结论。
- Kelmarsh PID994153已退出，真实报错ChunkedEncodingError；312,475,648分片字节保留，总量1,420,698,115字节尚未完成；旧日志`outputs/audits/upstream_screen/sealed_download_resume_20261027_1505.log`保留。
- 新增`scripts/resume_kelmarsh_sealed_transport.py`只包装原acquire，同一锁/参数/范围分片；最多24轮、连接/超时后等待300秒再续，身份/校验/HTTP拒绝不重试；4项本地测试通过，待服务器预检后恢复。
- 服务器目标盘可用约992 GB，满足本轮原始文件与拼接空间；不占GPU，不改其他用户任务。
- `257e4e3`已推送并服务器快进；原两下载脚本及两份服务器docs SHA不变。外部队列PID997938真实存活，状态WAITING_FOR_PREDECESSORS，未传输外部数据。
- 服务器inventory SHA`0753917c1b8519d013f64ba954deca858d2033640ca1d0b2f4beb5de71a0031d`；本地同内容换行不同，勿混用文件字节哈希。
- 队列入口`python scripts/acquire_sealed_external_validation.py`；状态/日志/PID在`outputs/audits/external_validation_download/{queue_status.json,queue_launch.log,queue.pid}`；不要重复启动。

**给Claude的下一步**
- 不把元数据清单/启动当下载完成；看`queue_status.json`和`download_manifest.json`，全部完成需`SEALED_VERIFIED`及37份字节校验记录。
- 原MIDAS同工况对照、双年lead2–4 MAE/RMSE≥2%门槛不变；半年双向筛选、L1/L2不同系数及归档气象时效局限仍需披露，不称独立确认或信息天花板证明。
- 外部数据不按效果挑选，但最终使用/报告规则仍待用户确认；SDWPF此前有开发实验，后续协议须披露已接触范围，本轮不宣称其整体未见。
- 完成Kelmarsh方案及外部协议冻结前禁止解析外部数据；不根据尚不存在的筛选结果扩大训练或阈值/seed搜索。
