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
   - 等 Kelmarsh 上的方案定型、外部验证协议冻结后才解析。用户决定两个都跑；论文里怎么写之后再定。所有结果照常完整登记，不因效果好坏删改记录。

## Codex → Claude
<!-- 由 Codex 覆盖。最后更新：2026-10-28，MIDAS筛选完成；外部队列HTTP403阻塞 -->

**结论与范围**
- 原MIDAS下载和开发筛选完成，passed=true；本轮没有训练、没有写v30、没有追加阈值/seed搜索。
- 用户已决定两个外部数据集都跑；论文呈现方式待定，所有结果完整登记，不按效果删改。
- Kelmarsh 2023/2024两ZIP已SEALED_VERIFIED，共1,420,698,115字节；官方MD5与独立复算SHA256一致，未解压/解析。
- MIDAS Open v202607：946文件、2,307,301,313字节全部登记DOWNLOADED_VERIFIED；743份2016–22开发文件、203份2023/24封存文件。
- plan SHA `a4d083627893bb8cbc76095e809b819b091047f94778a221ccae0b696fa3c31f`；下载实现仍为`a787660`，本轮只更新记录。

**筛选结果（原冻结规则）**
- 150km半径历史335站；36站满足2016–22逐年风速完整率≥85%，合格站最低年度完整率89.42395%。
- 1h观测滞后主分析覆盖：2021=99.5153%，2022=99.9646%；只在完整特征窗口比较同队列对照。
- 相对同工况intercept+issue-power-regime对照，lead2/3/4的MAE收益：2021=3.62365/3.58051/3.22665%；2022=2.52397/3.10578/3.59348%。
- lead2/3/4的RMSE收益：2021=3.27478/3.30785/2.73339%；2022=2.22215/2.84872/2.95919%。
- lead2–4均值MAE/RMSE：2021=3.47694%/3.10534%；2022=3.07441%/2.67669%，两年均超过事先固定2%门槛。
- 这不是新模型outer精度或Wiki贡献；半年双向开发筛选、L1/L2不同系数、归档QC版本不等于实时可用观测，不能称独立确认。
- 2021第4小时正向半年RMSE/MAE收益仅1.26078%/1.50026%；汇总通过不表示每个方向/步长都超过2%。
- 服务器结果：`outputs/audits/upstream_screen/midas_v202607/{screen.json,screen_completed.json,download_manifest.json}`。
- screen SHA `30fdddd2ff7b82148421f27c08d478f04d97eb4de8d736db6f1c5a0d778750f3`；completion SHA `feb607aff4bd5aee552caeba26ebd67be7ef6d17b6f58607cb5455c6b349f397`。
- 本地结果副本：`output/upstream_observation_review/completed_screen/`（不提交原始数据）。

**外部队列阻塞**
- 前置验收后原队列自行启动，已退出；`queue_status.json=FAILED/HTTPError`，`download_manifest.json=INTERRUPTED`，验收0/37文件。
- 真实报错：`https://ndownloader.figshare.com/files/46005789` HTTP403 Forbidden；不是SSH失败，也不是MIDAS认证失败。
- 按授权边界没有重启403任务、没有改URL/版本/清单、没有放宽校验或跳过文件；已收分片保留。
- 外部清单固定：SDWPF Figshare24798654v2全部11文件；Penmanshiel Zenodo16807304全部26文件；合计10,743,369,853字节、CC BY4.0。
- inventory SHA `0753917c1b8519d013f64ba954deca858d2033640ca1d0b2f4beb5de71a0031d`；根目录`datasets/external_validation/sealed/`。
- 日志/状态/清单：`outputs/audits/external_validation_download/{queue_launch.log,queue_status.json,inventory.json,download_manifest.json}`。
- 20分钟监控保持ACTIVE；403未经处理不得反复恢复；已完成MIDAS筛选与Kelmarsh不重跑。

**给Claude的下一步**
- 可依据已通过的开发筛选设计v30；先冻结训练、选择/校准、实时可用性假设及2023评估协议，再申请进入2023准备；2024继续封存。
- 外部下载需先确认Figshare403的访问/传输原因和合规处理方案，未获授权不改变冻结下载行为。
- 不把36站、正向开发信号解释为最终模型收益；外部数据下载未完成，不可宣称外部验证完成。
