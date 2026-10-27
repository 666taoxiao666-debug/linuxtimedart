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
<!-- 由 Codex 覆盖。最后更新：2026-10-27，认证/服务器预检通过，两个原始数据下载已启动 -->

**结论与本轮修改：只获取数据及开发筛选，不训练、不写v30**
- 评级B：筛选方法修正见b9e7b19；真实合格站点数、天气收益和passed仍待完整开发数据，不能先下结论。
- CEDA认证已解决：用户提供的令牌通过加密连接存入服务器`/home/XT/.ceda_token`，权限0600；官方元数据请求HTTP200。只作Authorization头，未写入仓库/命令参数/日志。
- 固定MIDAS Open v202607，DOI`10.5285/d04207b551674c07801b7d4e6d883e50`，150km内335个历史站点，其中118个元数据年份与2016–24重叠；不等于合格站点。
- 新增`scripts/acquire_midas_upstream_observations.py`和对应测试：官方大小/MD5固定计划、两请求限流、Range续传、失败保留字节、单进程锁、SHA256清单。
- 公共目录请求不发送令牌，丢弃content预览；原始数据请求拒绝重定向/登录HTML，错误不回显令牌。
- 2016–22单独development根目录；2023–24独立sealed根目录，仅原始字节校验，不解析/解压/推理。
- 真实Scampton2016文件MD5`76a21c8b7dbdf08d6485f594e7f02f2c`、3,223,499字节已验收；8,784条version1记录，无冲突重复。
- 发现真实CEDA把单位置于`long_name`第4列；最小修正筛选解析器并新增回归测试，支持`deg true`，冲突声明拒绝；knots仍按1852/3600转换。
- 开发文件下载完自动运行原双年筛选，再获取sealed字节；筛选绑定输入/源码/结果哈希，同一证据只算一次，部分发布或身份不一致拒绝重算。
- 原主对照为相同intercept+issue-power-regime，双年lead2–4 MAE与RMSE各≥2%规则不变；半年双向回顾诊断不是独立确认，L1/L2系数不同。
- 本地24项获取测试（23通过、1项POSIX权限检查跳过）+31项筛选测试+7项下载回归通过；服务器62项全部通过、无跳过，真实2016文件解析为8,784条m/s观测。

**位置与异步任务**
- MIDAS开发：`datasets/upstream_observations/midas_open_v202607/development`；封存：`datasets/upstream_observations/sealed/midas_open_v202607`。
- 计划/进度/筛选：`outputs/audits/upstream_screen/midas_v202607/{download_plan.json,download_manifest.json,screen.json,screen_completed.json}`。
- 入口：新获取脚本`--development-root`、`--sealed-root`、`--audit-root`及HANDOFF上半节原`--eval-2021/2022`，凭据用默认文件，不传正文。
- 源码提交`12b82e7`已推送GitHub并服务器快进；两份服务器docs原改动SHA不变；未初始化CUDA、未停止其他用户进程。
- 确认无同类存活进程后，MIDAS首次脱离SSH启动PID994152；日志`outputs/audits/upstream_screen/midas_v202607_launch.log`，PID文件同目录`midas_v202607.pid`。
- Kelmarsh旧PID984553已因ChunkedEncodingError退出；不是数据丢失，`.download_parts`四片合计231,997,440字节，顶层.part为零只是占位。
- Kelmarsh总量1,420,698,115字节，DOI`10.5281/zenodo.16807551`，目录`datasets/upstream_observations/sealed/kelmarsh`；尚无整年完成结论。
- 核对无同类运行后，Kelmarsh原命令原目录续传PID994153，日志`outputs/audits/upstream_screen/sealed_download_resume_20261027_1505.log`；原源码、分片和锁保留。
- 服务器存储I/O压力一度约94%，预检曾磁盘等待但已通过；两个任务已确认真实进程，下载完成数/筛选尚待产物，不把启动当完成。

**给Claude的下一步**
- 等真实开发筛选产物；核对合格站点、逐年/逐小时增益、覆盖和passed后才决定v30；输入无效passed=null，不能当阴性天气结果。
- qc-version1为回顾归档，1h滞后不证明实时发布可用；不把本轮筛选当信息天花板证明。
- 不重复旧模型训练，不解析封存年份，不新增阈值/种子搜索；凭据过期再由用户更新私有文件。
