# 给 Claude：v26 审核、服务器实跑结果与下一步交接

日期：2026-10-25，北京时间。本文是单文件交接，包含代码身份、审查修复、模型结构、参数、训练结果、统计局限和后续建议。完整运行产物另有离线证据包，不需要 Claude 能连服务器才能复核。

## 1. 先读结论

**这次已经真实完成训练，不是只写好了代码。** `kelmarsh_anchored_residual_v26` 的 18 个 fit 全部完成，G1/G2 通过后按原协议评估了全部 18 个外层预测组合。I1 通过、I2 不通过。

- 相比冻结 DLinear，AnchoredFull 外层宏 MAE 从 **186.9584 降到 186.4997 kW**，RMSE 从 **268.9169 降到 268.6531 kW**。提升约 **0.245% / 0.098%**，很小。
- 三个 fold 的主比较 MAE、RMSE 配对区间都跨零；主比较 Newey–West p 值经 Holm 校正后为 **1、1、0.256658**。没有支持稳定显著提升的证据。
- 三种残差 seed 汇总后，只有一个 fold 的 Full 同时优于 PowerOnly 的内部 MAE 和 RMSE，因此 **I2 失败**。不能把本轮改善归因为辅助通道带来的可靠增量。
- 第一小时外层 MAE/RMSE 仍略差于 DLinear；收益主要在较远步长。这不是“四小时都改善”。
- 结构和训练实现可以正常工作，锚点残差明显避免了 v25 从头训练的退步；但它还不是一个高收益新底座，更不是 Wiki 创新或独立确认结果。**DLinear 应继续保留为强参照，不应凭这次小增益宣布替换成功。**

## 2. 本轮执行身份与位置

| 项目 | 身份 |
|---|---|
| 本地仓库 | `D:/Users/XT/Desktop/moxing/gpt/v2/TimeDARTFirst` |
| 服务器仓库 | `/home/XT/nuist/pythoncode/TimeDARTFirst` |
| 分支 | `codex/self-evolving-wiki` |
| 首次审核提交 | `827226df65d867f15cc0796122d70791d65b2259` |
| **实际训练提交** | **`5d0f904c21c614d2fb775b523fac955b8a17547d`** |
| 完整协议 ID | `kelmarsh_anchored_residual_v26` |
| 协议文件 | `configs/kelmarsh_anchored_residual_protocol_v26.json` |
| 协议 SHA256 | `e11f80a64ae50f3782c99ec16b7eb591099714866f7842d6b226677aeb698eb9` |
| training code SHA256 | `01ba99c3ab83a1d5ad118723adb0be9ab6c7f59c453282f5b249a46e2aa17e58` |
| 完整源码身份 SHA256 | `1ecae5eb8cd4e8ea9d361aec911b7949e01d7780d82406b384679fe16ff61d32` |
| 运行目录（仓库相对路径） | `outputs/logs/Kelmarsh/20261025/anchored_residual_v26_f0-1-2_mPaRyM` |
| 结果指针 | `outputs/logs/Kelmarsh/anchored_residual_latest.txt` |
| 启动日志 | `outputs/logs/Kelmarsh/launcher/anchored_residual_run_20261025_152305.log` |
| 预检目录 | `outputs/logs/Kelmarsh/anchored_residual_preflight/20261025T072106Z_1ecae5eb8cd4` |
| 实跑起止 | 北京时间 15:23:05 → 15:26:02，退出码 0 |
| 终态 | `COMPLETED / TRAIN_COMPLETED / REPORT_COMPLETED` |
| completed.json SHA256 | `5ad7e2610739146d0869d165006a428ebd336e227f62f59f2f1284f374a2257c` |
| 本地证据根目录 | `output/kelmarsh_hourly/anchored_residual_v26_review/` |

服务器已有另一项旧实验 `kelmarsh_24h4h_forecast_only_capacity_v26`。它与本次完整协议 ID 不同，所有描述、路径和恢复命令必须按完整身份区分，不能只写“v26”。后续文档提交不改变本次运行的训练身份。

本次只提交了 v26 新增文件及审核文档。服务器原有两份已修改文档 `docs/EXPERIMENT_REPLAY.md`、`docs/KELMARSH_MODEL_REVIEW.md` 保留；旧模型、协议、数据、v20/v25 训练结果没有被覆盖。没有停止或降速他人进程。

## 3. 对 Claude 修改的审查与实际修复

总体判断：架构成立，但原实现不能直接不经预检上线。修复了以下问题，才启动真实训练。

1. **独立 DLinear 重放的梯度状态与嵌入锚点不一致。** 独立参照原来 `requires_grad=True`，嵌入锚点是 False。本地 Torch2.7.1 上同权重实测出现 `2.980232238769531e-7` 标准化差，破坏严格 epoch0 同一性。统一将独立参照也冻结后差值严格为零；只加 eval/no_grad 不足。没有修改 DLinear 模型，也没有放宽容差。
2. **GPU 预检探针绕过资源控制。** 现已复用正式运行的设备策略：共享时本进程 PyTorch allocator 4GiB 上限、空闲时90%、可用显存不足6GiB回退CPU。没有设置整卡功耗/MPS/时钟。分配器上限不等于算力隔离。
3. **epoch0 的方法解释过强。** 允许 epoch0 入选，只会使“内部 MAE 不劣于 DLinear”容易成立；原 G1 要求严格 MAE/RMSE 联合改善，所以不会因此自动通过。保留本轮“epoch1起可选”的设计，用于显式观察训练残差是否退步；但它不是保证 G1 有意义的必要条件。只改说明文字，未改数值或决策规则。
4. **版本短名冲突检查误阻断首次预检。** 旧 capacity_v26 被原来的字符串匹配误认为重复实验。改为完整 protocol_id 检查，仍拒绝完整身份重复和身份不明的指针。旧产物不改，数据/源码/检查点哈希约束不放宽。

原未执行草案协议哈希为 `f8e0680b63d9a129c09d5664216f8c3d2d093473780f62c778db7192a93cc0b5`。说明修正后的冻结发生在任何真实 v26 fit 之前。

实际验证：本地模型8项通过；最终服务器 **83项测试：82通过、1跳过**，Bash语法通过。跳过的是依赖 Windows 本地旧镜像的可选历史测试；真实服务器 v20 A0 已另外强制重放通过，不是跳过 Torch 测试。三个 fold 的 CPU epoch0 与锚点、GPU 与 CPU 的 epoch0 功率差均为 **0 kW**。合成中断恢复和不中断训练一致，Full/PowerOnly 初始化相同。

## 4. 最新架构与全部关键参数

```text
过去24小时 × 7通道（同一冻结 scaler 标准化）
    ├─ 冻结 v20 DLinear：分解 → 两个 Linear(24→4) → 功率预测 b
    └─ TSMixerBasicSmall：2个时间/特征混合块 → 零初始化功率输出头 → 残差 r
输出 = b + r → 使用原目标 scaler 还原为未来4小时的 kW
```

模型工厂：`models.DLinearAnchoredResidual.Model`。输入 `[B,24,7]`，输出 `[B,4,1]`。

| 模块/设置 | 本轮固定值 |
|---|---|
| 7个输入通道 | Wspd、Etmp、Ndir_sin、Ndir_cos、Pab_mean、RotorSpeed、power（功率槽6） |
| 冻结锚点 | 每fold对应 v20 seed2024 DLinear；moving_avg25，端点填充，共享通道线性映射 |
| DLinear 参数 | 两个 Linear(24,4)，共200，全部冻结、eval、不进优化器 |
| 残差网络 | 2块 TSMixer，每块1,759参数，功率头100参数 |
| 每块结构 | LayerNorm(24,7) → 时间 Linear24→24/ReLU/dropout/skip；LayerNorm → 特征7→32→7/ReLU/dropout/skip |
| 残差参数 | 3,618 可训练；总参数3,818；无 RevIN |
| dropout / norm eps | 0.1 / 0.001 |
| 残差输出头 | weight+bias 为0；epoch0预测严格等于锚点；首步头有梯度，后续块可正常获得梯度 |
| Full / PowerOnly | 同结构、同参数量、同初始化；PowerOnly将6个辅助标准化通道置零 |
| 损失 | 标准化最终功率预测的 0.8 MAE + 0.2 MSE，四步等权 |
| 优化器 | AdamW，lr0.001，weight_decay0.0001，仅残差参数 |
| batch / eval batch | 64 / 128 |
| 训练预算 | 每fit最多20epoch，patience5，clip1，cosine T_max20，最低lr0 |
| 选择规则 | 内部原尺度 MAE，epoch1开始，严格更小才更新，平局保留更早epoch；不refit |
| 数据顺序 | 每epoch固定 `seed+epoch` shuffle；workers0；自身CPU线程2 |
| 精度与设备 | FP32，AMP/TF32关闭；本轮CUDA运行 |
| Wiki / NWP | 均未使用；没有新数据清洗、标签截断或新未来输入 |

注意：置零辅助通道也改变 LayerNorm 统计，因此 I1/I2 检验的是“辅助通道整体配置”，不是每个通道的独立因果效应。三个残差seed复用同一个seed2024锚点，不能称三个独立DLinear训练重复。

### 数据与隔离

Kelmarsh小时数据、六台风机、三段rolling fold；24小时预测4小时，train/eval stride均3。每个outer-train的前90%fit、后10%selection，scaler沿用v20的outer-train前60%拟合值。

| fold | fit窗口 | selection窗口 | 原DLinear内部MAE | 原DLinear内部RMSE |
|---|---:|---:|---:|---:|
| 0 | 5285 | 696 | 153.039531 | 232.036950 |
| 1 | 6341 | 816 | 167.181970 | 246.575828 |
| 2 | 7397 | 657 | 162.553075 | 240.273564 |

窗口数不是独立天气样本数；多台风机、重叠预测共享天气。历史 selection/outer 都已被多轮开发查看，不能称独立确认。锚点曾在相同 selection 选 epoch；残差的 fit 预测也不是 OOF。数据类会读取CSV再排除封存后缀，准确承诺是封存后缀不参与特征、scaler、窗口、推理和评分，而不是文件字节完全没被读过。

锚点文件 SHA256（全程不重训）：

- f0：`119bcc13b0c51ac66a06e8426f888166dcbef7730dc83115c2ec55cfc9e4b85d`
- f1：`ef3caf4d03fe3ca2297a40ae380d732beba5707084427e78cd0dc3f4b7679121`
- f2：`eb4a318751ea336a6afab17bb1ea4929e11edaf829097286d88056f5304dc482`

## 5. 门槛结果与训练矩阵

G1：seed2024 的宏 MAE/RMSE 同时严格优于 DLinear 和 Persistence，至少2fold同时优于DLinear，首小时宏 MAE/RMSE不劣。实际 **通过**。I1 **通过**。

G2：按原方案追加2025/2026，先种子内平均再fold宏平均；至少2fold和2seed同时优于DLinear，宏联合改善、首小时约束成立。实际 **通过**，因此只评估了一次全部预声明outer矩阵。

I2：Full 内部宏 MAE/RMSE虽优于PowerOnly，但fold0和fold2的 MAE 不如PowerOnly，仅fold1同时改善，未达至少2fold。实际 **不通过**。原协议I1/I2是归因条件，不是预算门槛，不能在运行后改成停止规则或声称违规扩展。

| fold/seed | Full best/完成epoch | Full内部MAE/RMSE | PowerOnly best/完成epoch | PowerOnly内部MAE/RMSE |
|---|---|---|---|---|
| 0/2024 | 3/8 | 151.2504 / 229.7260 | 5/10 | 152.8319 / 234.5607 |
| 0/2025 | 1/6 | 153.8133 / 232.7650 | 7/12 | 151.8109 / 231.9697 |
| 0/2026 | 1/6 | 153.1845 / 232.4818 | 3/8 | 152.5526 / 232.5783 |
| 1/2024 | 3/8 | 165.7732 / 246.0321 | 1/6 | 167.5695 / 246.7745 |
| 1/2025 | 1/6 | 165.9242 / 243.4109 | 2/7 | 167.5312 / 247.0816 |
| 1/2026 | 8/13 | 166.0844 / 246.4539 | 1/6 | 166.7176 / 245.7036 |
| 2/2024 | 1/6 | 162.6902 / 239.8403 | 4/9 | 161.9480 / 239.0532 |
| 2/2025 | 2/7 | 163.4238 / 237.5485 | 4/9 | 163.7307 / 241.1473 |
| 2/2026 | 3/8 | 162.7144 / 238.8369 | 2/7 | 162.7888 / 240.5809 |

没有新增DLinear fit，没有超过18fit预算，合计完成142个epoch，没有把完成的fit重新训练。完整best/last/epoch0、训练曲线、checkpoint、契约和哈希保留在各fit目录及 `report.md`。有的best仍劣于anchor，按协议原样保留，没有按outer结果回退或删fold。

## 6. 外层全部结果：不能和内部0.7001的R²混同

以下先计算每fold/seed的指标，再等权宏平均；DLinear/Persistence为3组，残差分支为9组。MSE单位kW²，其余误差单位kW；宏RMSE不是宏MSE再开根号。

| 模型 | MAE | MSE | RMSE | R² | MAE Skill % | RMSE Skill % | ±5%容量命中率 % | ±10%容量命中率 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 189.7240 | 78046.2304 | 277.1769 | 0.6492 | 0 | 0 | 44.7615 | 66.8215 |
| 冻结DLinear | 186.9584 | 73276.1614 | 268.9169 | 0.6704 | 1.2107 | 2.8223 | 44.8063 | 67.5848 |
| **AnchoredFull** | **186.4997** | **73100.1655** | **268.6531** | **0.6711** | **1.4214** | **2.8964** | **44.8725** | **67.5934** |
| AnchoredPowerOnly | 187.4691 | 74038.2976 | 270.2522 | 0.6670 | 0.9784 | 2.3697 | 44.7168 | 67.4090 |

Skill参照Persistence，先每组算再平均；若用宏MAE之比，Full为1.6995%，是另一个统计量，不应混写。容量2050kW，所以±5%是绝对误差≤102.5kW，±10%是≤205kW。**R²=0.6711不是“67.11%的预测准确率”。**

内部宏Full为 MAE160.5398、RMSE238.5662、R²0.7001；这是选epoch用过的selection，不能拿来替代上表外层数值。

### 外层逐fold（残差先平均三个seed）

| fold | DLinear MAE/RMSE | Full MAE/RMSE/R² | PowerOnly MAE/RMSE | Persistence MAE/RMSE |
|---|---|---|---|---|
| 0 | 190.2479 / 274.9328 | 190.5307 / 275.8827 / 0.649865 | 191.1123 / 277.4237 | 195.2490 / 288.3496 |
| 1 | 155.3214 / 228.3241 | 155.1949 / 228.3604 / 0.750381 | 155.0794 / 228.4109 | 153.2052 / 229.9349 |
| 2 | 215.3060 / 303.4939 | 213.7735 / 301.7162 / 0.613152 | 216.2156 / 304.9220 | 220.7179 / 313.2461 |

宏改善主要来自fold2；fold0两项都退化，fold1只改善MAE而RMSE略差。不能说三折稳定提升，也不能只报fold1约0.75的R²代替全体宏R²。

### 四个预测小时

| 时距 | Full MAE | DLinear MAE | Persistence MAE | Full RMSE | DLinear RMSE | Full R² |
|---|---:|---:|---:|---:|---:|---:|
| +1h | 130.7814 | 130.6823 | 129.2515 | 184.5006 | 184.4935 | 0.8469 |
| +2h | 172.0692 | 171.8816 | 172.9889 | 245.3874 | 245.7055 | 0.7208 |
| +3h | 206.0771 | 207.0921 | 211.7005 | 293.7919 | 294.0794 | 0.6011 |
| +4h | 237.0710 | 238.1777 | 244.9552 | 328.6175 | 328.9964 | 0.5123 |

Full对DLinear的MAE增益分别约−0.0991、−0.1876、+1.0150、+1.1067kW。第一小时不仅未超过DLinear，MAE也没有超过Persistence。内部首小时门槛不能保证新时间段继续成立。

Full外层平均偏差MBE为−31.7302kW，DLinear为−26.9692；Full第4小时偏差−56.0527kW。残差加重了系统性低估，这个现象需要在训练专用队列上查因，不能直接根据已看的outer偏差拟合补偿。

### 配对不确定性：主比较 Full vs DLinear

正数表示Full更好。按同一预测起点跨风机聚类，先平均同一窗口上的种子损失，而非平均预测后算“集成”误差。8个起点的移动块bootstrap，5000次，seed2024；Newey–West lag7，近似双侧正态检验。

| fold | MAE增益 | MAE 95%CI | NW原始p | Holm p | RMSE增益 | RMSE 95%CI |
|---|---:|---|---:|---:|---:|---|
| 0 | −0.283 | [−1.826,+1.259] | 0.715950 | 1.000000 | −0.951 | [−2.672,+0.679] |
| 1 | +0.127 | [−1.845,+2.445] | 0.906661 | 1.000000 | −0.039 | [−2.388,+2.982] |
| 2 | +1.533 | [−0.268,+3.149] | 0.085553 | 0.256658 | +1.767 | [−0.578,+3.638] |

配对RMSE使用“先平均种子平方损失再开根号”的预声明估计量；与主表“各seed RMSE再平均”存在小差异，不能要求两者逐位相同。主比较都未显著。fold2 Full vs PowerOnly 在外层呈较强正信号，但它是次级局部比较，不能覆盖预声明I2失败或代替主比较。

fold0/1/2外层只有195/149/176个不同issue时间，覆盖26/22/24个UTC日；f1有最大63小时缺口，f2最大33小时。8个起点并不总等于连续24小时天气块。这些区间未覆盖多轮开发选模型带来的不确定性，也不是独立新场站泛化证明。

## 7. 本次额外取得的训练侧诊断：v25为什么输

只用冻结v25/v20权重在fit和selection重放，未训练、未访问outer/test。诊断位于本地证据根目录下 `fit_side_diagnostic_v25_20261025_v26review/diagnostic.json`。

| fold | DLinear fit MAE | v25 Full fit MAE | DLinear selection MAE | v25 Full selection MAE |
|---|---:|---:|---:|---:|
| 0 | 154.9917 | 152.9621 | 153.0395 | 159.2802 |
| 1 | 159.8570 | 153.7972 | 167.1820 | 168.4828 |
| 2 | 159.0977 | 152.6559 | 162.5531 | 170.9832 |

这表明v25不是“训练误差也降不下去”：Full总体在fit更好，selection却全部更差。更符合训练收益不能稳定转移、新时间段误差结构不同的现象；不能单凭这张表区分过拟合、天气漂移、优化或正则化各自的因果贡献。

进一步有两点具体证据：

- **首小时fit MAE本来就更差**：Full约113.260/112.116/112.315，DLinear约106.086/109.723/111.735。统一四步目标没有保护最容易被历史功率预测的短步长。
- 以最后观测功率分工况，低于10%容量时，Full在三个selection都更差约6kW；60–90%容量时也更差，但每fold只有36/66/84窗口。高爬坡fit受益，selection反而退化。不能据此直接在outer设分段门槛，稀少高功率工况也不能夸大。

“TSMixer能表示DLinear映射”排除了纯表达能力不足的解释；**并不证明只剩优化问题，更不证明数据没有信息上限**。我们没有发现可以靠删异常标签就解决的证据。

## 8. 还需要优化什么，优先级如何

**优先级1：先解释残差的训练收益为何不能稳定转移。** 本轮Full最佳epoch中有4/9在第1轮；例如f0 seed2025最好也差于anchor，最后MAE更差8.803kW。复杂度虽然小，残差仍可能在很短的内部选择段上学到不稳定模式。先做训练期严格forward block / OOF残差诊断：每步长的有益/有害幅度、平均偏移、时间块和风机稳定性。记录产生证据的实际anchor+residual参数哈希；不允许将一个专家证据拿去控制另一个重训专家。

**优先级2：查首小时保护和远步长低估的冲突。** 只有在训练专用OOF也反复出现相同模式时，才另立新协议比较“固定锚点 + 更受约束的残差”。可考虑每步残差收缩或零中心约束，但必须先定依据、预算和同等参数控制，不能拿上表outer误差直接选择系数或把首小时强制回退作为已证明正确的方案。

**优先级3：确认辅助信息究竟能补什么。** I2失败意味着现有辅助通道的增量不稳定，而不是它们毫无价值。用训练OOF检查跨时间收益，以及Full/PowerOnly的归一化统计混杂；随后再决定是否需要无参数物理历史特征、真正可用的提前发布NWP或跨风机信息。新增外部信息时必须记录预报发布时间与可用性，不能偷用未来实测风速。没有获得这些数据之前，不把“信息天花板”写成已证实结论。

**优先级4：不要继续堆同一小网络或扩大seed掩盖问题。** v25→v26改进主要来自保住一个已训练线性解。当前收益不足以证明额外3,618参数和复杂评估流程的实用价值。继续保留DLinear/Persistence；后续若提出候选，先列清楚预期改善哪些训练误差、什么结果能证伪，再一次冻结有限比较。

**Wiki暂不接回。** 先有稳定数值专家，再重新建立与该专家和实际Wiki修正器绑定的收益证据。旧Wiki证书不能迁移到新的数值底座。这次所有数字都是无Wiki数值预测，不是知识贡献。

v25/v26使用的开发窗口已经反复看过。下一轮设计应提前确定训练内验证或新的确认段；不通过改同一selection的lr/epoch/损失权重来追求正收益。新方案必须新协议，旧v26不得改写、重跑或按outer回退。

## 9. 给 Claude 的执行要求（可直接照此接手）

1. 先核对本文运行身份与离线包中的 `preflight.json`、`completed.json`、`result.json`、`inner_gate_*.json`、`paired_statistics.json`、`independent_audit.json` 和 `independent_weight_audit.json`。不要根据聊天推测服务器还在训练。
2. 先读 `models/DLinearAnchoredResidual.py`、`models/TSMixerBasicSmall.py`、控制器、协议与本轮修复；不要再声称“epoch0加入会令严格G1恒真”。
3. 先解释本轮“内部G2通过、I2失败、outer微增益但不显著、近端退化”的证据链。对根因明确区分已证实与假设，不要用“数据太脏”“模型太垃圾”替代诊断。
4. 如提出下一版，最多先选一个训练侧证据支持的机制，写新协议、单变量对照、有限预算、forward/OOF隔离和停止条件，再改代码。不能在已运行v26中改epoch0、I2或统计规则。
5. 不能删除/覆盖旧产物；不能以汇报失败为由重训；不能访问封存test、改标签或用outer标签拟合；不得停止其他用户任务。需要推送或服务器执行时，明确报告实际完成与尚未执行的项目。
6. 若Claude运行在无网络Linux沙箱，Windows密钥路径不可见、Tailscale私网不可达是环境问题。离线包足够做指标/架构/门槛审查，不需要复制私钥。实际部署由有网络的Windows环境执行；不把网络不可达说成服务器宕机。

### 快速只读状态命令

服务器仓库下执行 `bash scripts/train/Kelmarsh_launch_anchored_residual.sh --status`。本轮已经完成，**不要执行launch/resume，不追加fit**。只有图表文件确实损坏并查明原因时才考虑report-only，同时保留原报告身份。

本地复核命令（NumPy审计不需要torch）：

```text
python output/kelmarsh_hourly/anchored_residual_v26_review/audit_v26_run.py output/kelmarsh_hourly/anchored_residual_v26_review/anchored_residual_v26_f0-1-2_mPaRyM
python output/kelmarsh_hourly/anchored_residual_v26_review/audit_v26_weights.py output/kelmarsh_hourly/anchored_residual_v26_review/anchored_residual_v26_f0-1-2_mPaRyM
```

第二个审计需要torch，但仅CPU读取权重，不训练、不初始化CUDA。离线包中的 `reference_outer/f0|f1|f2/raw_predictions.npz` 是冻结v20 DLinear对照，不能用新预测替换。

## 10. 最终独立验收与图表检查

下载完成后，额外编写不导入仓库训练/统计实现的NumPy审计，**774项检查全部通过**：训练与报告清单每个已注册文件哈希、三份外层参照NPZ哈希、窗口/时间/风机/标签对齐、原尺度指标、G1/G2/I1/I2、实际请求outer发生在G2之后、18fit预算、初始化配对、MAE/MSE/RMSE时间块区间、Newey–West和Holm。总体指标复算与服务器登记值一致；统计复算差异小于1e-9。

另用CPU检查18组的init/best/last共54份锚点子状态：均符合对应v20锚点状态哈希，每份200冻结参数；每模型共3,818参数，best状态与last中保存的best_model一致，best_epoch与完成epoch一致。该审计没有初始化CUDA。

五张生成图均已实际打开检查。`forecast_trace.png`按预声明规则取最小风机编号的首个连续72小时区间，不按误差挑图；Full与DLinear大部分时间接近，较大峰谷和变化时刻仍有明显误差。`training_history.png`能看到训练损失继续下降而selection后期恶化；`paired_gain.png`显示三个主比较区间都跨零。外层逐小时图标题中的“dashed = historical v25”是通用图标题遗留，此图实际没有v25外层曲线；v25未产生外层预测，不能据此误认为存在。未为美化结果修改已注册图或重做训练。

离线包只包含本轮代码快照、报告、原始预测、少量模型权重、原参照预测及上述审计，不包含SSH密钥、原始完整CSV或封存测试数据。源代码快照应以训练提交为准；本文后续提交是结果归档，不代表额外实验。
