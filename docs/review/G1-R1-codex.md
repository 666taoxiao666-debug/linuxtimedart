# G1 模块 R1 审查（第69次，2026-11-11）

【最终评级】：C（待缺测插值规则修正；仅模块测试，不批准训练）。
【必须立即修复 Top 5】：R1-B1空拟合批次；R1-B2风场轨迹部分缺测。
【需重跑的实验与需补充的单元测试】：无实验；补B1回归及严格parity/梯度/功率通道测试；B2待作者处理。

## 详细问题清单

- R1-B1【阻断】【确定缺陷】【已按最小修法先行】空归一化迭代器令total=0，仍将NaN统计写入并置normalized=True。
  - P2；位置：models/NWPDisplacement.py，G1TSMixerModel.fit_station_normalization的mean=s1/total。
  - 反例：fit_station_normalization([])；后续预测均NaN。最小修法为total=0时报错，保持原状态。
  - 验证：空迭代器及零行批次应ValueError、normalized仍False；置信度高。
- R1-B2【阻断】【确定缺陷】displacement_block的num/wsum在缺一个插值端点时变成单端点外推，全缺时仍输出lambda*b_F，不能视为定稿线性插值轨迹。
  - P1；位置：models/NWPDisplacement.py:304，displacement_block的g_hat计算。
  - 反例实测：delta_F=0.5、两端值10/20、第二端mask=0，返回10；全部gF_mask=0且观测加10时，四步全为3.8、confidence=1，缺少对应的有效轨迹标志。
  - 影响：允许部分缺测的batch会向M提供非定稿修正轨迹，原测试仅覆盖站点缺测/整周期缺失。
  - 修法由Claude确定：明确有效端点不足时的填充值及有效性传播，并补回归；不擅自新增通道或更改五臂信息范围。置信度高。
- R1-S1【建议】【确定缺陷】原parity未设rtol=0，实际并非绝对误差1e-8；梯度只要求有限且M两标量之和非零，封装只核形状。
  - P2；位置：tests/test_nwp_displacement.py，Torch两项测试。补绝对容差、逐参数张量非零、实际捕获功率末通道；不改变模型规格。
- R1-S2【暂缓】【不确定】单周期身份、观测截止、拟合段偏差扣除及切分尚无数据脚本，模块仅接收数值与age，不能证明端到端无泄漏；等GFS到位再审。

已核：lead区间[3,18]、整窗shift下界、共享核形式、28/35通道、phi599参数、M7两站反例；五臂未训练。
服务器原始14项全通过（0.683秒），环境timedart，CPU；隔离目录/tmp/g1_review_69，无真实数据访问。
隔离空包仅避免models/__init__.py的无关导入，TSMixerBasicSmall为仓库原文件；不等价于未来训练入口集成测试。
修后服务器15项全通过（0.103秒），本地12通过3缺torch跳过；无需重跑实验，因为本轮没有实验。
模型SHA256：cf5183fee6955dda47d269eb6ec502f1424d4c87b00da11cbad6deb696148460；测试：b2ee618fc19868be8fe8f980453aa07bb84b8ab47839c753d07e974f5e3348a5；本地与服务器一致。
测试中的geo显式复制为float64后做绝对1e-8比较，避免构造时float32舍入影响数值一致性断言；生产模型仍float32。
