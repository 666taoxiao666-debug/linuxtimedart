# X2-A2 R1/2：第92次

结论：需修改，剩2条阻断；本地测试通过不等于执行门禁通过，当前不交Cursor正式运行。

- R1-B1【阻断】【确定缺陷】observed_wind以标准化值0判缺测会删除有效历史 —— 原prepare_g1_batches.nwp_input把有效(h-mean)/scale和缺测均可编码成0；h=mean=8时反解为NaN，VB原本恰3个有效小时可被判不足3小时。必须从已批准的逐小时历史数组按issue取过去6小时及有效性，不能凭数值猜mask；补有效风速恰等于均值和真实缺测的对照测试。置信度高。
- R1-B2【阻断】【确定缺陷】run_a2一致性门禁没有在所有拟合之前完成 —— for farm内检查后立即run_a_farm；若Penmanshiel样本或HoT曲线不一致，Kelmarsh已拟合/测试推理。需先完成三场全部预检再进入拟合循环，预检结果保存并在执行时核对；补最后一场不一致时fit调用数为0的测试。置信度高。
- R1-B3【阻断】【确定缺陷】【已按最小修法先行】run_report2把TECHNICAL_FAILURE变成NEGATIVE_STOP_A_LINE —— CI宽度0时gate_decisions返回技术失败，但passing/near均为空；现汇总前拒绝该状态，由现有异常处理写failed.json，不写负结果或继续测试。补退化回归测试。置信度高。
- R1-S1【建议】【确定缺陷】【已修复】status_x2未读取A2/completed.json，已完成A2仍显示result=-；补入该标记。
- R1-S2【建议】【不确定】VN/VB对负校正风速截到0，计划公式未显式写该操作；请Claude在文档明确此边界约定，非新一轮阻断。

核对：VN/VR只用当前观测与起报时可用GFS；VB取过去6小时，无读取未来观测路径。风速反归一化参数对应K/P scaler[0]与HoT mean/scale[0]，新增绑定检查位置正确。
Bonferroni通过：三变体z=NormalDist.inv_cdf(1-.05/6)=2.3939797998；单场95%区间用于原SE估计，合并区间使用校正z。联合通过两个门槛仍仅限开发证据，不纠正历次探索选择。
报告顺序通过：开发决策写盘后才进入测试统计；参照commit/协议锁定，预测registry校验，样本三键比较已实现。
GFS曲线比较目前仅points allclose、未校验旧curves文件本身SHA；建议按dev_freeze中的curves_sha256核验后比较机组集合/样本数/回退来源，避免误用被改动曲线。
第一轮JSON52个、日志3个及DECISIONS已先行原样存档3c135b7；不提交npy/npz/权重，原NEGATIVE不改。
初始本地25项：X2 14与Legacy numpy6通过，torch5跳过，8.419秒；未访问服务器或干预B线。
修复后26项：X2 15与Legacy numpy6通过、torch5跳过，7.026秒；两个shell脚本语法通过。
需Claude：处理B1/B2及回归测试，复核B3/S1最小修复，交R2；不改阈值、种子、变体数、年份。
