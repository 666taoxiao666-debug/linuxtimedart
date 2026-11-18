# X3-code R2/2：第99次复审（2026-11-18）

基线310f7d0；只复审本轮diff、R1决议和直接依赖的数据语义。不上服务器、不运行真实模型。
结论：R1-B1–B5关闭；新增防御校验缺陷R2-B1按唯一最小修法直接修复后放行。Cursor必须先用同一提交跑x3smoke，再进入正式链。

## R1落实
- B1/B2/B3及B4时间边界：Claude已明确复核310f7d0修复；本轮未回退分组隔离、V1/V2冒烟、身份哈希或真实标签时刻门禁。
- B4缺测先验：mask_flagged_upstream仅处理11个上游通道，按逆变换后的missing标志置NaN；q1_jobs只在验证上下文生成独立masked文件，不进入Q1候选；Q4按登记哈希读该文件。
  证据：prepare_g1_batches.py:383–384、train_g2b.py:95–96均在任一上游值缺测时置标志；data_provider/kelmarsh_multiyear.py:46起的窗口规则要求历史SCADA全有限，未填补SCADA。旧小时表缺失处保留NaN。
- B4信息集：新增24小时原始GFS历史，111列含固定类别列；gfs_binding核对x50通道20，与prepare_g1_batches.py:nwp_input一致。mean/scale与A3上下文逐值核对；GFS统计量仅作绑定检查。
- B4预算：目标窗口最早输入小时也必须≥D-k，全部标签严格<D；masked C_zs保留512小时预算门禁。coverage按场分别保存且登记哈希，避免覆盖。
- B5：协议v2、方案和报告统一为Q2-FT20k固定抽样预算；单序列训练不称为分组赢家微调，不把负结果外推至全量微调。预检冻结模式和V1 MAE选点未改变。
- S1：增加训练预算、实际Trainer设置记录；S2：状态脚本核对run_x3.py，适配launch中包含该命令的链shell。

## 本轮意见
- R2-B1【阻断】【确定缺陷】【已按最小修法先行】gfs_binding在有效历史对应的x50值为NaN时，NaN>atol为False，错误绑定会被当作通过；nwp_mean/scale为NaN也同样绕过 —— scripts/run_x3.py:gfs_binding。
  唯一修法：比较前拒绝非有限通道和非有限/非正尺度，保留原始历史NaN作为合法缺测；加回归，不改数据、阈值、实验臂。置信度高。
- R2-S1【建议】【确定缺陷】“约17窗/步”只是除法近似；SDK累计至≥256才停，每窗15行时实际每批18窗/270行，1000步且单进程累积1时为18000窗口次；随机有放回抽取不能称遍历完20000窗口。
  不改变注册步数/批量；论文和运行解释按实际回执说明，预算JSON的approx字段不能当实测。DECISIONS“够一遍”需改为预算量级说明。
- R2-S2【建议】【不确定】同意保留已登记31/50开展探索性比较，但不同训练预算、迁移混合及特征集下，A1选中超参数不保证仍提升L_kX，不能声称偏差“只会有利于对照”。
  例如两组参数在A1排序为a优于b，在少样本迁移下可反转；账本应写“存在既往V1暴露，方向未经验证”。不重选参、不增加臂；Q4不得升级为无暴露的独立确认结论。

## 验证与执行边界
原本轮本地tests/test_x3.py + tests/test_x2.py为51项通过；R2-B1修后52项通过（30+22，7.49秒）。
launch/status的bash -n通过。真正Chronos/LightGBM接口、内存、真实数组绑定由同提交x3smoke验证；本地单测不替代该门禁。
执行顺序：x3smoke成功→x3gpu；正式q1完成后x3q4可与q2并行；q2结束且q4完成后x3final。失败保留证据，不自行重试。
Claude后续只需复核直接修复并修正S1/S2措辞，不进入第3轮代码审核；不要求等措辞修订才做已批准的探索性运行。

SDK计数依据（锁定2.3.2官方源码，_generate_train_batches）：https://raw.githubusercontent.com/amazon-science/chronos-forecasting/v2.3.2/src/chronos/chronos2/dataset.py
