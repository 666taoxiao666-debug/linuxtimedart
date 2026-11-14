# G2b 第1步改造影响诊断

2026-11-14 完成，按第83次预登记规则选择情形 A：18/21 台机组 S2≤−5%，其余3台 |S2|<5%，无不可判定机组。

拟合采用2016-01-01至2023-06-30，验证采用2023-07-01至2023-12-31；所有臂保留改造通道、共用施工排除规则。
S2仅用于机械选择预登记情形，不能据此断言AeroUp造成下降；同月份比较仍存在跨年工况混杂。

服务器：`/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/G2/g2b_retrofit_d667d1b_r1/diagnostic`。
`scores.json` 包含逐机组S1/S2及各箱样本数、中位数；`status_distributions.json.gz` 是原始完整状态分布JSON的无损压缩副本。
未压缩状态JSON不提交Git；其服务器SHA256为 `92385c5dd44678dd600d0aeae6ba0abd5c75285a8a00b86e07edb676792abf3d`。
本地解压后143,689,176字节，SHA256与服务器一致，见retrieval_verification.json；压缩副本完整保留全部字段分布。
状态分布是各月导出记录的频数，边界重复行可能计两次；仅作描述，不作输入或分支选择依据。
ActRegSt在P0/P1/P2的非缺失值均为NotActive；BoostAva/BoostAct在已提供这些列的记录里非缺失值均为0。
字段可用年份不同，不能把未提供字段的年份当作0；这些字段没有直接支持“P1特有限功率”的解释，也不足以排除其他控制策略或传感器变化。
汇总见status_summary.json；tblSCTurFlag及其他可用性/状态列的完整逐机组分布保存在压缩原始文件中。
本轮只读取2016–2023开发数据，没有访问2024–2026观测值或解码封存GFS。
