# P3 首次真实实例：准备阶段停止

运行代码1981921，run为`/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Penmanshiel/p3_v35_1981921_r1`。
2026-11-03 22:10:37 UTC（北京时间11月4日06:10:37）失败，用时684.432848秒。06:30定期巡检发现PID289435已退出。

原始SCADA准备已执行，旧年份一致性门禁拒绝`n_state_ticks`的dtype差异。只读诊断发现：

- 重建2016–2021与P1参考均664944行。
- 两列dtype不同：`n_state_ticks`、`n_capacity_ticks`由int64变为float64。
- 保持`check_exact=True`、仅忽略dtype的诊断比较通过：全部列数值、标签、NaN位置相等。
- 此诊断没有修改实际程序、数据或门禁，不能把尚未执行完成的全部准备验收记为通过。
- 真实拟合0/72；没有selection_freeze、test_evaluation_started、completed或prepared_complete，没有P3最终测试成绩。
- 授权的2022–2024原始SCADA已经读取；这次是准备技术失败，不是模型负结果。

证据：`audit/failed.json` SHA256 `094e1b63855995ca4cb01f6cd1c2669ea7703e6f31245c3fa45a05ee86c45d14`；`failure_run.log` SHA256 `29a41fa50e874775e5b14b09dbb43140331baf77c9a4ffef1b99420a00021bd3`。

本轮未修改源码/协议、未删除产物、未重启。修复和恢复提案见审查文件R1-P1，须明确批准后推进。故障监控p3-30已删除，避免对停止实例重复通知。
