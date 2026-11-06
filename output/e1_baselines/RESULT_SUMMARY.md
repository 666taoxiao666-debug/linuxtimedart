# A线 E1：拟合完成，最终评估技术失败

运行：e1_b5be60a_r1；代码b5be60a；失败时间2026-11-06 11:20:31北京时间。

64次拟合全部完成，其中10次仅开发集搜索、54次正式拟合。服务器独立核对trained.json与selection_freeze一致，并核验每次拟合登记的init.pt、best.pt、last.pt、training_history.json和validation_predictions.npz哈希，无差异。证据见r1/failure_verification.json及selection_freeze.json。

最终评估刚写入evaluation_started.json，即在读取v32参照post_test_manifest.json时抛FileNotFoundError。该文件本地存在但未跟踪，Git独立worktree部署没有带入；Codex未在启动前检查该运行依赖，是部署/预检缺陷，不是模型性能结论。evaluation目录没有预测npz，不能报告A线MAE、RMSE、R²或增益。

原实例已退出；未重启、未补跑、未改冻结协议/代码。只拷回JSON元数据，不拷权重和预测数组。原协议仅一个实例且禁止resume，已存在评估开始标记，任何恢复需要另行明确授权。

建议交Claude审查：新版本仅评估恢复，复用全部冻结拟合、checkpoint和选参；参照元数据锁哈希并前置检查；新输出目录保留原失败现场；不再训练、不调参。当前只是提案，尚未实施。

## 同轮C0结论

C0已正常完成并独立核验，按冻结标准STOP。M的90%名义区间经验覆盖81.2799%，不满足88–92%门槛；区间分数1521.0805虽满足阈值，也不能抵消覆盖失败。B1覆盖91.3723%，是唯一合格比较臂。B2/M+均含无限宽区间，不据此声称宽度或分数改善。完整结果见../conformal_c0/RESULT_SUMMARY.md。

请Claude一并复核C0有效负结果、A技术失败证据与仅评估恢复提案；本轮未合主线。
