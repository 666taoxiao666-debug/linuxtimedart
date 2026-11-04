# P3 r2 独立验收记录

2026-11-04，Codex通过SSH在原服务器对已保存产物执行只读验收，退出码0。
运行代码2894746，run目录 `outputs/logs/Penmanshiel/p3_v35_2894746_r2`；没有重训、生成新预测或修改已部署源码。

- 使用原 `validate_freeze` 核验72个唯一拟合目录、trained.json登记及选中模型；冻结清单SHA256与test_evaluation_started.json一致。
- 当前50份源码身份与selection_freeze.identity及audit/completed.json一致。
- input_identity.json登记414个输入文件，逐个大小与SHA256匹配。
- prepared_complete.json登记6个准备产物全部SHA256匹配，CSV未回传本地。
- T22/T23/T24各25个保存数组文件SHA256全部匹配（共75）；逐seed起点数组与cohort一致。
- 从保存cohort和各臂各seed预测数组调用原evaluate_arrays复算；没有调用evaluate_fold、模型预测或训练。
- 三折全部metrics、comparisons、decisions、per_seed、windows与各折result.json一致：浮点rtol=1e-10、atol=1e-10，结构及其他值严格相等。
- 各折result.json与总result.json对应折完全一致。
- T22/T23/T24窗口数分别37741/36838/13449；normal分别34073/33833/12504。
- T22和T23的R/F1/F3全部confirmed=true；T24及Z1/Z2保持descriptive，不把confirmed=false解释为失败。
- 本地239份导出元数据的大小和SHA256与原audit/export_sha256.json逐个一致。

复算复用了冻结统计实现，因此证明保存结果可复现、产物未发生身份漂移，不等同于独立重写统计方法的数学审计。
第一次只读核验尝试在hashlib.file_digest处遇到服务器Python版本不支持；改用标准分块SHA256读取后完整核验通过。该兼容修正仅涉及临时核验脚本，不触碰实验源码或结果。
监控p3-r2-30已在上一轮删除；等待Claude复核及用户同意后再合主线。
