# X1-E1 R1：第95次局部审查

【最终评级】：A（限本轮容差改动及本地验证），通过，无阻断。
【必须立即修复 Top 5】：无。
【需重跑的实验与需补充的单元测试】：Cursor按决议用新提交、新目录执行E1 HoT-only；本轮不运行服务器任务。

## 详细核对
- scripts/run_x1.py:821 dense_equivalence：truth和anchor使用协议E1.equivalence_kw=0.01；x21/x50/gF/gF_mask仍用1e-5，rtol=0，shape检查不变。
- anchor是起报时原始小时功率，train_g2b.py:122取raw[starts+23,-1]并转float32；单位为kW，非归一化输入。只改变相等性门禁，不修改数组或标签。
- tests/test_x1.py:52覆盖anchor差0.000244140625通过、0.0101失败；truth超限失败，x50差0.000244140625仍失败。
- run_e1仍按(issue,turbine)对齐旧窗口，并独立检查预测差≤0.01；本次未改变推理、模型、访问年份或指标口径。
- 历史r7误差归因来自Claude交接，本轮未重查服务器；本地通过不等于真实HoT E1已完成。

## 验证与归档
- python -X utf8 -m unittest tests.test_x1：12项全部通过，0.389秒。
- output/x2/x2_655e921：22个JSON可解析、2个日志；仅归档这24个文件，不提交数组或PID。
- A2/report2的launch脚本和协议SHA与git提交655e921原始内容一致。
- 本地100个registry文件SHA、registry本身及report2的results/dev_decisions完成哈希全部一致。
- dev_decisions为NEGATIVE_STOP_A_LINE，passing=[]、near_pass=[]；与DECISIONS停止结论一致。本轮未另做bootstrap复算。
- 不提交论文草稿、路线图改动、旧B归档或大型未跟踪文件；未连接服务器，不干预B2和同提交A。
