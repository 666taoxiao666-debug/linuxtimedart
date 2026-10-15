# Kelmarsh v7：可拒绝数值残差与共同底座Wiki

日期：2026-10-15（Asia/Shanghai）。用户明确授权诊断、最小修正、测试、推送、部署和训练。

## 问题与改动

旧f1/seed2024的OriginalTrend实际选中epoch0，预测近似Persistence。数值适配器的训练期
收益随时间减少，outer MAE反而从101.888844升至107.236914kW；大部分损害不在Wiki介入处。
因此本轮不继续只调Wiki阈值，而将冻结数值残差本身视为可拒绝的执行候选。

实现：`utils/certified_numeric_residual.py`。仅用原训练selection拟合每步的凸收缩系数；
目标为平均的归一化MAE与MSE，在二者均不劣的可行区间上求解，不做outer阈值搜索。
原fresh calibration按issue时钟分三段：

1. 数值执行证据：再分三个连续时块，每块至少24窗口、3台风机、4个不同issue，要求
   风机宏平均MAE/MSE收益均正、至少2/3风机MAE受益。任一块失败，对应步长精确回退。
2. Wiki epoch选择：固定20轮，MAE/RMSE均不劣于共同底座，保留epoch0回退。
3. Wiki执行证据：不变的旧收益阈值，但证书重新绑定实际执行的底座门和新专家。

跨边界目标移除；风机6不提供知识/门控标签。第一阶段检查点不重训。数值策略确定后，
用原memory+early训练WikiIncrement和ZeroMemoryIncrement，所有优化设置和参数量匹配；
仅后者的类型化记忆token置零。记忆置零仍保留相同知识支持mask，不能称为完全无知识。

## 预声明验收

协议：`configs/kelmarsh_certified_fallback_protocol_v7.json`。
固定fold1、seed2024，两条20epoch分支；不自动扩折/seed，不重训旧完成任务。

同时比较Persistence、OriginalTrend、冻结NumericContext、旧NumericCalibrated、旧v6 Wiki、
新CertifiedNumeric及新Wiki/匹配置零分支。报告MAE/MSE/RMSE/R²、相对Persistence的双Skill、
容量容差命中率、分步长误差、coverage、selected gain、harm、abstention、时间簇bootstrap/
Newey-West及预测图。所有评估样本保留、功率标签不改、不截负功率，不访问封存test。

Wiki需MAE和RMSE同时优于共同CertifiedNumeric、匹配置零及OriginalTrend/Persistence，才支持
有用的端到端Wiki精度贡献。数值回退获得的收益应单独归因。负结果全部保留。

## 实现自审

【最终评级】：C（工程检查通过但新实验未完成，不具备新的精度结论）。

【必须立即修复Top 5】：原先默认执行不稳定残差已修正；证书/执行对象错配以门ID和检查点
哈希防护；校准与选epoch隔离；相关窗口不冒充独立样本；数值回退收益不计为Wiki贡献。
后两项是报告边界，不代表发现新的代码错误。

【需重跑的实验与需补充单元测试】：只启动新v7的两条分支，不重跑v1-v6。已完成本地41项测试，
包含10项新门控测试及既有预测、Wiki质量、残差、哈希回放和时序支持回归。服务器测试另记。

=== 详细问题清单 ===

- [P1，已加控制] `scripts/train_kelmarsh_wiki_quality.py`的旧对照保留原始NumericContext，
  其outer跨时段退化已有真实指标支持；这不是单纯代码bug。新runner将其作为可拒绝候选，
  失败精确回退，测试覆盖全块通过要求、翻负、稀疏支持与零误差。置信度高。
- [P1，已加控制] `scripts/train_kelmarsh_certified_fallback.py`的prepare/run/evaluate：
  新base会改变增量专家的目标，不能沿用旧Wiki证书。每个新证书绑定gate ID、新增量检查点
  及源检查点，实例化/推理不读取outer标签。测试覆盖身份篡改与证据可用时间。置信度高。
- [P2，保留限制] 内部fresh calibration分段后样本更少，且天气/风机相关。固定块检查不是
  统计保证；outer已被历史查看，本轮只能提供开发证据。须保留所有CI/p值和失败结果。
  不能为正收益改变块数、阈值或seed。置信度高。
- [P2，保留限制] 新流程重新分配内部样本，旧v6与新v7差异不能作为纯Wiki机制消融。
  真正的记忆贡献只在v7内部相同底座/预算的两分支上比较。置信度高。

## 部署记录

- 模型提交：`a6fdf699cfdef9742ff20211ad53f59286f30324`，已推送并服务器快进。
- 协议SHA256：`bcbd83611d86ef3c5fe268970c85fd893246947b2d3f66e9d809acca5e96ef8d`，服务器一致。
- 本地、服务器均41项测试通过；服务器编译检查和Bash语法通过。
- 2026-10-15 09:50:32 +08已部署启动等待队列，Bash PID1943173、Python PID1943181。
  状态是`WAITING_FOR_GPU`，不是已开始GPU训练。09:41检查空闲，但09:50启动前复查时
  用户gyx的新PID1941402已经占8754MiB显存、GPU利用率100%；未与其并行训练或修改其任务。
- 启动日志：`outputs/launch_logs/certified_fallback_20261015_095032.log`。
- 结果目录：`outputs/logs/Kelmarsh/20261015/001_certified_fallback_h2_f1_s2024_ep20`。
- 指针：`outputs/logs/Kelmarsh/certified_fallback_latest.txt`。
- 队列每30秒检查全部计算进程，GPU空闲后自动继续；等待上限24小时。到期失败保留原目录，
  须核对实际进程/完成状态后`--resume`，不能新建重复实验。队列无需Codex持续在线。
- 训练侧CPU预检已通过，未读取outer目标或做outer推理：原selection拟合alpha=[1,1]，
  但后续三个固定块分别接受[真,真]/[假,真]/[假,假]，因此最终数值执行权重=[0,0]，
  回退OriginalTrend。不能因为全拒绝改阈值来放行；这就是本协议预声明的风险控制结果。
- 新内部选epoch115窗口、独立Wiki证书110窗口、数值证据115窗口，跨边界移除0；
  memory/early拟合仍为360/685窗口。Wiki分支将围绕这个共同回退底座训练。
- 新精度指标尚未产生；不以CPU预检代替训练结果，不宣称本轮Wiki贡献成立。
