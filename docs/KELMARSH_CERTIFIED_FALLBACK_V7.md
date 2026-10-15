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

### 2026-10-15 用户追加授权：共享GPU执行

用户随后明确要求“帮我并行运行”，覆盖原来仅空闲运行的部署约束；不覆盖模型/数据/评估协议。
不修改冻结训练协议JSON或哈希，单独用`resource_policy.json`记录共享授权及资源设置。
仅允许撤下本用户尚未训练的排队进程，再从原目录`--resume --shared`，不得停止其他用户任务。

`utils/gpu_runtime_policy.py`默认仍为idle-only。显式共享模式将本进程PyTorch allocator上限
设为约2GiB，初始需至少6GiB空闲（预算2+余量4），运行期间检查至少4GiB空闲；低于该值
停止本任务而不操作他人进程。保留2个CPU线程，提高本进程nice值，每训练批次同步后等待20ms。
这既不是GPU算力百分比限制，也不是总CUDA显存硬隔离，更不能保证他人零减速或零OOM。
未启用MPS/MIG，不改整卡功耗/时钟/计算模式。保留batch size、LR、loss、seed、epoch和全部
模型/标签不变；新可选回调只控制资源检查及执行节奏。新增5项资源测试与41项回归共46项本地通过。

共享执行实现提交：`41b34e187f2e386d57bc3c64166f9b41d9fc6773`，已推送并服务器快进；
服务器46项测试及Bash语法通过。仅向已核实属于XT、状态WAITING且未产生prepared/训练产物的
排队Python PID1943181发送SIGTERM，原目录和旧日志保留，其他进程未操作。

2026-10-15 10:26:42 +08，以`--resume --shared`成功实际启动，Bash PID1947900、Python
PID1947905；新启动日志为`outputs/launch_logs/certified_fallback_shared_20261015_102642.log`。
首次确认实际训练时，我方GPU显存338MiB，对方gyx PID1941402仍为8754MiB且正常存活；
全卡9195MiB，温度78℃。该快照不能代替峰值测量，也未测量对方训练速度变化。
首次确认时WikiIncrement已20轮完成、内选epoch13，ZeroMemoryIncrement正在第4轮；
这是内部选择指标，不是outer精度结论。资源模式及授权分别保存在原结果目录的
`resource_policy.json`和`shared_execution_authorization.json`，两者不会改变冻结协议哈希。

### 本轮完成结果

随后确认`status=COMPLETED`、exit_code=0，两个分支都完成20轮，内部均选epoch13。
对方原GPU进程仍为1941402；我方进程正常结束并释放GPU。已核验全部269个登记产物哈希，
并下载核心结果、训练历史、证书、预测数组和图；不自动扩展seed/fold。

最终WikiIncrementSelective与ZeroMemoryIncrementSelective均完全回退共同底座：
MAE=101.8888443279292kW，MSE=24171.238136705197kW²，RMSE=155.47102024719976kW，
R²=0.7832358704272058，相对Persistence双Skill约0（浮点误差量级）；±5%容量命中率64.5768%，
±10%容量命中率86.5204%。Wiki coverage=0%，不介入=100%，selected gain和harm为N/A，
不能把“没有介入”写成“介入伤害率0”。预声明联合成功条件为false。

未路由原始Wiki候选MAE101.535493/RMSE155.153206，未路由置零候选101.525857/155.149995；
置零候选还略优，因此即使看raw也不能证明记忆贡献。raw未通过预声明证书，不能在看到outer后
将其替换为正式输出，更不能修改门槛把这次实验改报成功。新执行回退避免旧数值适配器损害，
但这属于数值安全策略，不是Wiki预测精度创新已获得验证。

本地核心产物：`output/kelmarsh_hourly/certified_fallback_v7_review/`。该outer依然只是历史
已查看的开发集，无新独立确认结论。本次用户请求的并行执行已完成，未干预他人任务。
