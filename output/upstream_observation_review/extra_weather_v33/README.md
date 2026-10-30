# S：新增天气字段单次开发筛查

执行日期2026-10-29，记录完成2026-10-30。依据HANDOFF中Claude第7次保留的第5次批准。结论：**NOT_PASSED，不进入L，不调门槛重跑**。这是开发字段筛查，不是新预测模型的独立测试。

## 实际结果

- 唯一主组thermal_cloud，2021、1h可用性假设、h1–4平均增益：RMSE **−0.2389854639%**，MAE **−0.2089273209%**。负增益表示误差略增。
- 前半年拟合→后半年评价：RMSE −0.1583865458%、MAE −0.2887412304%；反向：−0.3181447911%、−0.1220021569%。
- 覆盖15604/15680个风机窗口，99.51530612%；原队列没有落在半年度边界需剔除的窗口，因此boundary_purged_windows=0。
- 原门槛不变：主组两项平均增益均≥1%，两个方向两项均>0，覆盖≥80%。本次并非覆盖不足，主指标和两个方向均未通过。
- thermal、cloud仅为描述性组；不因个别h正增益替换主组。

## 审查与验证

【最终评级】B：原批准的开发筛查用途通过审查，筛查统计结果未通过。没有发现需要先改脚本才能在该范围执行的P0/P1问题。

【必须立即修复Top5】无阻止此次获批执行的代码修复项；不为凑数量列问题。下列解释边界必须随结果保留。

- 只加载v28的2021队列及预测，核对completed记录中的文件哈希、时间/标签形状；没有加载2022–24残差。
- 36站身份来自既有screen，不重新选站。复用基础加载器的2016–22风压资料，登记252文件并核对原screen哈希；新增温度/露点/云字段只来自2020/2021的72文件。所有计算用于2021，未读封存的2023/2024原始观测。
- 特征只用issue−1h及更早数据；新增组与控制组在相同行上比较；控制已有11个上游特征、夜间标记和功率区间交互。
- L2与L1分别拟合、分别计算RMSE和MAE，**不是同一个训练后模型同时改善两项**。半年度双向交叉拟合包含反向开发诊断，不能冒充严格向前在线验证。
- QC version 1是归档值，1h仅为可用性假设，不证明历史实时到达；基本加载器可能先解析后筛除非候选文件，252是纳入并验哈希的文件数，不宣称整个目录每次打开均已列入结果。
- 未裁剪、插补或修改标签。线性/IRLS拟合是原批准S的诊断步骤；没有启动预测器训练、调用LLM或下载新数据。

【需重跑实验与补充单测】无需重跑。原`--self-test`本地、服务器各一次均PASS，覆盖字段单位/清洗、夜间标记、上风掩码同一性、增量信号/零信号、覆盖判定。脚本未改。后续不同设计必须另审，不能复用本授权试门槛。

## 执行与身份

远端仓库`/home/XT/nuist/pythoncode/TimeDARTFirst`，输出`outputs/audits/extra_weather_v33_review_20261029/`；UTC约11:18完成（北京时间2026-10-29约19:18）。初次SSH在密钥交换阶段断开；后续只读确认输出目录不存在、无同类Python后才重新提交启动请求。实际S执行一次，完成后无同类Python。没有停止或修改其他用户任务。

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /home/XT/miniconda3/envs/timedart/bin/python -u scripts/screen_kelmarsh_extra_weather_v33.py \
  --midas-root datasets/upstream_observations/midas_open_v202607/development \
  --screen-json outputs/audits/upstream_screen/midas_v202607/screen.json \
  --eval-2021 outputs/logs/Kelmarsh/20261026/multiyear_benchmark_v28_Co2TSf/evaluation \
  --output outputs/audits/extra_weather_v33_review_20261029/result.json
```

用nohup脱离SSH，输出run.log；此次没有单独保存应用退出码，依据完整原子结果、日志及进程退出确认完成，不把SSH退出码当应用退出码。

- S脚本：`e79c50e3355d002ec8ac06e164a90bfe718b77c012366f064c4faaf6620206e6`
- base脚本：`c68477c52fa9405b12095560c446f7bfaf160743842efeccc276717070aa214b`
- screen.json：`30fdddd2ff7b82148421f27c08d478f04d97eb4de8d736db6f1c5a0d778750f3`
- v28 completed.json：`3a0f5183642edfe028212b5738c5f198165a30d99a091ddf2a5d2e2270aac190`
- 原始result.json：`77601086222d8817df1a598a9b32f243c91c8b087ba10827d2398f32a9d7d44d`
- 原始run.log：`fe814d108c69ded32f4576cbc7ac7af003a6ad215a2c110d6b9cb7d0a28bcea1`

输入身份本地/服务器一致，结果与日志回传后哈希同远端一致。原JSON和日志原样保留；不重新计算、选择或改写筛查结果。
