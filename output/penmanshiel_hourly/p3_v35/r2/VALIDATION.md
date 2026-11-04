# P3 r2 启动验收

- 运行代码：2894746；协议14ec00a，LF SHA256 `a99b7ca91910b159088991bdbc680ba95ed240410e02c414f3e68ada22fe8926`，未修改。
- 服务器命令：`python -m unittest discover -s tests -p test_train_penmanshiel_p3_v35.py -q`；18项通过，4.619秒。
- 合成smoke：三折24拟合，33.782128515秒，source_unchanged=true；不计正式预算，不作科研结果。
- 修复后只读复算r1旧年份：664,944行、53列严格一致，标签修改0；仅两计数列比较副本统一Int64。
- P1参考CSV SHA256：`c9f7e803d4da8e7b753f35c338eff3f787017357993423fc6c45e5b10ac3b3e6`。
- r1失败文件SHA256：`094e1b63855995ca4cb01f6cd1c2669ea7703e6f31245c3fa45a05ee86c45d14`，原样保留。
- 恢复前注册SHA256：`5fbbf2d8a0158b2b8dea2ee179d00973a0b8f0d2c09dff7ba6304576aa1e6b18`；dry-run不改注册，只有一次既有attempt。
- 启动前50个source_code文件均与2894746 Git blob SHA256一致，服务器worktree干净、无P3运行进程，GPU0可用24,089MiB。
- 新worktree：`/home/XT/nuist/pythoncode/TimeDARTFirst_P3_2894746`。
- 新run：`/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Penmanshiel/p3_v35_2894746_r2`。
- PID298517；完整启动参数见audit/launch.json；脱离SSH运行，OPENBLAS_NUM_THREADS=2、OMP_NUM_THREADS=2。
- 启动后进程存活、launch存在、failed不存在；正式准备/72拟合/最终评估尚待完成，不把合成指标当正式成绩。
- r2为第二个也是最后一个实例；失败停止，不重启或补跑。30分钟监控p3-r2-30，正常保持安静。
- 本目录只存元数据，r1的根目录记录未覆盖；没有复制CSV、权重或预测数组。
- 本地复制核验：launch SHA256 `ca426c9628d736db627ecee8e14b71f18df0486fe9d1d51fc59781df307c16f3`；smoke_completed SHA256 `c43689fb49c483433f738b82336e066b4be04883dba246806dd67f08207a0e11`，与服务器一致。
- `collab_check.py --codex`退出1：只有第7项失败，历史r1的三份源码SHA与已批准修复后的当前源码不同；检查器没有区分准备失败历史实例，且固定深度glob不包含r2嵌套launch。它不是r2源码校验失败；r2的50份源码已经独立逐文件核对Git blob。
- 未为通过检查而删除旧审计、改旧SHA或放宽冻结协议；通用检查器的历史实例策略单独交接，未修改。其余检查项通过；待审diff为空产生1项警告；git diff --check通过。
