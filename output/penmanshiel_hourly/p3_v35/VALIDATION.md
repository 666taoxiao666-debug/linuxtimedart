# P3 启动前验收

- 运行源码：1981921；冻结协议先于解封提交推送14ec00a。
- 服务器独立worktree：`/home/XT/nuist/pythoncode/TimeDARTFirst_P3_83f44f5`。
- Python：`/home/XT/miniconda3/envs/timedart/bin/python`。
- 单测：`python -m unittest discover -s tests -p test_train_penmanshiel_p3_v35.py -v`，11/11通过，3.291秒。
- 合成smoke：`python scripts/train_penmanshiel_p3_v35.py --smoke --device cuda:0 --out /home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Penmanshiel/p3_v35_smoke_1981921`。
- smoke三折共24个单seed小拟合完成，31.098875秒；不使用真实数据；详见`validation/smoke_completed.json`。
- 50个源码文件逐个与smoke完成记录和Git HEAD blob核对SHA256一致；真实启动见`audit/launch.json`。
- 两次早期单测失败均来自合成数据生成器（日期范围、新列赋值），修复后通过，未消耗真实运行预算。
- 正式run：`/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Penmanshiel/p3_v35_1981921_r1`，初始PID289435。
- 本记录仅证明启动前验收及进程启动，不代表真实准备已通过、72拟合已完成或测试确认成功。
- 按用户偏好每30分钟只读检查，正常运行静默；失败或完成再通知。
