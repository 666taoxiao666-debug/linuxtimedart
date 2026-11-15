# G2b 第84次：3a通过，3b准备完成，停止于3c之前

- 实现：bc7730e，独立入口 scripts/evaluate_g2b_sealed.py；冻结 train_g2b.py、16项源码、原协议未修改。
- 服务器：TimeDARTFirst_G2b_bc7730e；run=/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/G2/g2b84_bc7730e_r1。
- 3a：2023验证段重建30056窗，14个模型MAE与冻结记录最大差0.000030517578125kW，低于0.01门槛。
- 3b：PID837985已退出，2026-11-15 00:09:40至01:32:39 UTC，约83分钟；只准备，无模型加载、无误差或标签统计。
- 2024：59587窗，改造占比100%；GFS完整1464/1464=100%。
- 2025：56857窗，改造占比100%；GFS完整1457/1460=99.7945%，最后3周期因有效时间触及2026排除。
- 2023/2024/2025 MIDAS冲突均0；SCADA屏蔽小时0；上游缺测行占比5.7332%是整条2023–2025准备小时轴的比例，不是各测试年的比例。
- 数据包2024：490048870字节，SHA256 a4dfc88037a7b14fecccf6fbcd56e1ed65b9a3d9f996d43d0446e7a65c942420。
- 数据包2025：467597350字节，SHA256 07f34631dece57d7fafb52cfe647c58e3745fddd58e2f0909dc8018462952c02。
- 两包仅在服务器复算字节SHA，未读取数组作统计；元数据取回到server/，权重、CSV、NPZ不取回。
- 入口LF SHA cbdc1b980c652095978366b7f3f2ccdfa5160a52d59f930cb448fb7c9aee97fe；冻结SHA 981f3ae27e6792f5da71c177af1c24a0e91c4dfcd880322773fb4f9fe34607f6。
- 验证：timedart测试18通过1跳过，GFS新增准备测试5通过；解码环境旧窗口精度测试1失败已披露于../rehearsal/preflight.json，实际窗口仅在timedart构造。
- 无failed.json，无test_evaluation_started.json，无新增拟合，无2026文件打开；不安装、不下载。
- 下一步：Claude复核C1–C4实现、3a/3b审计后明确放行3c；当前停止，不推理、不合主线。
