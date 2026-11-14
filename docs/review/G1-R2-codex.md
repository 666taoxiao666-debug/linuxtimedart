# G1 模块 R2/2 复审（第70次，2026-11-11）

【最终评级】：A（仅本轮模块改动及合成测试范围通过，不代表端到端实验验收）。
【必须立即修复 Top 5】：无，R1-B2关闭。
【需重跑的实验与需补充的单元测试】：无实验；要求的18项测试已在服务器全部通过。

## 详细问题清单

- R1-B2【确定缺陷，已修复】：四步平移插值有效权重均达到1−1e-5才使用修正轨迹；否则整样本回退原风场预报乘掩码，并将错位、偏差、置信度清零。
  - 位置：models/NWPDisplacement.py，displacement_block末段；模块文档、STAGE_PLAN和DECISIONS一致。
  - 证据：缺少一个读取端点、全风场缺测且观测加10、缺测点不在读取范围三项回归通过；torch parity包含全缺测回退样本。
  - 影响：消除缺测插值的单端点外推和纯偏差伪轨迹；不增通道、不改变五臂信息范围。置信度高。
- R1-B1及R1-S1：Claude已在DECISIONS复核，既有空批次拒绝、绝对容差、梯度和功率末通道测试仍通过，不重审旧方案。
- R1-S2：保持暂缓。周期身份、真实时间对齐、拟合段预处理与封存边界，等数据脚本阶段核验；本轮不批准训练。

## 验证记录

- 本地：python -m unittest tests.test_nwp_displacement，18项中15通过、3项缺torch跳过（0.221秒）。
- 服务器：/tmp/g1_review_69；/home/XT/miniconda3/envs/timedart/bin/python -m unittest -v tests.test_nwp_displacement，18/18通过、无跳过（0.165秒）。
- CPU，OPENBLAS_NUM_THREADS=2、OMP_NUM_THREADS=2，仅合成输入；沿用R1的隔离空包与原TSMixerBasicSmall，无真实数据或训练。
- 模型SHA256：b1bacfd801db1e61c54809cf48e519b14506cf21322ed90426e10b3e18c21d4a。
- 测试SHA256：8613d875d392c6385a0165ea74276be3d797f42063e7b1870f6461020e1df53c。
- 两项服务器哈希与本地完全一致；本轮Codex未修改模型或测试，只写复审与交接记录。
