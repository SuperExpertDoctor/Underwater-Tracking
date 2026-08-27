# 多 UUV 协同跟踪与蛇形覆盖审查验收

## 结论

本次审查在独立仓库、独立分支和离线确定性场景中完成。结论分为三层：

1. **协同跟踪可以运行并产生控制效果。** 4 艘已部署 UUV 在 360 个物理步内执行了 1,416 个受控区间，其中 1,412 个区间发生位移；融合估计得到 37 个与目标真值同时间戳的有效误差样本。跟踪 RMSE 为 488.26 m，P95 为 1,012.15 m，最大误差为 1,151.21 m。因此可以证明“控制与估计链路工作”，但不能据此声称高精度跟踪。
2. **蛇形路线生成及执行保持逻辑基本合理，但覆盖效果是部分完成。** 已分配路线的几何与负载平衡检查通过，Task 01、Task 02 和 Task 03 的采样主动声呐足迹均为 100%，Task 04 为 52.08%；但 Task 02 没有分配路线且最终仍为 `UNCOVERED`，说明足迹指标不等同于蛇形路线执行。Task 04 最终为 `PLANNED`，其 UUV 尚未执行航点。当前证据支持滚动式、按阶段覆盖，不支持“所有任务区同时完整覆盖”的结论。
3. **目标真值已从规划输入隔离。** 目标真值仅保留在传感器物理门控和评估输出中；公开规划使用有来源、有有效期的搜索先验，后续预测与意图使用融合估计历史。主动声呐命中后才公开带噪位置，过期先验及其缓存不会继续驱动规划。

正式验收的全部硬检查为 `true`，总状态为 `PASS`。这里的 `PASS` 只表示下述已定义门槛通过，不代表统计鲁棒性、实机安全性或研究指标已经充分验证。

## 验收范围

- 仓库：`D:\Air\反Q\Underwater-Tracking`
- 分支：`review/uuv-tracking-coverage-20260827`
- 基线提交：`63b13f60f7de639bed4751260c83236c67e9e54c`
- 正式运行代码提交：`5f7f95d`（目标真值隔离修复提交后）
- 配置：`configs/scenario/uuv_only_single_target.yaml`
- Python：仓库内 `.venv`，Python 3.12.13
- 随机种子：42
- 每次步数：360
- 物理步长：5 s
- 重复次数：2
- UUV 配置数量：12；正式运行中已部署并产生轨迹的 UUV：4
- 外部 LLM：禁用并采用 fail-closed 本地哨兵
- ROS、真实硬件、服务器和两个现有 ROS 工作区：未启动、未修改

## 正式命令

```powershell
.\.venv\Scripts\python.exe scripts/run_uuv_tracking_coverage_audit.py `
  --config configs/scenario/uuv_only_single_target.yaml `
  --seed 42 `
  --steps 360 `
  --repeat 2 `
  --work-dir outputs/audit-20260827 `
  --evidence-dir docs/verification/2026-08-27-uuv-tracking-coverage
```

两次运行的轨迹摘要完全一致：

```text
run-a  1f600986181578c2aaa4cbbb0125a0e587c725aaedd0132a67a257c9a34a4790
run-b  1f600986181578c2aaa4cbbb0125a0e587c725aaedd0132a67a257c9a34a4790
```

## 硬检查

| 检查项 | 结果 |
| --- | --- |
| 已分配路线存在 | 通过 |
| 已分配路线几何有效 | 通过 |
| 发出 UUV 命令 | 通过 |
| 观察到受控 UUV 运动 | 通过 |
| 配置的物理不变量 | 通过 |
| 融合跟踪估计可用 | 通过 |
| 指标均为有限数 | 通过 |
| 评估真值存在 | 通过 |
| 两次运行确定性一致 | 通过 |

物理审计覆盖初始基线安装后的 361 个连续采样（frame 0 至 360），17 个预期实体全部存在，无帧缺口、重复帧、瞬移、越界、速度/加速度/转向率违规或路线违规。最小 UUV 两两间距为 118.31 m。

## 跟踪结果

| 指标 | 结果 |
| --- | ---: |
| 时间对齐融合样本数 | 37 |
| RMSE | 488.26 m |
| 中位误差 | 338.16 m |
| P95 | 1,012.15 m |
| 最大误差 | 1,151.21 m |
| 公开观测数 | 54 |

修复前，仿真器会把目标精确坐标写入公开 contact 和全局轨迹历史，预测器与意图节点可以直接消费该历史。这会使控制效果失真。修复后：

- 初始 contact 只公开“潜艇”分类，不公开精确位置；
- 初始搜索走廊来自公开先验中心 `[-6800, -6800]`，与目标初始真值不同；
- 搜索先验在 `valid_until_s` 到期后，prediction、intent、diff 和 gate 缓存按 prior 身份失效；
- 预测和意图只读取融合 belief history；
- 主动声呐使用私有真值判断是否产生回波，但公开的是带噪测量；
- 评估真值只在审计轨迹和可视化中出现，并明确标记为 controller 不可用。

## 覆盖结果

| 任务区 | 最终阶段 | 分配路线数 | 主动发射数 | 采样声呐足迹 | 航点访问 |
| --- | --- | ---: | ---: | ---: | --- |
| Task 01 | `PASSIVE_TRACK` | 2 | 60 | 100% | uuv_04 50%，uuv_05 50% |
| Task 02 | `UNCOVERED` | 0 | 60 | 100% | 无分配路线 |
| Task 03 | `ACTIVE_SCAN` | 2 | 60 | 100% | uuv_01 50%，uuv_02 50% |
| Task 04 | `PLANNED` | 2 | 60 | 52.08% | uuv_00 0%，uuv_03 0% |

蛇形实现的主要源码问题是活动扫描期间每个观察周期都会重新写入整条路线，导致已消费航点被恢复、UUV 难以推进。修复后只在当前路线不是期望路线后缀时重写航点，从而保留已消费进度。正式运行证明活动 UUV 能持续运动且路线几何有效；但调度层仍允许任务区保持 `UNCOVERED` 或 `PLANNED`，所以完整区域覆盖需要额外的部署和时限门槛。

## 可视化

- [跟踪关键帧](tracking-keyframe-final.png)
- [蛇形覆盖关键帧](coverage-keyframe-final.png)
- [跟踪与控制视频](tracking-control-final.mp4)
- [蛇形覆盖搜索视频](coverage-search-final.mp4)

视频以 10 fps、每 3 个仿真帧采样一次，共 121 帧，分辨率为 1200 x 720。编码后重新解码并核对准确帧数、首帧、末帧和尺寸。黑色星标/轨迹是仅供评估的目标真值，控制器不可访问；橙色轨迹是融合估计。

## 验证证据

- 目标真值隔离 scoped 回归：`90 passed, 11 skipped, 2 deselected`
- 11 个跳过项：既有真实 LLM/凭据门禁
- 2 个精确排除项：已在 HEAD 基线出现的 integration 失败
- 完整 non-opt-in 选择命令：`.\.venv\Scripts\python.exe -m pytest -q -m 'not real_llm and not long_running and not live_acceptance'`
- 基线结果（`outputs/audit-20260827/logs/baseline-suite.txt`）：`40 failed, 1586 passed, 46 skipped, 26 deselected`
- 当前结果（`outputs/audit-20260827/logs/final-suite-02.txt`）：`37 failed, 1694 passed, 46 skipped, 26 deselected`
- 完整选择的失败 nodeid 差异：`ADDED 0`，`RESOLVED 3`
  - `tests/integration/test_platform_core_scenario.py::test_explicit_platform_core_tracks_passive_observations_and_calls_carrier`
  - `tests/simulation/test_active_sonar.py::test_uuv_only_active_ping_uses_public_prior_but_requires_physical_echo`
  - `tests/simulation/test_active_sonar.py::test_uuv_only_active_echo_reaches_public_group_report`
- 相关修改文件 Ruff：通过；`cli.py` 和 `engine.py` 的全文件检查仍有与 HEAD 相同的既有告警
- `git diff --check`：通过
- 正式审计：`PASS`
- 视频独立读取：两段均为 121 帧，首尾帧均为 720 x 1200 x 3

原始证据：

- `metrics.json`：正式指标与硬检查
- `trajectory.json`：第一轮 360 步审计轨迹
- `outputs/audit-20260827/run-a`、`run-b`：本地未提交的两次原始运行目录
- `outputs/audit-20260827/logs/audit-run.txt`：正式运行摘要日志

## 限制与后续修改建议

1. 两次运行使用同一 seed，只证明确定性，不构成统计鲁棒性证据。建议至少运行 30 个不同 seed，报告误差、覆盖率、安全间距和任务完成率的置信区间。
2. 当前验收只要求融合估计存在，没有预先确认 RMSE/P95 门槛。建议在科研验收前明确例如 RMSE、P95、最大失联时长和新鲜估计比例的阈值，再据此判定控制质量。
3. 若目标是四个任务区同时完整覆盖，应把 `UNCOVERED` 数量、`PLANNED` 超时、每区最小主动/被动 UUV 数、航点访问率和足迹率纳入硬门槛，并让调度器部署 uuv_00/uuv_03 或重新分配已部署平台。
4. 本次没有运行 ROS、真实 AUV、网络 LLM 或硬件闭环。实机前还需验证通信延迟、声学误检/漏检、定位漂移、能耗和急停边界。
5. 物理审计从确定性基线安装后开始；基线安装时的初始部署位置调整不计入运动审计。若要证明全流程无瞬移，应把部署过程建模为显式运动并纳入审计。
6. 完整 non-opt-in 选择的剩余 37 条失败均继承自基线；与基线相比没有新增失败，并解决了 3 条既有失败。完整测试仍非全绿，不能将本次结果描述为全仓测试全部通过。
