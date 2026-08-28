# 真实运行可视化与算法执行稳定性设计

日期：2026-08-28

状态：设计已确认，待实施计划

范围：`main.py` 默认 UUV-only 单目标场景中的目标运动、全局轨迹、IMM 预测、四区域滚动规划、权威执行快照、操作帧、HTTP/WebSocket/JSONL/Replay 一致性、地图可视化和真实运行验收。

## 1. 背景与问题陈述

当前仓库同时存在两种效果证据：

1. `outputs/imm-confidence-trajectory-effect.png` 等图片由 Playwright 效果测试中的手写帧生成。测试替换 WebSocket，并拦截 `/api/operational/snapshot` 返回固定的目标、轨迹、半径、区域和平台位置。
2. `python main.py` 使用真实仿真、真实规划、真实执行快照和真实操作帧。长时间运行后，目标运动、预测协方差、任务区域和执行版本可能进入失效状态。

因此，合成效果图可以证明前端组件能够渲染理想输入，但不能证明真实算法链能够持续产生同等质量的输入。

2026-08-28 的真实运行审查观察到：

- 目标估计到达地图左下边界 `(-12000, -12000)`；
- 60 个预测点的横坐标全部退化为 `-12000`；
- 预测走廊半径一度超过 `60 km`，远大于 `24 km x 24 km` 地图；
- 目标运动在 `1515 s` 触发 `target_navigation_guard_failed`；
- 执行快照停留在 revision 3，数据年龄超过 `6000 s`；
- 四区域状态为一个 `planned` 和三个 `uncovered`；
- 四个任务组为一个 `prepositioning` 和三个 `degraded`；
- 规划数据库持续出现 `task region cannot be aligned around prediction centerline` 和 `map bounds cannot retain minimum dynamic region area`；
- 前端为了容纳异常置信走廊而扩大相机范围，使探测圈、区域、UUV 和目标压缩成一团。

这是一条端到端失效链，而不是四个独立的 CSS 问题：

```text
目标边界运动失败
  -> 轨迹停留或贴边
  -> 有效观测减少、协方差膨胀
  -> IMM 中心线和走廊退化
  -> 四区域几何生成失败
  -> 执行 revision 不再滚动
  -> UI 消费陈旧、越界或尺度异常的数据
  -> 相机和所有主要图层一起失去可读性
```

## 2. 目标与非目标

### 2.1 目标

1. `main.py` 在完整 `28800 s` 仿真生命周期内持续发布可解释、可执行、可渲染的状态。
2. 目标不能因导航保护失败永久锁死在地图边界。
3. 正常路径使用完整 IMM 预测；异常路径提供有界、标明来源的降级预测。
4. 每个目标始终拥有恰好四个当前任务区域和四个两艇任务组，执行快照按固定周期滚动。
5. 探测范围、任务区域、预测中心线和不确定性阴影引用同一目标估计与同一 prediction/execution revision。
6. 前端忠实呈现算法健康状态，同时阻止异常数据破坏地图取景和标签可读性。
7. 发布验收必须使用真实 `main.py`、真实传输通道、真实缓存和真实浏览器截图。

### 2.2 非目标

1. 不要求真实运行逐像素复刻 `imm-confidence-trajectory-effect.png`。
2. 不固定真实运行中的目标、区域或 UUV 坐标。
3. 不通过缩短场景、冻结目标、关闭 LLM 或在前端伪造数据来获得理想截图。
4. 不重建全部事件溯源架构。
5. 不改变非 UUV-only 场景的历史回放语义，除非共享校验逻辑需要兼容性调整。
6. 不让 LLM 直接输出任意区域多边形或物理航点。

## 3. 方案比较与决策

### 3.1 选定方案：算法有界化、权威快照和健康感知呈现

从目标运动、预测、区域、执行快照、操作帧到相机完成端到端修复。异常状态在源头被识别，并通过结构化健康字段进入操作帧和 UI。

优点：解决根因；算法、执行与视觉使用同一事实；支持长时间真实验收。

代价：涉及仿真、预测、规划、运行时、API、前端和验收工具多个边界，必须分阶段实施。

### 3.2 未选方案：只调整场景与运行时长

通过修改初始位置、目标路线、机动幅度或验收时刻获得好看的画面。

拒绝原因：只能推迟边界失效，不能保证完整 `28800 s` 生命周期，也不能解决陈旧执行快照。

### 3.3 未选方案：只在前端裁剪和缩放

限制显示半径、过滤异常点并强制聚焦目标。

拒绝原因：会把算法失败伪装成正常预测，任务控制仍然消费无效或陈旧几何。

## 4. 系统不变量

1. 操作帧不得包含非有限预测坐标、非有限协方差或负半径。
2. 业务预测中心线的每个点必须位于 `map_bounds` 内。
3. 异常原始预测只进入审计证据，不能直接进入执行快照或地图业务图层。
4. 每个 UUV-only 目标的执行快照必须包含恰好四个有序区域。
5. 四个区域、四个任务组和执行快照必须引用同一 `execution_revision` 与 `prediction_id`。
6. 同一 UUV 不能同时属于两个任务组，也不能同时是执行成员和替补资源。
7. `data_status` 必须由实际数据年龄计算，不能由调用方任意填写。
8. 超过硬过期阈值的快照不得继续驱动物理任务或显示为正常运行。
9. HTTP、WebSocket、JSONL 和 Replay 必须发布同一个不可变操作帧对象。
10. 相机可见范围不得超出地图边界。
11. 前端不得用显示裁剪替代后端健康校验。
12. 发布验收不得替换 WebSocket、拦截业务 API 或注入手写业务帧。

## 5. 总体架构

```text
Bounded Target Motion
        |
        v
GlobalTargetTrack
        |
        v
IMM Forecast -> Prediction Health Validator -> AcceptedPrediction
                         |                         |
                         | rejected                | valid/degraded
                         v                         v
                    Audit Evidence      Deterministic Four-Region Baseline
                                                   |
                                                   v
                                      Optional LLM Semantic Optimization
                                                   |
                                                   v
                                    OperationalExecutionSnapshot
                                                   |
                                                   v
                                        OperationalFrame Validator
                                                   |
                             +---------------------+---------------------+
                             |                     |                     |
                            HTTP               WebSocket          JSONL / Replay
                             |                     |                     |
                             +---------------------+---------------------+
                                                   |
                                                   v
                                      Health-Aware Map Rendering
```

现有 `OperationalExecutionSnapshot` 继续作为 UUV-only 物理执行和 UI 的唯一权威执行状态。设计不新增平行计划状态。

## 6. 目标导航与边界恢复

### 6.1 当前失败机制

目标导航层会把请求航向调整为向内航向，但实际积分仍受最大转向率和减速度约束。请求航向合法不等于下一物理子步必然合法。积分失败后当前逻辑进入安全保持，但没有完整、可观测、可终止的恢复状态机。

### 6.2 恢复状态机

```text
NORMAL
  -> BOUNDARY_DECELERATING
  -> BOUNDARY_TURNING
  -> BOUNDARY_RECOVERING
  -> NORMAL
```

状态语义：

- `NORMAL`：执行任务或 LLM 决策航路。
- `BOUNDARY_DECELERATING`：当前速度和转弯能力无法保证下一段合法，优先减速至可安全转弯速度。
- `BOUNDARY_TURNING`：以受限转向率向内部安全航向转弯。
- `BOUNDARY_RECOVERING`：驶向地图内部恢复点，直到重新获得完整机动余量。

### 6.3 运动约束

边界预判距离为：

```text
guard_distance = stopping_distance + turn_radius + 50 m
```

其中：

```text
stopping_distance = speed^2 / (2 * max_deceleration)
turn_radius = speed / max_turn_rate
```

每个 `0.5 s` 子步必须先计算受加速度和最大转向率限制后的真实候选状态，再用候选线段验证地图边界和禁航区。不能只验证期望航向。

无法安全转向时允许减速至 `0 m/s`。恢复点位于当前位置指向地图内部安全中心的合法线段上，并保持至少一个 `guard_distance` 的边界余量。

连续两个物理步合法，且当前位置到所有地图边界的最小距离大于当前 `guard_distance` 后，状态恢复为 `NORMAL`。

边界恢复持续超过 `300 s` 时，运行进入 `target_navigation_recovery_failed` 终止错误。终止前发布最后一个明确的失败操作帧和审计事件。

### 6.4 可观测字段与事件

目标公开状态增加：

- `navigation_state`
- `navigation_state_since_s`
- `navigation_recovery_waypoint_xy`
- `navigation_guard_distance_m`
- `last_navigation_error`

状态变化产生单次事件：

- `target_boundary_recovery_started`
- `target_boundary_turn_started`
- `target_boundary_recovery_completed`
- `target_navigation_recovery_failed`

## 7. 预测健康与降级链

### 7.1 新模型

新增不可变 `PredictionHealth`：

```text
status: valid | degraded | unavailable
regime: imm | bspline | short_history | boundary_recovery
reason_codes: tuple[str, ...]
source_track_age_s: float
clipped_point_fraction: float
maximum_radius_m: float
raw_prediction_id: str | null
```

`AcceptedPrediction` 包含通过健康校验的 `PredictedTrackRef` 和 `PredictionHealth`。后续区域规划、执行快照和操作帧只消费 `AcceptedPrediction`。

### 7.2 配置

在 `configs/tracking.yaml` 增加：

```yaml
prediction_health:
  refresh_interval_s: 450
  hard_stale_s: 900
  max_clipped_point_fraction: 0.20
  max_corridor_radius_m: 6000.0
  max_corridor_map_fraction: 0.25
  minimum_point_confidence: 0.02
  coordinate_tolerance_m: 0.000001
  boundary_recovery_timeout_s: 300
```

有效最大走廊半径为：

```text
min(max_corridor_radius_m, min(map_width, map_height) * max_corridor_map_fraction)
```

默认地图上限因此为 `6000 m`。

### 7.3 健康校验

预测成为业务预测前必须满足：

1. 时间、坐标、协方差、半径和概率均为有限值。
2. 时间严格递增，且点数、半径数、协方差数和置信度数一致。
3. 所有中心线点位于地图内，允许 `1e-6 m` 数值误差。
4. 裁剪点比例不超过 `0.20`。
5. 最大半径不超过有效半径上限。
6. 每步距离不超过 `submarine_max_speed_mps * sample_step_s`。
7. 每步航向变化不超过 `submarine_max_turn_rate_rad_s * sample_step_s`。
8. `point_confidence` 位于 `[0, 1]`，并随预测时间单调不增。
9. 最后一点置信度不低于 `0.02`。
10. 来源轨迹不晚于当前仿真时间，且年龄未超过预测硬过期阈值。

违反硬约束的原始预测记录：

- 原始 prediction ID；
- 失败约束；
- 最大半径；
- 越界点索引；
- 来源轨迹和观测 ID；
- 失败时仿真时间。

### 7.4 降级顺序

```text
IMM
 -> bounded B-spline
 -> bounded short-history extrapolation
 -> boundary-recovery prediction
 -> unavailable
```

每个降级预测必须通过相同健康校验。降级来源不会被隐藏：`status=degraded`，`regime` 和 `reason_codes` 必须进入操作帧。

`boundary_recovery` 预测使用当前全局轨迹和目标导航恢复航点，生成受速度、转向率和地图边界约束的中心线。它不是正常战术预测，只为恢复期间的区域和视觉连续性提供可辩护几何。

如果所有候选均失败，则发布 `unavailable`。此状态不包含虚构中心线，区域层进入上一有效链重投影或终止流程。

### 7.5 置信度

后端根据每个点的健康校验后半径生成 `point_confidence`。前端不重新计算概率。置信度只控制视觉强调，不改变走廊真实几何宽度。

## 8. 四区域确定性基线

### 8.1 责任边界

确定性算法始终先产生可执行的四区域基线。LLM 只能调整：

- 区域优先级；
- 四个时间窗的受限比例；
- 主动/被动声呐策略；
- 任务组角色建议；
- 替补优先级；
- 解释与证据。

LLM 不再输出任意多边形坐标、区域 ID、UUV 物理航点或未注册的 UUV ID。

### 8.2 生成顺序

区域生成按以下顺序执行：

1. 沿 `valid` IMM 中心线生成。
2. 沿通过健康校验的 `degraded` 预测生成。
3. 沿 `boundary_recovery` 预测生成。
4. 将上一有效四区域链按新的预测锚点和时间原点滚动重投影。

任一路径都必须输出恰好四个区域。无法输出合法四区域链时，不得继续发布过期正常状态。

### 8.3 几何不变量

每个区域链必须满足：

1. 区域 ID 稳定为 `target_id:task:01` 至 `target_id:task:04`。
2. 区域包含对应时间段的预测中心线样本。
3. 区域全部位于地图内。
4. 相邻区域存在受控交接重叠。
5. 非相邻区域不重叠。
6. 区域面积和最短边满足当前最小任务几何阈值。
7. 前驱、后继和交接窗口完整。
8. 四个区域引用同一 prediction ID。
9. `geometry_revision` 仅在几何变化时递增。
10. 边界恢复区域明确标记 `generation_mode=boundary_recovery`。

区域生成器必须在构造前把中心线锚点投影到合法内边界，而不是先生成超大多边形再依赖裁剪修复。裁剪只处理数值余量，不承担主要几何生成责任。

### 8.4 时间窗

区域窗口使用绝对仿真时间：

```text
absolute_time = prediction.origin_sim_time_s + window_offset_s
```

默认窗口保持：

| 区域 | 相对时间窗 |
|---|---:|
| R01 | 0-540 s |
| R02 | 450-990 s |
| R03 | 900-1440 s |
| R04 | 1350-1800 s |

相邻窗口重叠 `90 s`。滚动更新后全部窗口使用新 prediction origin，不能保留旧绝对时间。

## 9. 执行快照与两阶段提交

### 9.1 新鲜度

确定性滚动检查间隔为 `450 s`：

| 数据年龄 | 状态 | 行为 |
|---|---|---|
| `0-450 s` | `current` | 正常执行 |
| `>450-900 s` | `degraded` | 保留执行并高优先级刷新 |
| `>900 s` | `expired` | 禁止继续驱动物理任务 |

`data_status` 由当前仿真时间减去 `source_sim_time_s` 计算。调用方不能覆盖结果。

### 9.2 两阶段提交

每个滚动周期：

1. 捕获不可变物理快照和当前 execution revision。
2. 生成并校验 `AcceptedPrediction`。
3. 生成确定性四区域、任务组、替补资源和航点。
4. 使用 compare-and-set 提交确定性基线 revision。
5. 异步运行 LLM 语义优化。
6. 校验 LLM 输出、证据和基线 revision。
7. 合法时用 compare-and-set 提交优化 revision；过期或非法时只记录审计。

LLM 超时、内容错误、非法策略或几何错误不能阻止步骤 4。

### 9.3 过期处理

确定性刷新失败时保留最后有效快照，但不得延长其 `valid_until_s`。进入 `degraded` 后每个观察周期重试确定性刷新。

达到 `expired` 时：

1. MissionController 停止接受该快照的新航点。
2. UUV 执行受控保持或安全分离，而不是继续追逐过期区域。
3. 发布 `execution_failed` 操作状态和失败原因。
4. 服务保持可查询，以便 UI、缓存和审计读取最后状态。

## 10. 操作帧与传输一致性

### 10.1 帧模型扩展

`PredictionCorridorView` 增加：

```text
prediction_id: str
origin_sim_time_s: float
health: PredictionHealthView
```

保留：

```text
horizon_s
sample_step_s
centerline_xy
radius_m
point_confidence
diff
```

`ExecutionView` 增加：

```text
valid_from_s: int
valid_until_s: int
health_status: current | degraded | expired | failed
health_reasons: tuple[str, ...]
region_generation_mode: imm | degraded_prediction | boundary_recovery | reprojected_previous
```

### 10.2 发布校验

发布前验证：

- frame ID 单调递增；
- sim time 单调不减；
- prediction ID 在目标、区域和执行对象间一致；
- execution revision 在区域和任务组间一致；
- 四区域几何合法且在地图内；
- 时间窗未过期或状态明确为 expired/failed；
- UUV 分配互斥；
- 所有 evidence ID 可解析；
- `data_status` 与数据年龄一致；
- `valid`/`degraded` 预测通过健康校验；
- `unavailable` 预测不携带虚构走廊。

失败时不发布半成品帧。系统发布上一完整帧和单独的当前运行健康状态；达到硬过期条件时发布终止失败帧。

### 10.3 通道一致性

同一 `OperationalFrame` 对象经序列化后写入：

- `/api/operational/snapshot`；
- WebSocket；
- `operational_frames.jsonl`；
- Replay 索引。

验收通过 frame ID、execution revision、prediction ID 和规范化 JSON hash 检查通道一致性。

## 11. 地图可视化

### 11.1 图层顺序

从底到顶：

1. 海图背景和公里网格；
2. 四个任务区域与交接箭头；
3. 预测置信走廊；
4. 预测中心线和置信采样点；
5. 目标探测范围；
6. UUV 主动/被动声呐扇区；
7. 目标、UUV、任务组和状态标签；
8. 选中态和异常态。

### 11.2 探测范围

- 目标探测范围使用红色虚线圆和低透明度填充。
- 主动 UUV 使用琥珀色声呐扇区。
- 被动 UUV 使用青色声呐扇区。
- 扇区方向优先使用 `sensor_heading_rad`，缺失时回退 `heading_rad`。
- 目标探测圆和 UUV 声呐扇区默认显示，并允许分别切换。
- 图形裁剪到视口，但标签显示真实米制范围。

### 11.3 任务区域

只显示权威执行快照中的四个区域：

| 状态 | 视觉 |
|---|---|
| `active` | 亮青色实线和轻填充 |
| `handoff_ready` | 琥珀色实线 |
| `planned` / `prepositioning` | 紫灰色实线 |
| `degraded` | 虚线和斜纹 |
| `uncovered` | 红色虚线 |

当前区域和下一交接区域优先显示完整标签。其他区域使用简洁标签。标签至少提供区域号、任务组、状态和时间窗。

### 11.4 预测轨迹和不确定性

| 健康状态 | 视觉 |
|---|---|
| `valid` | 青色半透明走廊、黄色虚线中心线、置信采样点 |
| `degraded` | 低透明度、虚线边界、斜纹走廊、降级标识 |
| `unavailable` | 不绘制走廊，只显示目标和不可用状态 |

前端使用后端已校验的 `radius_m`。`point_confidence` 只影响采样点大小和视觉透明度，不改变几何宽度。

### 11.5 语义相机

默认相机候选集合：

- 当前目标；
- 有效预测走廊；
- 四个执行区域；
- 八个执行 UUV；
- 目标探测圆。

候选边界与 `map_bounds` 求交后增加 `8%` 内边距。任何预测半径或范围图层都不能把相机扩展到地图外。

可读性约束：

- 目标标记直径 `24-32 px`；
- UUV 标记直径 `22-30 px`；
- 任务区域投影最短边至少 `48 px`；
- 中心线物理长度超过 `2 km` 时，屏幕长度至少 `120 px`；
- `5 km` 目标探测圈屏幕直径至少 `160 px`；
- 字号不随 viewport 宽度缩放；
- 标签冲突时按目标、当前区域、下一区域、当前任务组、其他区域、其他 UUV 的优先级隐藏。

保留 `prediction_corridor` 和 `full_area` 两种模式及复位按钮。自动重取景只在 prediction revision 变化或用户主动复位时发生；用户平移缩放期间不抢夺视图。

## 12. 错误处理与运行健康

### 12.1 可恢复错误

- 单个 IMM 预测失败：进入预测降级链。
- LLM 超时或非法内容：保留已提交确定性基线。
- 一次区域几何失败：尝试边界恢复或上一有效链重投影。
- WebSocket 丢帧：HTTP snapshot 补偿后继续。
- 前端收到旧 frame ID：丢弃旧帧并记录计数。

### 12.2 终止错误

- 导航恢复超过 `300 s`；
- 所有预测候选均不可用且无法合法重投影区域；
- 执行快照年龄超过 `900 s`；
- 操作帧违反 revision 或几何不变量；
- HTTP、WebSocket、JSONL 产生不同规范化帧。

终止错误停止物理任务推进，但保留 API 和静态 UI，以便读取最终失败帧、缓存和审计证据。

### 12.3 UI 失败呈现

UI 必须区分：

- 规划正在运行；
- 使用降级预测；
- 保留上一有效区域链；
- 执行快照已过期；
- 运行已终止。

异常消息显示具体 reason code 和 revision，不使用笼统的“运行中”覆盖失败状态。

## 13. 观测与审计

每个运行目录增加 `acceptance/`，包含：

```text
acceptance/
  manifest.json
  metrics.json
  frame-checkpoints.jsonl
  screenshots/
  browser-console.jsonl
  backend-errors.jsonl
```

核心指标：

- `target_boundary_recovery_count`
- `target_boundary_recovery_max_duration_s`
- `prediction_valid_fraction`
- `prediction_degraded_fraction`
- `prediction_unavailable_fraction`
- `prediction_max_radius_m`
- `prediction_max_clipped_fraction`
- `execution_max_data_age_s`
- `deterministic_baseline_commit_count`
- `llm_optimization_commit_count`
- `region_generation_failure_count`
- `expired_execution_frame_count`
- `frame_channel_mismatch_count`
- `browser_console_error_count`
- `required_layer_missing_count`

运行 manifest 记录代码 revision、配置 hash、场景、seed、provider 身份、前端 bundle hash、开始/结束时间和终止原因。

## 14. 测试与验收

### 14.1 单元测试

覆盖：

- 受限转向下的真实边界预判；
- 减速、转向、恢复和超时状态机；
- 预测健康的每个硬约束；
- IMM、B-spline、短历史和边界恢复降级顺序；
- 边界附近的四区域生成；
- 上一有效区域链滚动重投影；
- 执行快照年龄状态；
- 操作帧 revision/geometry 校验；
- 相机边界裁剪和最小可读尺度；
- 三种预测健康状态的视觉图层。

### 14.2 集成测试

覆盖：

- 目标驶近四条地图边界和四个角；
- 观测丢失后协方差增长与降级预测；
- LLM task-region 内容错误时确定性基线继续提交；
- 区域滚动跨越多个 `450 s` 周期；
- MissionController 不消费 expired 快照；
- HTTP/WebSocket/JSONL/Replay 帧一致；
- API 在物理线程终止后仍可读取最终失败帧。

### 14.3 真实运行检查点

真实 `python main.py` 固定场景和 seed，在以下仿真时刻检查：

```text
600, 1800, 3600, 7200, 14400, 21600, 28800 s
```

每个检查点必须满足：

1. 恰好一个目标、四个区域和四个任务组。
2. 八个执行 UUV，任务组成员互斥。
3. 执行快照年龄不超过 `900 s`。
4. 预测点有限、有界，半径不超过配置上限。
5. HTTP、WebSocket、JSONL 的 frame/revision/prediction ID 一致。
6. 区域全部在地图内，时间窗未过期。
7. 页面包含预测、四区域、目标探测范围和 UUV 声呐范围图层。
8. Canvas 非空，主要图层具有足够像素面积。
9. 桌面 `1600 x 1000` 和移动端 `390 x 844` 无重叠、裁切和溢出。

### 14.4 视觉验收

旧效果图仅用于视觉语言对照。发布门禁使用 DOM、Canvas 元数据、像素统计和截图人工审查组合：

- `.imm-confidence-band` 在 `valid/degraded` 状态下恰好一个；
- `.region-map-overlay polygon` 恰好四个；
- 目标探测圆和所需 UUV 声呐图层可见；
- 预测中心线超过 `2 km` 时屏幕长度至少 `120 px`；
- 目标探测圈屏幕直径至少 `160 px`；
- 主要彩色图层像素占比非零且不过度覆盖地图；
- 浏览器控制台无未处理错误；
- 状态标签与后端健康字段一致。

合成效果测试保留为组件级测试，但文件名、测试名和输出 manifest 必须明确包含 `synthetic-fixture`。它不计入真实运行发布门禁。

## 15. 兼容与迁移

1. 新操作帧字段使用可选读取兼容旧 JSONL；新实时帧必须完整提供字段。
2. Replay 对缺少 `health` 的旧预测标记为 `legacy_unknown`，不推断其为 valid。
3. 非 UUV-only 场景继续使用现有计划路径，但共享有限值、边界和相机裁剪校验。
4. 旧合成截图不删除，移动到明确的测试证据语义或在 manifest 中标记来源。
5. 前端 bundle hash 写入运行 manifest，防止源码与 `dist` 不一致被误判为算法问题。

## 16. 实施顺序与发布门禁

实施必须按以下依赖顺序推进：

1. 目标导航恢复和边界测试。
2. 预测健康模型、校验器和降级链。
3. 确定性四区域基线和边界重投影。
4. 执行快照新鲜度与两阶段提交。
5. 操作帧契约和多通道一致性。
6. 健康感知地图图层和语义相机。
7. 真实运行验收工具和完整 `28800 s` 发布门禁。

每一阶段必须先通过自己的单元和集成测试，再进入下游阶段。最终发布要求：

- 所有现有非真实-provider 测试通过；
- 新增边界、预测、区域、快照、帧和 UI 测试通过；
- 真实-provider 启动校验通过；
- 完整真实运行检查点全部通过；
- 验收目录材料完整；
- 不存在 expired 执行帧、通道不一致或缺失主要地图图层。

## 17. 成功定义

修复完成不是指某一张截图与旧参考图相似，而是同时满足：

1. 目标运动在完整运行中保持物理合法，并能从边界风险中恢复。
2. IMM 或明确降级预测持续提供有界、可解释的数据。
3. 四区域和任务组按时滚动，执行 revision 不停滞。
4. 探测范围、任务区域、预测轨迹和不确定性阴影在真实页面中持续可读。
5. UI 对 valid、degraded、unavailable、expired 和 failed 状态诚实呈现。
6. 真实 HTTP、WebSocket、JSONL、Replay、数据库和截图证据相互一致。

