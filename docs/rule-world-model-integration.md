# 规则世界模型展示版：集成与交接说明

## 1. 本次交付做了什么

本分支在现有“纯方位观测 → IMM 跟踪 → B-spline 轨迹预测 → 多 UUV 规划”链路后，增加了一个**只读的未来事件推演模块**。

它不训练神经网络，也不直接给 UUV 下控制命令。它做的事情可以通俗理解为：

1. IMM 告诉它“潜艇现在更像直行、左转还是右转”；
2. B-spline 告诉它“照现在的估计，未来轨迹大概会经过哪里”；
3. 当前 UUV 状态和已提交计划告诉它“未来还有几艘 UUV 能看到目标、测向夹角是否合适”；
4. 规则模块据此给出“未来可能发生什么、预计何时发生、依据是什么”；
5. 前端按 H1～H4 时间段展示事件，并在地图上标出预计发生位置。

这是一版用于联调和演示的“规则世界模型”，后续可以在保持输入、输出接口基本不变的前提下替换为训练后的模型。

## 2. 数据链路

```text
纯方位观测
    ↓
IMM 目标估计（位置、速度、转向模型概率、协方差）
    ↓
B-spline 轨迹预测（未来时间、位置、不确定走廊）
    ├──────────────┐
    ↓              ↓
原有规划链路       规则世界模型（只读，无控制权）
                   ↑
          UUV 状态、已提交航路、跟踪质量、
          可观测性事件、任务区边界
                   ↓
          H1～H4 未来事件 + 证据 + 规则置信度
                   ↓
          OperationalFrame → WebSocket/回放 → React 前端
```

关键安全边界：世界模型和轨迹预测器只读取估计结果，不读取仿真中的潜艇真实位置。输出中固定带有 `control_authority: false`，运行时不会把预测事件变成计划、航路点或动力学控制量。

## 3. 审查后修正的问题

### 3.1 去掉轨迹预测中的真值入口

原预测端口允许传入 `global_trajectory_history`。这会给调用方留下“用潜艇仿真真值覆盖估计历史”的可能，虽然画图评估可以使用真值，在线决策链路不能使用。

本分支删除了这个参数。B-spline 预测只能从 `belief_history`（目标估计历史）生成，测试同时锁定了函数签名，防止以后重新把真值接回去。

### 3.2 改正预测约束使用对象

原接线把 UUV 的最大速度和转弯率传给了潜艇轨迹预测器。两者不是同一类运动对象，会让潜艇的未来轨迹被 UUV 参数错误限制。

本分支改为使用：

- `submarine_sprint_speed_mps`：潜艇冲刺速度上限；
- `submarine_turn_rate_rad_s`：潜艇转弯率上限。

### 3.3 统一同一轮预测数据

运行时先生成一份新的 IMM/B-spline 预测，再把**同一份预测对象**交给世界模型。这样页面上的预测轨迹和未来事件不会来自不同轮次。

## 4. 当前可推演的事件

| 事件 | 通俗含义 | 主要依据 |
| --- | --- | --- |
| `target_turn_left/right` | 潜艇可能向左/右转 | IMM 转向模型概率 + B-spline 航向变化 |
| `high_speed_escape` | 潜艇可能高速脱离 | 当前估计速度 + 预测分段速度 |
| `area_exit_risk` | 轨迹或不确定范围可能碰到任务区边界 | 预测位置 + 不确定走廊 + 地图边界 |
| `geometry_degradation` | UUV 的测向夹角可能变差 | UUV 预计位置 + 方位观测几何指标 |
| `uuv_coverage_gap` | 未来可用 UUV 数量可能不足 | 声呐覆盖、健康、通信和电量状态 |
| `track_loss_risk` | 航迹可能失去可靠观测 | 覆盖、几何、轨迹不确定性和当前跟踪质量的组合 |
| `decoy_or_new_contact_ambiguity` | 可能出现诱饵或新目标，关联开始混乱 | 接触数量、关联变化或已有可观测性反馈 |
| `target_abnormal_stop` | 目标可能持续低速或停止 | B-spline 连续低速预测 |

`target_abnormal_stop` 只表示“二维轨迹显示低速或停止”，不能据此宣布潜艇沉没或坠毁。二维位置数据不包含足够的深度、姿态、设备和生命状态证据。

## 5. H1～H4 时间窗口

| 窗口 | 相对当前时刻 | 用途 |
| --- | ---: | --- |
| H1 | 0～120 秒 | 近期动作和突发风险 |
| H2 | 120～300 秒 | 短期变化 |
| H3 | 300～900 秒 | 中期任务区与跟踪风险 |
| H4 | 900～1800 秒 | 较长期趋势 |

每条事件包含预计时间 `time_to_event_s`、预计位置 `predicted_position_xy`、规则编号 `rule_id`、依据 `evidence` 和 `confidence`。

这里的 `confidence` 是“规则证据强度”，用于排序和展示，不是经过真实样本校准的概率。前端明确显示为“规则置信度”。

## 6. 代码结构

### 6.1 规则核心

- `src/underwater_tracking/world_model/models.py`：严格、只读的输入输出数据结构；禁止多余字段，降低误接真值的风险。
- `src/underwater_tracking/world_model/rules.py`：八类事件规则、H1～H4 划分、未来 UUV 投影和测向几何计算。
- `src/underwater_tracking/world_model/adapter.py`：把当前项目的快照、IMM/B-spline 结果、已提交计划和可观测性事件转成规则输入。
- `src/underwater_tracking/world_model/config.py`：读取并校验规则配置。
- `src/underwater_tracking/world_model/demo.py`：九个固定场景的命令行演示。
- `configs/world_model_rules.yaml`：开关、时间窗口和阈值。

### 6.2 运行时和接口

- `src/underwater_tracking/agent/runtime.py`：每次刷新轨迹预测后生成未来事件。
- `src/underwater_tracking/agent/state.py`：保存当前轮 `world_model_forecasts`。
- `src/underwater_tracking/api/live.py`：把运行时结果送进实时帧。
- `src/underwater_tracking/api/frame_builder.py`：转成对外展示结构，并按地图范围裁剪事件位置。
- `src/underwater_tracking/domain/ui_models.py`：向前端公开的严格数据契约。

### 6.3 前端

- `src/underwater_tracking/ui/src/components/WorldModelPanel.tsx`：未来事件侧栏。
- `src/underwater_tracking/ui/src/components/map/WorldModelEventOverlay.tsx`：地图事件标记。
- `src/underwater_tracking/ui/src/components/CanvasMap.tsx`：把事件点纳入地图范围并叠加显示。

旧的操作帧没有 `world_model` 字段时仍可正常加载，回放文件不需要迁移。

## 7. 配置和开关

默认场景配置会自动加载 `configs/world_model_rules.yaml`。若要临时关闭，只需修改项目内配置：

```yaml
enabled: false
```

不需要修改系统 Python、ROS、全局环境变量或共享服务器配置。

主要阈值都在 `thresholds` 下，例如：

- `turn_model_probability_min`：IMM 转向模型至少要有多大支持；
- `sprint_speed_threshold_mps`：多快算高速脱离；
- `min_tracking_uuvs`：至少几艘有效 UUV 才不算覆盖缺口；
- `geometry_warning_od`：测向几何开始变差的门限；
- `track_loss_corridor_m`：预测不确定走廊多宽时进入丢失风险判断；
- `event_min_confidence`：事件最低展示强度。

## 8. 运行方式

项目要求 Python 3.11 或 3.12。

### 8.1 独立规则演示

```bash
python -m underwater_tracking.world_model.demo --scenario all --pretty
```

也可以只看一个场景：

```bash
python -m underwater_tracking.world_model.demo --scenario left_turn --pretty
```

支持的场景为 `normal`、`left_turn`、`sprint`、`area_exit`、`decoy`、`geometry_bad`、`coverage_gap`、`track_loss` 和 `stop`。

### 8.2 完整系统

```bash
python main.py --config configs/scenario/default.yaml --seed 42
```

打开控制台页面后，在右侧“未来事件推演”查看 H1～H4 事件；地图上菱形标记表示事件预计发生位置。关闭页面原有的预测叠加开关时，事件位置标记也会一起隐藏。

## 9. 验收方法

### 9.1 后端关键链路

```bash
python -m pytest \
  tests/world_model \
  tests/prediction/test_port.py \
  tests/agent/test_live_prediction_intent_runtime.py \
  tests/agent/test_runtime.py \
  tests/api/test_frame_contracts.py \
  tests/api/test_frame_pipeline.py \
  tests/api/test_live_publisher.py::test_publisher_projects_world_model_forecast_to_replay \
  tests/config/test_loader.py \
  tests/config/test_models.py \
  tests/cli/test_cli.py::test_agent_dependencies_keep_prediction_truth_safe_and_use_target_limits
```

验收重点：

- 九个固定场景输出符合预期；
- 相同输入输出逐字节一致；
- 输入结构拒绝仿真真值字段；
- 轨迹预测端口不再接受真值历史；
- 运行时使用本轮新预测和已提交计划；
- 未来事件可经过实时帧写入 JSONL，并完整回放；
- 旧帧缺少世界模型字段仍可读取。

### 9.2 前端

```bash
npm --prefix src/underwater_tracking/ui test
npm --prefix src/underwater_tracking/ui run build
npm --prefix src/underwater_tracking/ui run test:e2e
```

如果机器没有下载 Playwright Chromium，可以把已有 Chromium/Chrome/Edge 路径临时传给 `PLAYWRIGHT_EXECUTABLE_PATH`，不必改全局浏览器环境。

本分支开发验收结果：后端世界模型及相关关键链路 93 项测试全部通过；前端 26 个测试文件、121 项测试全部通过，生产构建通过；包含未来事件侧栏、规则说明和地图事件点的 3 个桌面/移动端浏览器联调用例通过。

在最终增加实时帧回放测试之前，曾用 Windows/Python 3.13 临时兼容环境做过一次仓库全量基线检查，结果为 1572 通过、70 跳过、46 失败；失败集中在远端基线已有的 POSIX 信号处理、Windows 路径/编码、旧测试夹具缺字段和部分 UUV 物理场景预期。它们不在本次功能改动范围内，且本次新增和受影响的关键测试均通过。正式运行应按仓库声明使用 Python 3.11 或 3.12，并建议在目标 Linux 环境再跑一次全量测试。

## 10. 已知边界与后续路线

当前版本适合功能展示和接口联调，但仍有以下边界：

1. 规则置信度没有经过真实数据校准，不能当成严格事件概率；
2. 诱饵判断只能表达“诱饵或新目标关联混乱”，不能仅凭接触增加就确认诱饵；
3. 只有二维轨迹时不能可靠判断沉没、上浮、深度突变等三维事件；
4. UUV 未来位置优先使用已提交航路点，没有航路时才做匀速外推，并把数据状态标为 `degraded`；
5. 当前输出只用于展示和反馈，不参与自动改计划。

建议后续按下面顺序推进：

1. 补齐接触关联历史、深度、声学环境和设备健康等可观测输入；
2. 用仿真批量生成带事件标签的数据，校准规则置信度和误报率；
3. 将规则结果作为基线，与 Sea-Air DreamerV3 Lab 的学习型世界模型做同接口对比；
4. 只有在独立验证通过后，才设计“事件建议 → 人工确认 → 规划调整”的安全闭环。
