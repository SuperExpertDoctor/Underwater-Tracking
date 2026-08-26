# 全局轨迹驱动的 IMM 四区域 UUV 执行闭环实施计划

状态：待用户审阅，尚未授权实施

设计依据：docs/superpowers/specs/2026-08-26-global-track-imm-region-execution-design.md

目标：建立单一权威执行快照，使全局目标轨迹、完整 IMM 预测、确定性意图、四个动态任务区域、四个两艇 task group、边界替换、LLM 增量优化、实时数据链、地图、助手记忆和运行生命周期形成可审计闭环。

## 全局约束

- 当前阶段只审阅计划，不修改代码。
- 实施时每个行为变更必须先增加失败测试。
- UUV-only 正常运行不得等待 LLM 才开始运动。
- 目标当前位置和历史轨迹全局可知；目标隐藏意图和未执行决策不可知。
- 地图始终只显示四个执行任务区域。
- 四个区域各配置一个两艇 task group，共八艘执行 UUV。
- 其余四艘 UUV 为非空间替补，只有替换时从区域边界进入。
- 不恢复航母、母舰、投放、回收或返航机制。
- TrackingPlan 只保留审计和旧回放兼容用途。
- HTTP snapshot、WebSocket 和 Replay 必须投影同一个完整帧。
- 一次正式运行只创建一个 run-*，不创建 serve-*。
- 实施期间保留工作区现有未提交修改，不得盲目回退或覆盖。
- 每个阶段单独提交；提交前运行对应定向测试。

## 预期新增模块

- src/underwater_tracking/domain/execution_models.py
- src/underwater_tracking/tracking/global_track.py
- src/underwater_tracking/prediction/imm_forecast.py
- src/underwater_tracking/intent/deterministic.py
- src/underwater_tracking/planning/dynamic_regions.py
- src/underwater_tracking/planning/task_groups.py
- src/underwater_tracking/runtime/execution_coordinator.py
- src/underwater_tracking/runtime/process_supervisor.py
- tests/domain/test_execution_models.py
- tests/tracking/test_global_track.py
- tests/prediction/test_imm_forecast.py
- tests/intent/test_deterministic_intent.py
- tests/planning/test_dynamic_regions.py
- tests/planning/test_task_groups.py
- tests/runtime/test_execution_coordinator.py
- tests/runtime/test_process_supervisor.py

## Task 0：确认基线和隔离现有原型修改

文件：

- 检查当前 git status 中所有已修改源文件和测试。
- 不修改生产文件。
- 生成只读差异清单供后续任务逐项吸收。

步骤：

- [ ] 记录 git status、git diff --stat 和当前 HEAD。
- [ ] 将现有修改按“区域显示、候选几何、全局轨迹、IMM、航点缓存、动态滚动、Playwright”分类。
- [ ] 标记哪些修改已经有先红后绿的测试证据，哪些仍是未验证原型。
- [ ] 运行当前定向测试，记录基线失败，不在本任务修复。
- [ ] 确认现有 run/serve 进程均不参与测试基线。

基线命令：

    python -m pytest tests/prediction/test_port.py tests/api/test_live_publisher.py tests/cli/test_cli.py tests/simulation/test_mission_waypoint_geometry.py -q
    npm --prefix src/underwater_tracking/ui test -- --run
    npm --prefix src/underwater_tracking/ui run build
    git diff --check

输出：

- 一份任务内基线记录。
- 不产生代码提交。

## Task 1：建立权威执行领域契约

文件：

- 新建 src/underwater_tracking/domain/execution_models.py
- 修改 src/underwater_tracking/domain/mission_models.py
- 修改 src/underwater_tracking/domain/ui_models.py
- 新建 tests/domain/test_execution_models.py
- 修改 tests/domain/test_mission_models.py

接口：

- GlobalTargetTrackView
- IMMModelForecast
- IMMPredictedTrack
- DeterministicIntentState
- ExecutionRegion
- TaskGroupAssignment
- ReserveUUVState
- ExecutionDegradation
- OperationalExecutionSnapshot

步骤：

- [ ] 先写严格模型测试，覆盖四区域数量、四 task group、每组两艇、UUV 分配互斥、稳定区域 ID、完整拓扑和证据 ID。
- [ ] 写 execution_revision、prediction_revision、intent_revision 和 source_snapshot_revision 的校验测试。
- [ ] 写非法第五区域、重复 UUV、未知 current_region_id、跨版本区域和空证据测试。
- [ ] 运行 tests/domain/test_execution_models.py，确认因模型不存在而失败。
- [ ] 实现冻结的 Pydantic 严格模型和验证器。
- [ ] 为旧回放字段提供只读兼容默认值，不让旧类型进入新执行模型。
- [ ] 运行领域测试和模型 JSON round-trip。

验证命令：

    python -m pytest tests/domain/test_execution_models.py tests/domain/test_mission_models.py -q

提交：

    feat: add authoritative UUV execution contracts

## Task 2：建立全局目标轨迹服务

文件：

- 新建 src/underwater_tracking/tracking/global_track.py
- 修改 src/underwater_tracking/simulation/engine.py
- 修改 src/underwater_tracking/config/models.py
- 修改 configs/scenario/uuv_only_single_target.yaml
- 新建 tests/tracking/test_global_track.py
- 修改 tests/simulation/test_engine.py

接口：

- GlobalTrackStore.observe
- GlobalTrackStore.snapshot
- GlobalTrackStore.history
- GlobalTrackStore.checkpoint
- GlobalTrackStore.restore

步骤：

- [ ] 写固定种子轨迹测试，断言位置、速度、航向、加速度和转向率来自已执行目标运动。
- [ ] 写等时间戳替换、逆序样本拒绝和 bounded retention 测试。
- [ ] 写 checkpoint/restore 确定性测试。
- [ ] 写 UUV-only 预测读取全局轨迹、非 UUV-only 仍读取估计历史的边界测试。
- [ ] 运行新测试，确认 GlobalTrackStore 尚不存在。
- [ ] 实现有界轨迹存储和派生运动特征。
- [ ] 在 SimulationEngine 物理状态提交后更新轨迹，禁止读取目标未执行决策。
- [ ] 将轨迹证据 ID 持久化为公共蓝方证据。
- [ ] 验证同种子产生相同轨迹摘要。

验证命令：

    python -m pytest tests/tracking/test_global_track.py tests/simulation/test_engine.py tests/cli/test_cli.py -q

提交：

    feat: expose globally known target trajectory

## Task 3：实现完整 IMM 多模型预测

文件：

- 新建 src/underwater_tracking/prediction/imm_forecast.py
- 修改 src/underwater_tracking/tracking/imm.py
- 修改 src/underwater_tracking/domain/models.py
- 修改 src/underwater_tracking/domain/agent_models.py
- 修改 src/underwater_tracking/prediction/port.py
- 新建 tests/prediction/test_imm_forecast.py
- 修改 tests/prediction/test_port.py
- 修改 tests/tracking/test_imm.py

接口：

- IMMModelStateProjection
- IMMForecastResult
- forecast_imm
- moment_match_forecasts

步骤：

- [ ] 写 CV 直线传播测试。
- [ ] 写 CT_LEFT 和 CT_RIGHT 对称转向测试。
- [ ] 写模型概率归一化和概率加权均值测试。
- [ ] 写矩匹配协方差测试，确保同时包含模型内部不确定性和模型分歧。
- [ ] 写最大速度和最大转向裁剪测试。
- [ ] 写正常长历史必须返回 prediction_regime=imm 的测试。
- [ ] 写 IMM 状态缺失回退 bspline、历史不足回退 short_history 的测试。
- [ ] 运行预测测试，确认当前实现只提供混合状态或 B 样条。
- [ ] 从 IMM estimator 导出三个模型的均值、协方差、概率、创新和似然。
- [ ] 实现各模型完整时域传播和矩匹配。
- [ ] 输出逐点二维协方差、地图置信半径和模型分支。
- [ ] 保留 prediction diff 的绝对时间对齐语义。
- [ ] 更新 PredictionRegime 和帧序列化。

验证命令：

    python -m pytest tests/tracking/test_imm.py tests/prediction/test_imm_forecast.py tests/prediction/test_port.py tests/prediction/test_diff.py -q

提交：

    feat: forecast target motion from full IMM state

## Task 4：实现确定性意图基线和 LLM 修订门

文件：

- 新建 src/underwater_tracking/intent/deterministic.py
- 修改 src/underwater_tracking/agent/nodes/intent.py
- 修改 src/underwater_tracking/agent/graphs/central.py
- 修改 src/underwater_tracking/agent/runtime.py
- 修改 src/underwater_tracking/domain/agent_models.py
- 新建 tests/intent/test_deterministic_intent.py
- 修改 tests/agent/test_prediction_intent_wiring.py
- 修改 tests/agent/test_semantic_nodes.py

接口：

- MotionIntentFeatures
- DeterministicIntentClassifier
- IntentLatchState
- ConfirmedIntentRevision

步骤：

- [ ] 写 transit、loiter、patrol、evade、approach、withdraw 的轨迹样例测试。
- [ ] 写进入阈值、退出阈值和连续周期确认测试。
- [ ] 写一次模型概率抖动不能改变语义意图的测试。
- [ ] 写 LLM 超时、低置信度、证据越界和过期预测仍保留确定性意图的测试。
- [ ] 写两次高置信度一致修订才能替代基线的测试。
- [ ] 运行新测试，确认当前路径依赖 LLM 或 unknown。
- [ ] 实现运动特征提取、规则分类和滞回状态。
- [ ] 将确定性意图放入每次预测刷新结果。
- [ ] 将 LLM 定位为异步解释和受限修订，不再作为区域滚动前置条件。
- [ ] 持久化规则版本、阈值、LLM provenance 和证据。

验证命令：

    python -m pytest tests/intent/test_deterministic_intent.py tests/agent/test_prediction_intent_wiring.py tests/agent/test_semantic_nodes.py -q

提交：

    feat: add deterministic intent baseline with LLM revision

## Task 5：实现沿 IMM 中心线的四区域生成器

文件：

- 新建 src/underwater_tracking/planning/dynamic_regions.py
- 修改 src/underwater_tracking/planning/regions.py
- 修改 src/underwater_tracking/domain/regional_models.py
- 修改 src/underwater_tracking/agent/nodes/regions.py
- 新建 tests/planning/test_dynamic_regions.py
- 修改 tests/planning/test_regions.py
- 修改 tests/planning/test_regional_standoff.py

接口：

- RegionWindowPolicy
- DynamicRegionChain
- build_dynamic_region_chain
- normalize_region_chain

步骤：

- [ ] 写严格四区域和稳定 ID 测试。
- [ ] 写默认 0-540、450-990、900-1440、1350-1800 时间窗测试。
- [ ] 写每个区域包含对应中心线样本的属性测试。
- [ ] 写相邻区域允许受控重叠、非相邻区域必须裁剪的属性测试。
- [ ] 写转弯中心线下区域随曲线旋转或分段包络的测试，防止退化为主轴大矩形。
- [ ] 写地图边界裁剪后最小宽度和面积测试。
- [ ] 写 LLM 缺失中心线、重叠、逆序和越界建议均能规范化的测试。
- [ ] 写无法修复时保留上一可执行区域链的测试。
- [ ] 运行测试，确认现有固定矩形或旧规范化不满足新契约。
- [ ] 实现按预测时间和弧长分段的区域生成。
- [ ] 实现稳定区域槽位、geometry_revision 和交接拓扑。
- [ ] 保留底层候选单元仅供审计，不进入执行集合。

验证命令：

    python -m pytest tests/planning/test_dynamic_regions.py tests/planning/test_regions.py tests/planning/test_regional_standoff.py -q

提交：

    feat: generate four rolling IMM task regions

## Task 6：实现四个两艇 task group 和四艇替补池

文件：

- 新建 src/underwater_tracking/planning/task_groups.py
- 修改 src/underwater_tracking/planning/mission_optimizer.py
- 修改 src/underwater_tracking/domain/mission_models.py
- 修改 src/underwater_tracking/runtime/mission_controller.py
- 新建 tests/planning/test_task_groups.py
- 修改 tests/planning/test_mission_optimizer.py
- 修改 tests/runtime/test_mission_controller.py

接口：

- TaskGroupPolicy
- TaskGroupAllocator
- ReplacementQueue
- allocate_four_task_groups

步骤：

- [ ] 写 12 艘健康 UUV 必须产生四个两艇组和四艇替补池的测试。
- [ ] 写每组优先一艘主动能力艇和一艘被动艇的测试。
- [ ] 写 UUV 不能重复分配的测试。
- [ ] 写资源不足时明确 degraded 且不伪造成员的测试。
- [ ] 写滚动 revision 优先保持当前组连续性的测试。
- [ ] 写区域几何改变但槽位 ID 稳定时尽量保留成员的测试。
- [ ] 运行测试，确认现有优化器只部署当前和直接后继。
- [ ] 实现专用四区域 task group 分配器。
- [ ] 从 MissionOptimizer 中隔离旧批次和航母资源逻辑。
- [ ] 将 task group 作为完整执行状态提交给 MissionController。
- [ ] 为替补池建立确定性优先级。

验证命令：

    python -m pytest tests/planning/test_task_groups.py tests/planning/test_mission_optimizer.py tests/runtime/test_mission_controller.py -q

提交：

    feat: allocate four two-UUV regional task groups

## Task 7：完成区域约束航点和边界替换执行

文件：

- 修改 src/underwater_tracking/simulation/engine.py
- 修改 src/underwater_tracking/runtime/mission_controller.py
- 修改 src/underwater_tracking/domain/mission_models.py
- 新建 tests/simulation/test_task_group_waypoints.py
- 修改 tests/simulation/test_mission_waypoint_geometry.py
- 修改 tests/simulation/test_uuv_boundary_rotation.py
- 修改 tests/integration/test_truthful_bootstrap_deployment_frames.py

接口：

- plan_task_group_waypoints
- begin_boundary_exit
- complete_boundary_exit
- begin_boundary_entry
- complete_boundary_replacement

步骤：

- [ ] 写同一目标不同区域的航点历史互不覆盖测试。
- [ ] 写当前区域组围绕全局目标形成测向几何的测试。
- [ ] 写下一交接组围绕预测进入点预置的测试。
- [ ] 写未来组航点始终位于所属区域的测试。
- [ ] 写最大位移、最大转向和最小艇间距测试。
- [ ] 写不可用 UUV 向最近区域边界驶出并逐渐消失的测试。
- [ ] 写替补从同一区域边界进入并接管同一任务槽位的测试。
- [ ] 写新运行绝不发出航母投放、回收和返航事件的测试。
- [ ] 运行测试，确认航点缓存冲突或旧生命周期仍可达。
- [ ] 将区域任务航点缓存键改为 task_group_id + region_id。
- [ ] 在融合报告未收敛时使用全局目标或预测进入点，不围绕组质心盲目扩散。
- [ ] 删除或隔离 UUV-only 航母路径，使其不可从新执行快照触发。
- [ ] 保留旧回放事件的读取兼容。

验证命令：

    python -m pytest tests/simulation/test_task_group_waypoints.py tests/simulation/test_mission_waypoint_geometry.py tests/simulation/test_uuv_boundary_rotation.py tests/integration/test_truthful_bootstrap_deployment_frames.py -q

提交：

    feat: execute regional UUV groups through boundary rotation

## Task 8：建立 ExecutionCoordinator 和原子版本提交

文件：

- 新建 src/underwater_tracking/runtime/execution_coordinator.py
- 修改 src/underwater_tracking/cli.py
- 修改 src/underwater_tracking/agent/runtime.py
- 修改 src/underwater_tracking/runtime/mission_controller.py
- 修改 src/underwater_tracking/persistence/plans.py
- 新建 tests/runtime/test_execution_coordinator.py
- 修改 tests/cli/test_cli.py
- 修改 tests/runtime/test_planning_epoch.py

接口：

- ExecutionCoordinator.current
- ExecutionCoordinator.propose
- ExecutionCoordinator.commit
- ExecutionCoordinator.preserve
- ExecutionCommitResult

步骤：

- [ ] 写启动 revision 1 立即可执行测试。
- [ ] 写每 450 仿真秒滚动检查测试。
- [ ] 写预测离开区域链立即重划测试。
- [ ] 写 compare-and-set 拒绝旧 base revision 的测试。
- [ ] 写物理 revision 前进但证据仍有效时允许受控 rebase 的测试。
- [ ] 写目标、资源或人工版本改变时拒绝 rebase 的测试。
- [ ] 写提交失败保留当前快照和运动的测试。
- [ ] 写 active mission reader 必须返回最高已验证 revision 的测试。
- [ ] 运行测试，确认现有计划、任务和发布状态可能混合。
- [ ] 实现不可变候选、CAS 提交和 preserve 结果。
- [ ] 让确定性滚动与 LLM commit 使用同一个协调器。
- [ ] 持久化 execution revisions 和提交报告。
- [ ] 将 TrackingPlan 改为新执行快照的审计投影。

验证命令：

    python -m pytest tests/runtime/test_execution_coordinator.py tests/runtime/test_planning_epoch.py tests/cli/test_cli.py -q

提交：

    feat: coordinate atomic UUV execution revisions

## Task 9：收紧 LLM 增量优化契约

文件：

- 修改 src/underwater_tracking/agent/prompts.py
- 修改 src/underwater_tracking/agent/nodes/regions.py
- 修改 src/underwater_tracking/agent/nodes/optimize.py
- 修改 src/underwater_tracking/agent/graphs/central.py
- 修改 src/underwater_tracking/domain/regional_models.py
- 修改 src/underwater_tracking/persistence/ledger.py
- 新建 tests/agent/test_execution_strategy_contract.py
- 修改 tests/agent/test_regional_strategy.py
- 修改 tests/agent/test_background_cycle.py

接口：

- ExecutionStrategyProposal
- RegionSlotPolicy
- StrategyValidationReport

步骤：

- [ ] 写 LLM 只能引用四个既有区域槽位的测试。
- [ ] 写任意多边形、第五区域、未知 UUV、未知证据和非法枚举被拒绝的测试。
- [ ] 写 provider timeout、invalid output 和 stale 输出均保留 active plan 的测试。
- [ ] 写 LLM 运行时帧和物理位置持续推进的测试。
- [ ] 写人工确认前建议不能进入规划 mailbox 的测试。
- [ ] 运行测试，确认现有 LLM 仍可能控制区域几何或阻塞结果。
- [ ] 将 LLM 输出改为时间窗比例、宽度、重叠、角色和优先级建议。
- [ ] 将所有几何和航点交给确定性规范化器。
- [ ] 记录模型、prompt 版本、请求响应哈希、基础执行版本和校验失败字段。
- [ ] 明确 planning health 状态和 active_plan_preserved 字段。

验证命令：

    python -m pytest tests/agent/test_execution_strategy_contract.py tests/agent/test_regional_strategy.py tests/agent/test_background_cycle.py -q

提交：

    feat: constrain LLM execution strategy revisions

## Task 10：统一操作帧、HTTP、WebSocket 和 Replay

文件：

- 修改 src/underwater_tracking/api/frame_builder.py
- 修改 src/underwater_tracking/api/live.py
- 修改 src/underwater_tracking/api/app.py
- 修改 src/underwater_tracking/api/hub.py
- 修改 src/underwater_tracking/api/replay.py
- 修改 src/underwater_tracking/domain/ui_models.py
- 新建 tests/api/test_execution_frame_contract.py
- 修改 tests/api/test_live_publisher.py
- 修改 tests/api/test_app.py
- 修改 tests/api/test_replay_compatibility.py

接口：

- ExecutionView
- ExecutionRegionView
- TaskGroupView
- FrameConsistencyReport

步骤：

- [ ] 写一个目标只发布四个 regional_missions 的测试。
- [ ] 写候选网格不得进入执行区域集合的测试。
- [ ] 写 mission region polygon 在候选缓存缺失时仍完整发布的测试。
- [ ] 写 execution、region 和 task group revision 必须一致的测试。
- [ ] 写 HTTP snapshot、hub snapshot 和 replay JSON 完全一致的测试。
- [ ] 写不一致语义状态发布上一有效执行快照并标记 degraded 的测试。
- [ ] 写新 UUV-only 帧不含航母、舰船和旧生命周期字段的测试。
- [ ] 运行测试，复现当前双粒度和混合 revision。
- [ ] 让 publisher 读取单个 OperationalExecutionSnapshot。
- [ ] 使用同一个序列化对象发布全部通道。
- [ ] 保留旧回放读取适配，不将旧字段补入新实时帧。

验证命令：

    python -m pytest tests/api/test_execution_frame_contract.py tests/api/test_live_publisher.py tests/api/test_app.py tests/api/test_replay_compatibility.py -q

提交：

    feat: publish one coherent UUV execution frame

## Task 11：统一前端 frame store 和地图任务表现

文件：

- 修改 src/underwater_tracking/ui/src/types/frames.ts
- 修改 src/underwater_tracking/ui/src/hooks/useWebSocket.ts
- 修改 src/underwater_tracking/ui/src/components/CanvasMap.tsx
- 修改 src/underwater_tracking/ui/src/components/RegionTimeline.tsx 或当前等价组件
- 修改 src/underwater_tracking/ui/src/App.tsx
- 修改相关 CSS
- 修改 src/underwater_tracking/ui/src/components/CanvasMap.test.ts
- 新建或修改 frame store 测试
- 修改 src/underwater_tracking/ui/e2e/task-region-effect.spec.ts
- 修改 src/underwater_tracking/ui/e2e/uuv-live-timeline.spec.ts

步骤：

- [ ] 写旧 frame_id 不能覆盖新快照的 reducer 测试。
- [ ] 写帧跳跃触发 HTTP snapshot 补偿的测试。
- [ ] 写存在 regional_missions 时不显示同目标 cell 网格的测试。
- [ ] 写地图严格显示四区域、四 task group 和八艘执行 UUV 的测试。
- [ ] 写当前组显示详细标签、其他组显示简洁标签的测试。
- [ ] 写相机只包围目标、预测、四区域和执行 UUV 的测试。
- [ ] 写区域箭头和时间线只有四行的测试。
- [ ] 写无航母、无回收连线和无候选网格的测试。
- [ ] 写 WebSocket 断连后心跳和收发任务全部回收的测试。
- [ ] 运行 Vitest，确认现有显示函数会合并不同粒度区域。
- [ ] 将前端 store 改为单一 OperationalFrame reducer。
- [ ] 删除 CanvasMap 的规划与执行区域自合并。
- [ ] 实现预测走廊、四区域、交接、组标签和短尾迹的稳定视觉层。
- [ ] 使用真实后端帧驱动 Playwright，不把合成 fixture 当作实跑证据。

验证命令：

    npm --prefix src/underwater_tracking/ui test -- --run
    npm --prefix src/underwater_tracking/ui run build
    npm --prefix src/underwater_tracking/ui run test:e2e -- task-region-effect.spec.ts

提交：

    feat: render coherent four-region UUV execution

## Task 12：打通助手、反馈、记忆和证据查询

文件：

- 修改 src/underwater_tracking/api/app.py
- 修改 src/underwater_tracking/agent/runtime.py
- 修改 src/underwater_tracking/api/dependencies.py
- 修改 src/underwater_tracking/memory/source_reader.py
- 修改 src/underwater_tracking/memory/service.py
- 修改 src/underwater_tracking/persistence/ledger.py
- 修改前端助手组件
- 新建 tests/api/test_execution_evidence.py
- 修改 tests/api/test_app.py
- 修改 tests/memory/test_source_reader.py
- 修改助手 UI 测试

步骤：

- [ ] 写“为何这样制定方案”返回轨迹、IMM、意图、区域、task group 和提交证据的测试。
- [ ] 写证据 ID 缺失时明确返回 unresolved_evidence 的测试。
- [ ] 写证据查询不改变 execution_revision 的测试。
- [ ] 写方案建议只有人工确认后进入 mailbox 的测试。
- [ ] 写 conversation、memory、memory stream、questions、directives、assignments 和 sensor modes 共享 execution_revision 与 frame_id 的测试。
- [ ] 写记忆 LLM 失败仍返回执行上下文和 degraded memory 状态的测试。
- [ ] 运行测试，确认当前接口可能读取不同版本状态。
- [ ] 建立 ExecutionDecisionRecord 和统一 EvidenceResolver。
- [ ] 将短期和长期记忆来源绑定到 execution revision。
- [ ] 在助手 UI 显示算法、LLM、人工三类贡献和可点击证据。

验证命令：

    python -m pytest tests/api/test_execution_evidence.py tests/api/test_app.py tests/memory/test_source_reader.py -q
    npm --prefix src/underwater_tracking/ui test -- --run

提交：

    feat: trace assistant answers to execution evidence

## Task 13：统一运行目录和进程监管

文件：

- 新建 src/underwater_tracking/runtime/process_supervisor.py
- 修改 src/underwater_tracking/runtime/run_controller.py
- 修改 src/underwater_tracking/runtime/run_catalog.py
- 修改 src/underwater_tracking/cli.py
- 修改 main.py
- 修改 src/underwater_tracking/api/app.py
- 新建 tests/runtime/test_process_supervisor.py
- 修改 tests/runtime/test_run_controller.py
- 修改 tests/cli/test_cli.py
- 修改 tests/api/test_app_lifespan.py

步骤：

- [ ] 写一次 main.py 调用只创建一个 run-* 的测试。
- [ ] 写正式运行不创建 serve-* 的测试。
- [ ] 写所有组件接收同一个 run_dir 的测试。
- [ ] 写正式前端由 FastAPI 静态托管、不启动 Vite 子进程的测试。
- [ ] 写关闭顺序和幂等 close 测试。
- [ ] 写 LLM 或记忆 worker 超时仍产生 shutdown report 的测试。
- [ ] 写退出后端口关闭、子进程为空、日志不再增长的测试。
- [ ] 运行测试，复现目录重复或残留生命周期。
- [ ] 让 RunController 成为唯一 run 目录和资源所有者。
- [ ] 实现 ProcessSupervisor 的线程、子进程、端口和文件句柄登记。
- [ ] 生成 process-shutdown.json。
- [ ] 将旧 serve 入口改为连接已有 run 的开发辅助路径，或从正式 CLI 中移除。

验证命令：

    python -m pytest tests/runtime/test_process_supervisor.py tests/runtime/test_run_controller.py tests/cli/test_cli.py tests/api/test_app_lifespan.py -q

提交：

    fix: own one run directory and close all workers

## Task 14：真实前后端数据链和八分钟完整验收

文件：

- 修改 tests/integration/test_uuv_only_production_acceptance.py
- 修改 tests/acceptance/test_default_live_acceptance_driver.py
- 修改 src/underwater_tracking/ui/e2e/uuv-live-timeline.spec.ts
- 修改 src/underwater_tracking/verification/live_demo.py
- 修改 scripts/monitor_main_battle.py
- 修改 tests/verification/test_live_demo_monitor.py
- 生成当前 run 目录中的验收报告和截图

### 14.1 自动化前置验证

- [ ] 运行完整后端定向集合。
- [ ] 运行完整前端 Vitest。
- [ ] 运行 TypeScript build。
- [ ] 运行固定效果 Playwright。
- [ ] 启动真实 main.py，设置 PLAYWRIGHT_BASE_URL，运行真实数据链 Playwright。
- [ ] 验证助手 POST、记忆查询、memory stream 和证据问答。
- [ ] 验证 HTTP、WebSocket 和 Replay 的 frame_id、execution_revision 和区域集合一致。

### 14.2 八分钟运行

- [ ] 使用 underwater-tracking 环境启动一次正式 main.py。
- [ ] 记录运行前 outputs 目录快照。
- [ ] 运行 480 秒墙钟时间，目标 sim_time_s 至少达到 28800。
- [ ] 在 0 至 30 秒验证四区域和八艇边界进入。
- [ ] 在 30 至 120 秒提交助手反馈并读取记忆状态。
- [ ] 在 120 至 240 秒观察目标机动、IMM 变化和区域滚动。
- [ ] 在 240 至 360 秒观察交接和替换。
- [ ] 在 360 至 450 秒执行证据问答并核对来源。
- [ ] 在 450 至 480 秒截取桌面和移动端真实地图。
- [ ] 正常关闭并检查端口、PID 和目录。

### 14.3 必须通过的断言

- [ ] frame_id 持续递增。
- [ ] final sim_time_s 至少 28800。
- [ ] 始终存在有效 execution revision。
- [ ] 每个真实帧只有四个任务区域。
- [ ] 正常资源下有八艘执行 UUV。
- [ ] 每个区域恰好一个两艇 task group。
- [ ] 当前区域组实际跟踪目标。
- [ ] 至少一次区域 revision 滚动。
- [ ] 至少一次区域交接。
- [ ] 至少一次目标机动和蓝方响应。
- [ ] LLM 非法区域建议期间 UUV 仍运动。
- [ ] 没有航母投放、回收或返航事件。
- [ ] 助手、记忆和证据查询成功。
- [ ] 桌面和移动端无空白、无重叠和无横向溢出。
- [ ] 本次调用只新增一个 run-*。
- [ ] 本次调用不新增 serve-*。
- [ ] 关闭后无残留子进程或监听端口。

验证命令：

    python -m pytest -q
    npm --prefix src/underwater_tracking/ui test -- --run
    npm --prefix src/underwater_tracking/ui run build
    npm --prefix src/underwater_tracking/ui run test:e2e
    python scripts/monitor_main_battle.py --main main.py --scenario configs/scenario/uuv_only_single_target.yaml --wall-timeout-s 600 --expected-duration-s 28800 --require-real-provider

提交：

    test: prove global-track regional execution closure

## 最终审查门

实施完成后必须依次执行：

1. 系统性调试所有失败，不允许通过放宽断言掩盖真实缺陷。
2. 检查 OperationalExecutionSnapshot 是否成为唯一 UUV-only 执行来源。
3. 检查旧航母路径是否在新运行中不可达。
4. 检查固定截图与真实后端截图的数据来源区别是否明确。
5. 检查所有新增事件和证据 ID 可从持久化存储解析。
6. 检查所有测试、构建、Playwright 和八分钟运行证据。
7. 检查 git diff 中没有无关重构、生成物或用户文件。
8. 通过代码审查和完成前验证后再合并。

## 完成定义

只有当 Task 1 至 Task 14 全部完成，后端、前端和真实八分钟验收全部通过，并且运行只生成一个 run-*、无 serve-*、无残留进程时，才能声明本计划实施完成。
