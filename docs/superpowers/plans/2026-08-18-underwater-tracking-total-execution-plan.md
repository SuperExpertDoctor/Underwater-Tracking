# Underwater Tracking Total Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 将现有水下跟踪项目整合为一个可运行、可回放、可解释的单目标区域化跟踪指挥台：由真实 LLM 决定区域级 UUV/USV 编组和协作模式，确定性层负责安全校验与执行派生，目标具备平滑的敌我博弈机动，界面聚焦预测区域并通过知识图谱、时间轴和统一 LLM Client 展示跟踪效果。

**Architecture:** TargetRegionPlan/RegionTask 是空间和时间任务的唯一事实来源；预测数学负责中心线和不确定性走廊，LLM 负责区域策略、模式和成员，确定性层只检查实体、能力、链路、运动学和安全边界，不注入固定编组数量。OperationalFrame 输出区域、责任、效果、事件和对话结果，Canvas 地图、SVG 图谱、时间轴和右侧工作区都从该 frame 渲染。统一对话入口先由结构化 LLM 分类，再路由到只读证据回答或现有 directive preview/apply 安全链路。

**Tech Stack:** Python 3.10 / Pydantic / LangGraph / SQLite / FastAPI-WebSocket, React 18 / TypeScript / Vite / Vitest, HTML Canvas / SVG, Playwright.

---

## 0. 基线与执行规则

项目根目录：/home/shuixia/users/houguoqiang/projects/Underwater-Tracking

当前远端 master 存在已有未提交实现变更，且设计提交为 9794e8e。执行前必须记录状态，每个任务只暂存自己的文件，不回滚既有脏文件。

已有基础实现包括区域模型、栅格生成、区域策略 schema、验证/分配骨架、部分图路由、LLM 诊断、启动帧、折叠面板、回放和 SVG 图谱。总计划只补齐集成和产品化，不重复实现这些基础模块。

规则：

- main.py 是唯一验收入口，后端使用 lang_py310。
- LLM 决定区域策略和具体成员；界面、验证和启发式代码不得暗中增加 UUV/USV 数量限制。
- 保持两类协作：uuv_primary_usv_relay，以及互不混合主动跟踪域的 heuristic_uuv / heuristic_usv。
- 显示比例与世界坐标分离，改相机、雷达显示范围和图标大小不得改变仿真物理或任务分配。
- 证据质询只读；方案修正必须经过 preview、版本校验和显式 apply。

基线命令：

~~~bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking
git status --short --branch
/home/shuixia/miniconda3/envs/lang_py310/bin/python -m compileall -q src main.py
~~~

---

## 1. 接通区域计划数据链

依赖：现有区域模型、栅格、策略和分配基础。

文件：

- Modify: src/underwater_tracking/agent/nodes/optimize.py
- Modify: src/underwater_tracking/agent/nodes/verify.py
- Modify: src/underwater_tracking/agent/nodes/commit.py
- Modify: src/underwater_tracking/domain/agent_models.py
- Modify: src/underwater_tracking/agent/state.py
- Test: tests/agent/test_regional_plan_pipeline.py
- Test: tests/agent/test_plan_pipeline.py

步骤：

- [ ] 增加失败测试：区域任务先于 target-level 兼容字段成为事实来源；成员、角色、航点均从区域任务稳定派生；degraded/uncovered 区域不能被丢弃。
- [ ] 让 OptimizeNode 消费区域策略并调用 allocate_regional_tasks，先构造 region_tasks，再派生 member_ids_by_target、roles_by_member 和 waypoints_by_member。
- [ ] 让 verify 在旧计划检查前执行区域几何、角色、声纳、通信、relay 半径、重复占用和接力检查。
- [ ] 给 PlanCommand 增加可选 region_id，保留 group_id 和全部旧执行字段。
- [ ] 输出区域质量、覆盖率、relay、降级指标，并明确 proxy 指标不是传感器真值。
- [ ] 运行：

~~~bash
conda run -n lang_py310 python -m pytest tests/agent/test_regional_plan_pipeline.py tests/agent/test_plan_pipeline.py -q
~~~

- [ ] 提交：feat: derive legacy plans from regional tasks

---

## 2. 固化 LLM 自主编组、两类协作和反馈上下文

依赖：任务 1。

文件：

- Modify: src/underwater_tracking/domain/regional_models.py
- Modify: src/underwater_tracking/planning/regional_allocation.py
- Modify: src/underwater_tracking/planning/regional_validation.py
- Modify: src/underwater_tracking/domain/agent_models.py
- Modify: src/underwater_tracking/agent/nodes/directives.py
- Modify: src/underwater_tracking/agent/runtime.py
- Modify: src/underwater_tracking/agent/nodes/strategy.py
- Modify: src/underwater_tracking/agent/prompts.py
- Test: tests/planning/test_regional_allocation.py
- Test: tests/planning/test_regional_validation.py
- Test: tests/agent/test_directives.py
- Test: tests/agent/test_assignment_directives.py

步骤：

- [ ] 测试 LLM 成员集合被原样保留，不填充 advisory count；空集合为 uncovered；未知、重复、不可用成员被拒绝或降级。
- [ ] 测试 relay 模式允许多 UUV 跟踪加 USV relay；heuristic_uuv 拒绝 USV 主跟踪；heuristic_usv 拒绝 UUV 主跟踪；不产生混合主动跟踪域。
- [ ] 测试 feedback directive 能带 target/region 范围，进入下一轮 strategy context，但不直接改写 assignment。
- [ ] 将 required_uuv_count / required_usv_count 降为解释性 metadata；保留能力、链路、运动学和安全硬约束。
- [ ] 将区域效果、降级原因、目标/预测 revision 和专家反馈加入策略上下文，LLM 改变成员时必须显式返回 ID。
- [ ] 运行：

~~~bash
conda run -n lang_py310 python -m pytest tests/planning/test_regional_allocation.py tests/planning/test_regional_validation.py tests/agent/test_directives.py tests/agent/test_assignment_directives.py -q
~~~

- [ ] 提交：feat: let llm choose regional tracking teams

---

## 3. 完成战略/战术 LangGraph 路由

依赖：任务 1-2。

文件：

- Modify: src/underwater_tracking/agent/graphs/central.py
- Modify: src/underwater_tracking/agent/state.py
- Modify: src/underwater_tracking/agent/runtime.py
- Test: tests/agent/test_central_graph.py
- Create or modify: tests/agent/test_regional_graph.py
- Test: tests/integration/test_agent_loop.py

步骤：

- [ ] 测试 trajectory_prediction -> region_generation -> regional_strategy -> verify -> optimize -> commit 的顺序和错误路由。
- [ ] 测试有效区域策略在普通 tactical cycle 中复用，不重复调用 regional strategy LLM。
- [ ] 测试 intent 变化、目标丢失/重捕获、协方差阈值、relay 半径、续航、链路、情报/方案变化和区域反馈会触发战略重规划。
- [ ] 确保确定性错误进入 handle_error，真正 LLMError 保留现有 retry/pause 语义。
- [ ] 运行 agent graph 和 agent-loop 测试。
- [ ] 提交：feat: route carrier graph through regional strategy

---

## 4. 持久化区域 revision 并构建 live/replay frame

依赖：任务 1-3。

文件：

- Modify: src/underwater_tracking/persistence/plans.py
- Modify: src/underwater_tracking/persistence/ledger.py
- Modify: src/underwater_tracking/persistence/sqlite.py
- Modify: src/underwater_tracking/api/frame_builder.py
- Modify: src/underwater_tracking/api/live.py
- Modify: src/underwater_tracking/api/replay.py
- Modify: src/underwater_tracking/domain/ui_models.py
- Modify: src/underwater_tracking/domain/decision_models.py
- Test: tests/api/test_frame_builder_regional_views.py
- Create or modify: tests/persistence/test_regional_replay.py

步骤：

- [ ] 测试 GridSpec、cell、visit window、角色、链路、声纳、handoff、降级原因、证据、LLM hash、触发事件和 revision 的 round-trip。
- [ ] 增加 TrackingEffectView、RegionTaskView、RegionalPlanView；状态限制为 planned/active/handoff_ready/degraded/uncovered，比例限制在 0-1。
- [ ] 从区域任务和现有 GroupQualityView 构建 frame，未测量质量明确标记 group_quality_proxy。
- [ ] 使用现有 JSON payload 存储区域数据，旧 frame 字段保持可选且可回放。
- [ ] frame 必须包含一个目标的有序区域任务、当前/下一接力、效果和因果事件。
- [ ] 运行：

~~~bash
conda run -n lang_py310 python -m pytest tests/api/test_frame_builder_regional_views.py tests/api/test_frame_contracts.py tests/persistence/test_regional_replay.py -q
~~~

- [ ] 提交：feat: persist regional revisions and replay data

---

## 5. 实现平滑敌方机动和快速蓝方响应

依赖：任务 1-4。

文件：

- Modify: src/underwater_tracking/simulation/target.py
- Modify: src/underwater_tracking/simulation/engine.py
- Modify: src/underwater_tracking/agent/nodes/adversary.py
- Modify: src/underwater_tracking/agent/nodes/strategy.py
- Test: tests/simulation/test_target.py
- Test: tests/simulation/test_engine.py
- Test: tests/agent/test_adversary_graph.py
- Test: tests/agent/test_runtime_master_slave_adversary.py

步骤：

- [ ] 测试位置连续、加速度/转向率有界、无瞬时航向跳变和 seeded 可复现。
- [ ] 在 target.py 增加 desired heading/speed、turn-rate、acceleration、expiry 和插值。
- [ ] 为 adversary 增加 cooldown/hysteresis 和 revision threshold，避免每个仿真 tick 都调用完整策略链。
- [ ] 目标机动、区域质量下降或 relay 失效超过阈值时，触发蓝方快速 regional replan。
- [ ] 记录 target maneuver -> prediction revision -> regional task revision -> effect change -> blue response 事件链和延迟。
- [ ] 运行：

~~~bash
conda run -n lang_py310 python -m pytest tests/simulation/test_target.py tests/simulation/test_engine.py tests/agent/test_adversary_graph.py tests/agent/test_runtime_master_slave_adversary.py -q
~~~

- [ ] 提交：feat: model smooth adversarial maneuver response

---

## 6. 加入显示尺度、预测走廊聚焦和细粒度地图

依赖：任务 4；可与任务 5 并行开发。

文件：

- Create: src/underwater_tracking/ui/src/types/viewConfig.ts
- Modify: src/underwater_tracking/ui/src/components/CanvasMap.tsx
- Modify: src/underwater_tracking/ui/src/components/map/geometry.ts
- Modify: src/underwater_tracking/ui/src/App.tsx
- Modify: src/underwater_tracking/ui/src/App.css
- Test: src/underwater_tracking/ui/src/components/CanvasMap.test.ts
- Test: src/underwater_tracking/ui/src/types/regionalTasks.test.ts

步骤：

- [ ] 测试默认 camera 只适配目标均值、预测中心线和可见区域，不把隐藏 detection range 纳入 bounds；测试图标尺寸按屏幕像素 clamp。
- [ ] 增加 ViewConfig：focusMode、radarScale、predictionPadding、gridDivisions、target/uuv/usv markerPixels、playbackRate。
- [ ] 默认 focusMode 为 prediction_corridor，showDetectionRange 为 false，gridDivisions 为 16；这些值不得进入后端规划 payload。
- [ ] 以预测走廊加 15% padding 自动适配，只有 full_area 或显式开层才包含探测范围。
- [ ] 保留细粒度区域数据用于 hit test；低缩放时聚合标签，高缩放时显示 R01/R02 等稳定编号。
- [ ] UUV/USV/target 圆环和角色状态默认显示，不再依赖点击。
- [ ] 运行：

~~~bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui
npm test -- --run src/components/CanvasMap.test.ts src/types/regionalTasks.test.ts
npm run build
~~~

- [ ] 提交：feat: focus command center on prediction corridor

---

## 7. 完成区域/实体知识图谱和地图区域层

依赖：任务 4、6。

文件：

- Create or modify: src/underwater_tracking/ui/src/components/assistant/RegionTaskGraph.tsx
- Create or modify: src/underwater_tracking/ui/src/components/assistant/RegionTaskGraph.css
- Test: src/underwater_tracking/ui/src/components/assistant/RegionTaskGraph.test.tsx
- Create: src/underwater_tracking/ui/src/components/map/RegionOverlay.tsx
- Test: src/underwater_tracking/ui/src/components/map/RegionOverlay.test.tsx
- Modify: src/underwater_tracking/ui/src/components/CanvasMap.tsx
- Modify: src/underwater_tracking/ui/src/types/frames.ts

步骤：

- [ ] 测试四区域、三 UUV、两 USV、时间箭头、责任边、relay 边、空计划和 64+ 区域布局。
- [ ] 方形区域节点显示 R01/R02，圆形节点显示 UUV/USV；时间边有箭头，主动跟踪责任边实线，relay 边虚线。
- [ ] 地图区域显示 probability/priority/active/handoff/degraded/uncovered 状态，图谱、地图和时间轴双向选择。
- [ ] 图谱宽度可滚动，节点至少 96x44px，实体圆至少 24px；relay 与主动跟踪责任分开标识。
- [ ] 运行：

~~~bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui
npm test -- --run src/components/assistant/RegionTaskGraph.test.tsx src/components/map/RegionOverlay.test.tsx
npm run build
~~~

- [ ] 提交：feat: visualize regional handoff knowledge graph

---

## 8. 将右侧重组为三个主卡

依赖：任务 4、6、7。

文件：

- Modify: src/underwater_tracking/ui/src/components/RightSidebar.tsx
- Modify: src/underwater_tracking/ui/src/components/SidebarPanels.css
- Modify: src/underwater_tracking/ui/src/App.tsx
- Modify: src/underwater_tracking/ui/src/components/assistant/AssignmentPanel.tsx
- Modify: src/underwater_tracking/ui/src/components/assistant/AssignmentReview.tsx
- Test: src/underwater_tracking/ui/src/components/RightSidebar.test.tsx

步骤：

- [ ] 测试右侧只有 当前态势、预测与接力、LLM Client 三个主区；不再显示独立专家反馈、态势问答或方案约束卡。
- [ ] 当前态势承载目标/对手、仿真、质量、覆盖、brain、carrier、UUV/USV roster 和链路状态。
- [ ] 预测与接力承载 RegionTaskGraph、区域效果、只读 AssignmentPanel，并提供 graph/timeline/list 三视图。
- [ ] 使用原生 details/summary 或 aria-expanded，预测卡获得最大滚动空间；桌面宽度 360-440px，窄屏变 drawer。
- [ ] 参考 Maritime-Surveillance-main 的地图优先、底部抽屉、紧凑指标和稳定方格布局，但不复制其业务字段。
- [ ] 运行：

~~~bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui
npm test -- --run src/components/RightSidebar.test.tsx src/components/assistant/AssignmentPanel.test.tsx
npm run build
~~~

- [ ] 提交：feat: organize command center sidebar

---

## 9. 合并专家反馈和证据质询为统一 LLM Client

依赖：任务 2、3、4、8。

文件：

- Create: src/underwater_tracking/domain/conversation_models.py
- Create: src/underwater_tracking/agent/nodes/conversation.py
- Modify: src/underwater_tracking/agent/nodes/directives.py
- Modify: src/underwater_tracking/agent/nodes/questions.py
- Modify: src/underwater_tracking/agent/runtime.py
- Modify: src/underwater_tracking/api/app.py
- Create: tests/agent/test_conversation.py
- Modify: src/underwater_tracking/ui/src/services/assistantApi.ts
- Create: src/underwater_tracking/ui/src/components/assistant/ConversationPanel.tsx
- Test: src/underwater_tracking/ui/src/components/assistant/ConversationPanel.test.tsx
- Modify: src/underwater_tracking/ui/src/App.tsx

步骤：

- [ ] 测试 plan_revision、evidence_query、mixed、clarification 四种分类；evidence_query 不产生事件，plan_revision 只返回 preview。
- [ ] 定义 ConversationMessage、ConversationTurnResult 和 classification schema，携带 target/region scope、evidence_ids、proposal、expected_plan_version。
- [ ] 用结构化 LLM 完成分类；evidence_query 路由到 questions.py，plan_revision 路由到 directives.py，mixed 返回两个独立结果但不自动应用，clarification 只返回一个追问。
- [ ] 后端校验证据 ID 和 expected_plan_version；旧 /api/directives、/api/questions 保留兼容。
- [ ] 在 api/app.py 增加 /api/conversation/messages 和 /api/conversation/{conversation_id}/apply。
- [ ] 前端用一个 ConversationPanel 展示角色消息、分类徽标、证据 chips、方案 diff 和显式应用按钮；不再展示两个对话卡。
- [ ] 运行：

~~~bash
conda run -n lang_py310 python -m pytest tests/agent/test_conversation.py tests/agent/test_directives.py tests/agent/test_questions.py tests/api/test_app.py -q
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui
npm test -- --run src/components/assistant/ConversationPanel.test.tsx src/services/assistantApi.test.ts
npm run build
~~~

- [ ] 提交：feat: unify expert feedback and evidence chat

---

## 10. 将回放条改为跟踪效果时间轴

依赖：任务 4、5、7、8。

文件：

- Modify: src/underwater_tracking/ui/src/components/PlaybackBar.tsx
- Modify: src/underwater_tracking/ui/src/components/BottomDrawer.tsx
- Modify: src/underwater_tracking/ui/src/App.tsx
- Test: src/underwater_tracking/ui/src/components/PlaybackBar.test.tsx

步骤：

- [ ] 测试 slider 使用 sim_time_s/sim_duration_s，主标签显示秒数，不能出现 3/3 作为主进度。
- [ ] 渲染 target maneuver、prediction revision、region activation、handoff、plan revision、degradation、expert confirmation 事件刻度。
- [ ] 显示 coverage、quality、target deviation、handoff latency 和 response revision，并标记 proxy 指标。
- [ ] playbackRate 只控制显示速度，不改变模拟物理和 planner 时间。
- [ ] 运行：

~~~bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui
npm test -- --run src/components/PlaybackBar.test.tsx
npm run build
~~~

- [ ] 提交：feat: show tracking performance on simulation timeline

---

## 11. 真实入口、浏览器和验收

依赖：任务 1-10。

文件：

- Create: tests/integration/test_regional_tracking_acceptance.py
- Create: src/underwater_tracking/ui/src/e2e/visualCommandCenterFlow.test.ts
- Modify only if needed: main.py
- Modify only if needed: README.md

步骤：

- [ ] 单目标集成 fixture 必须证明：有序无重叠区域、每区域一份 LLM 策略、LLM 成员、合法协作模式、平滑敌方机动、预测 revision、蓝方 response、区域效果和 replay。
- [ ] 执行完整回归：

~~~bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking
conda run -n lang_py310 python -m pytest -q
cd src/underwater_tracking/ui
npm test -- --run
npm run build
~~~

- [ ] 运行真实入口：

~~~bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking
conda run -n lang_py310 python main.py --steps 0 --host 0.0.0.0 --port 8020
~~~

- [ ] 浏览器检查桌面和窄屏：预测走廊聚焦、探测范围默认隐藏、UUV/USV 圆环无需点击、细粒度区域可读、知识图谱三类边、右侧三个主卡、统一 LLM Client、证据质询不改 plan version、方案修正先 preview/apply、目标移动、区域接力、无 console error 和布局重叠。
- [ ] 执行 git diff --check、git status 和 git log，提交验收测试：
  test: verify regional tracking acceptance flow
- [ ] 最终报告测试结果、环境阻塞、main.py PID、UI/API URL 和截图路径；未经过浏览器验证不得声称视觉效果通过。

---

## 依赖顺序

~~~text
baseline
  -> regional data path
  -> LLM-authoritative modes / feedback
  -> strategic-tactical graph
  -> persistence + frame contract
       -> smooth adversarial loop
       -> map scale + fine regions
            -> knowledge graph
            -> three-card sidebar
                 -> unified LLM Client
                 -> effect timeline
                      -> real main.py + browser acceptance
~~~

任务 5 和任务 6 在任务 4 完成后可以并行开发，但必须在任务 7 汇合，因为两者都依赖稳定的 regional frame revision。

## 完成定义

只有当一次真实 main.py 运行能展示单目标沿细粒度预测走廊运动、LLM 决定的区域责任和合法协作模式、平滑敌方机动、蓝方响应与接力、时间轴上的跟踪效果、可回放 revision，以及一个能够区分证据质询和方案修正的统一对话框时，总方案才算完成。
