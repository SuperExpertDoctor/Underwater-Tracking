# Adaptive Region Graph And Game Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将单目标区域跟踪仿真改造成由 LLM 自主编组、按目标预测区域接力跟踪、可视化任务效果并支持敌我动态博弈的完整闭环，同时保持 `main.py` 可以在 `lang_py310` 环境启动。

**Architecture:** 后端继续以 `TargetRegionPlan`/`RegionTask` 作为区域任务事实来源，扩展 `OperationalFrame` 输出区域、编组责任、任务效果和专家反馈状态。LLM 负责产生每个区域的成员集合与任务策略，规则层只校验实体存在性、生命周期、通信/声呐能力、空间冲突和安全边界，不补齐或限制 UUV/USV 数量。前端将区域任务映射为 SVG 知识图谱，将细分区域叠加到地图，并通过折叠卡片呈现任务效果、专家反馈和图层控制。目标模型与现有 adversary graph/engine 接口保持一致，通过有界曲率、决策冷却和蓝方快速重规划实现平滑的敌我响应循环。

**Tech Stack:** Python 3.10, Pydantic/dataclasses, LangGraph runtime, pytest, React 18, TypeScript, Vite, Vitest, HTML Canvas, SVG.

---

## Task 1: 建立区域任务与跟踪效果的前后端数据契约

**Files:**

- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/domain/ui_models.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/api/frame_builder.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/types/frames.ts`
- Create: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/tests/api/test_frame_builder_regional_views.py`
- Create: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/types/regionalTasks.test.ts`

### Step 1: Write failing backend contract tests

- [ ] 构造一个目标、三个按时间排序的区域、两艘 UUV、一艘 USV 和现有 `GroupReport` 的 fixture。
- [ ] 断言 `build_operational_frame(...)` 输出 `regional_plans`、区域前后继关系、LLM 成员集合和任务效果。
- [ ] 断言效果字段包含 `status`、`coverage_ratio`、`quality_score`、`handoff_progress` 和 `quality_source`，且当前来源明确为 `group_quality_proxy`。
- [ ] 断言空区域计划序列化为空对象且不破坏旧 frame。

~~~bash
conda run -n lang_py310 python -m pytest tests/api/test_frame_builder_regional_views.py -q
~~~

预期初始结果：因 `OperationalFrame` 没有区域任务视图而收集或断言失败。

### Step 2: Write failing TypeScript contract tests

- [ ] 在 `regionalTasks.test.ts` 中添加一个带目标、区域、成员、任务状态和效果的 frame fixture。
- [ ] 断言类型支持 `region_id`、`geometry`、`visit_windows`、`predecessor_region_ids`、`successor_region_ids`、`assigned_uuv_ids`、`assigned_usv_ids`、`group_id` 和效果字段。
- [ ] 断言旧 frame 中缺少 `regional_plans` 时仍可渲染。

~~~bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui
export PATH=/home/shuixia/miniconda3/envs/auv_env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/bin:/usr/bin
npm test -- --run src/types/regionalTasks.test.ts
~~~

预期初始结果：缺少前端区域类型而编译失败。

### Step 3: Implement explicit UI view models

- [ ] 在 `ui_models.py` 中增加 `TrackingEffectView`、`RegionTaskView` 和 `RegionalPlanView`。
- [ ] `TrackingEffectView` 使用受限状态：`planned`、`active`、`handoff_ready`、`degraded`、`uncovered`；比例统一限制在 0 到 1。
- [ ] `RegionTaskView` 包含目标、几何、时间窗口、前后继区域、成员集合、编组 ID、状态和效果。
- [ ] `OperationalFrame` 增加 `regional_plans: dict[str, RegionalPlanView] = {}`。
- [ ] 保留后端内部的 `constraints` 字段以兼容现有安全逻辑，但不再要求新 UI 显示它。

### Step 4: Build views from the existing plan and group quality

- [ ] 在 `frame_builder.py` 添加 `_build_regional_plan_views(...)`，并在 `build_operational_frame(...)` 中调用。
- [ ] 每个 `RegionTask` 映射为一个视图，保持稳定区域 ID 和时间顺序。
- [ ] 无成员为 `uncovered`；已有降级原因或硬防护原因则为 `degraded`；活动时间窗内为 `active`；满足交接阈值且有后继为 `handoff_ready`；其余为 `planned`。
- [ ] 使用现有 `GroupQualityView` 生成可解释的代理覆盖率/质量，不将代理指标伪装成区域传感器真值。

### Step 5: Run tests and commit

~~~bash
conda run -n lang_py310 python -m pytest tests/api/test_frame_builder_regional_views.py tests/api/test_frame_builder.py -q
cd src/underwater_tracking/ui
npm test -- --run src/types/regionalTasks.test.ts src/components/RightSidebar.test.tsx
~~~

- [ ] 仅提交本任务文件，提交信息为 `feat: expose regional tracking task views`。

## Task 2: Remove fixed group-size assumptions and add expert feedback directives

**Files:**

- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/domain/regional_models.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/planning/regional_allocation.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/planning/regional_validation.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/domain/agent_models.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/agent/nodes/directives.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/agent/runtime.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/agent/nodes/strategy.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/tests/planning/test_regional_allocation.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/tests/planning/test_regional_validation.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/tests/agent/test_directives.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/tests/agent/test_assignment_directives.py`

### Step 1: Add red tests for LLM-authoritative membership

- [ ] 给一个任务提供一艘 UUV 和两艘 USV，令 advisory `required_*` 为 0，断言 allocator 保留精确成员集合，不补充可用平台。
- [ ] 给一个空成员任务，断言结果为 `uncovered`，而不是自动选择默认成员。
- [ ] 继续断言未知 ID、重复 ID、不可用实体、通信/声呐冲突和区域重叠会被拒绝或降级。
- [ ] 断言没有通信硬要求的 UUV-only/USV-only 任务可以通过，不因最低数量失败。

~~~bash
conda run -n lang_py310 python -m pytest tests/planning/test_regional_allocation.py tests/planning/test_regional_validation.py -q
~~~

预期初始结果：当前 allocator 仍会按 `required_uuv_count`/`required_usv_count` 补选或报告数量不足。

### Step 2: Add red tests for expert feedback

- [ ] 扩展 directive 测试，解析 `directive_type=feedback`、目标 ID、区域 ID 和中文反馈文本。
- [ ] 断言反馈进入运行时 feedback ledger，并作为下一轮 strategy context 输入。
- [ ] 断言反馈不会直接改写 `assignment_uuv_ids`/`assignment_usv_ids`；显式 assignment 仍保留为高级人工覆盖入口。

示例：

~~~python
directive = parse_expert_directive({
    "directive_type": "feedback",
    "target_id": "target_01",
    "region_ids": ["region_2"],
    "feedback": "region_2 的交接延迟较大，请提高下一窗口的接力余量",
})
assert directive.directive_type == "feedback"
assert directive.feedback_region_ids == ("region_2",)
~~~

### Step 3: Implement advisory counts and authoritative membership

- [ ] 将 `required_uuv_count`/`required_usv_count` 改为默认 0 的 advisory plan metadata，只用于解释或 LLM 输出，不作为隐藏 UI 配额。
- [ ] allocator 以 LLM 提供的成员集合为初始且唯一的编组事实，做稳定排序、重复检查和实体诊断。
- [ ] 不填充数量；无成员为 `uncovered`，成员无效或不可用为 `degraded`。
- [ ] 保留物理、生命周期、声呐、接力、重叠和安全边界校验。
- [ ] validation 删除最低数量错误，但保留 `usv_relay_required` 必须由具备能力且可用的 USV 满足这一能力约束。

### Step 4: Implement feedback through the existing runtime

- [ ] 将 `ExpertDirective.directive_type` 扩展为 `constraint|assignment|feedback`，增加 `feedback_region_ids`、`feedback_text` 和 revision/timestamp。
- [ ] 更新 `directives.py` schema/parser；更新 `runtime.py` 存储和策略上下文，不把反馈自动提升为硬约束。
- [ ] 更新 `strategy.py` prompt/context，使 LLM 能看到区域效果、降级原因、专家反馈和预测 revision。
- [ ] LLM 改变成员时必须显式返回每个区域的 UUV/USV ID，代码不得在解析后注入固定成员数。

### Step 5: Run tests and commit

~~~bash
conda run -n lang_py310 python -m pytest tests/planning/test_regional_allocation.py tests/planning/test_regional_validation.py tests/agent/test_directives.py tests/agent/test_assignment_directives.py -q
conda run -n lang_py310 python -m pytest tests/planning tests/agent -q
~~~

- [ ] 提交信息为 `feat: let llm choose regional tracking teams`。

## Task 3: Increase adaptive region granularity and make detection range optional

**Files:**

- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/domain/regional_models.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/planning/regions.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/App.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/CanvasMap.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/CanvasMap.test.ts`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/App.css`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/tests/planning/test_regions.py`

### Step 1: Add red density tests

- [ ] 给定稳定目标预测 fixture，断言默认 adaptive plan 至少有 64 个候选细胞，并且高概率/高曲率位置比旧默认更细。
- [ ] 断言所有细胞在地图边界内、ID 稳定、前后继关系保持时间顺序。
- [ ] 断言密度配置只控制几何，不引入 UUV/USV 最低数量规则。

~~~bash
conda run -n lang_py310 python -m pytest tests/planning/test_regions.py -q
~~~

预期初始结果：旧默认 `target_grid_cells=25` 或旧走廊扩展不满足细粒度断言。

### Step 2: Implement finer adaptive subdivision

- [ ] 将 nominal target density 提高为 64，保留 cell-size 最小/最大安全边界。
- [ ] 在 `regions.py` 让高 occupancy 和高机动不确定性区域获得更小单元，低概率边缘只在维持连通接力走廊时保留。
- [ ] 每个保留细胞仍生成一个 `RegionTask`，区域 ID 和时间有向边确定性生成。
- [ ] 只有一个目标根节点；所有任务必须指向该目标，不新增多个目标。

### Step 3: Implement map layers

- [ ] 在 `CanvasMap.tsx` 增加 `showDetectionRange`，默认 false；仅在 true 时调用现有红色虚线探测范围绘制函数。
- [ ] 从 `frame.regional_plans` 绘制预测区域多边形，活动区域提高描边/透明度，并显示 `region_1`、`region_2` 标签。
- [ ] 区分目标预测走廊与探测范围的颜色/线型，区域时间箭头由区域接力层控制。
- [ ] 在 `App.tsx` 增加 `预测区域`、`区域接力`、`基础网格`、`探测范围` 状态；基础视觉网格从 8 分格提高到 16 分格。
- [ ] 保持 UUV/USV 平台环默认可见，不依赖点击选择。

### Step 4: Add map tests

- [ ] 默认选项不触发红色探测范围 stroke。
- [ ] `showDetectionRange=true` 触发探测范围绘制。
- [ ] 区域几何和标签在默认重点视图绘制。
- [ ] 区域多边形与预测走廊使用不同绘制样式。
- [ ] 切换图层不移除 UUV/USV 平台环。

~~~bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui
export PATH=/home/shuixia/miniconda3/envs/auv_env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/bin:/usr/bin
npm test -- --run src/components/CanvasMap.test.ts
conda run -n lang_py310 python -m pytest tests/planning/test_regions.py -q
~~~

- [ ] 提交信息为 `feat: render fine-grained predicted regions`。

## Task 4: Build the dynamic region/entity knowledge graph

**Files:**

- Create: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/assistant/RegionTaskGraph.tsx`
- Create: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/assistant/RegionTaskGraph.test.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/App.css`

### Step 1: Write pure layout tests

- [ ] 导出纯函数 `buildRegionGraphLayout(...)`，用四区域、三 UUV、两 USV fixture 验证方形区域节点、圆形实体节点和稳定布局。
- [ ] 验证按 `start_time_s` 排序的区域时间有向边。
- [ ] 验证 entity-to-region responsibility 边、无成员不生成边、重复成员不重复节点。
- [ ] 验证空计划和 64 区域计划均有边界约束，避免 SVG 溢出卡片。

~~~bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui
npm test -- --run src/components/assistant/RegionTaskGraph.test.tsx
~~~

预期初始结果：模块不存在而失败。

### Step 2: Implement accessible SVG graph

- [ ] 方形节点显示 `region_1` 等；圆形节点显示 `UUV_1`/`USV_1`，平台类型使用不同描边。
- [ ] 时间边带箭头 marker；责任边使用虚线。
- [ ] active/degraded/uncovered/handoff-ready 使用明确的状态样式。
- [ ] 节点和边包含 `aria-label`/title；点击区域高亮相关实体，点击实体高亮其负责区域。
- [ ] 组件只消费 `OperationalFrame.regional_plans`，旧 frame 无区域计划时显示空状态。

### Step 3: Add interaction tests and commit

- [ ] 测试 SVG 中的方形/圆形 class、箭头 marker、区域标签和效果状态属性。
- [ ] 测试点击 `region_2` 调用选择回调。
- [ ] 测试大图可滚动且不撑破右侧卡片。

提交信息：`feat: visualize regional handoff knowledge graph`。

## Task 5: Convert the right sidebar to collapsible cards and surface expert feedback

**Files:**

- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/RightSidebar.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/assistant/AssignmentPanel.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/assistant/DirectiveComposer.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/App.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/App.css`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/RightSidebar.test.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/assistant/AssignmentPanel.test.tsx`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/components/assistant/DirectiveComposer.test.tsx`

### Step 1: Add failing sidebar behavior tests

- [ ] 每个主要面板使用可访问的 `details/summary` 或等价的 `aria-expanded` 控件。
- [ ] 图谱、状态、实体列表、反馈卡可以独立打开和关闭。
- [ ] 用户侧不再出现“方案约束”或 `scheme.constraints` 内容。
- [ ] 图谱卡展示当前目标区域计划，效果卡展示 coverage、quality、handoff、降级原因和 proxy/measured 来源。

### Step 2: Implement collapsible cards

- [ ] 使用原生 `details/summary` 优先，保留键盘和读屏行为。
- [ ] 卡片保持紧凑、滚动稳定、圆角不超过现有设计规范；不在卡片内嵌套页面级卡片。
- [ ] 卡片包括目标预测/仿真状态、区域图谱、任务效果、UUV/USV roster、敌方状态、专家反馈。
- [ ] 删除 constraints 的可见渲染，保留后端安全校验。

### Step 3: Replace fixed-group manual assignment controls

- [ ] `AssignmentPanel.tsx` 改为只读区域任务/效果摘要，展示 LLM 选定成员和每个区域的状态。
- [ ] `DirectiveComposer.tsx` 支持以目标和区域为范围提交 `feedback`，保留 preview/apply 的响应展示。
- [ ] 不在界面中暗示必须选择固定数量；显式 assignment 仅作为高级人工覆盖。
- [ ] 图谱区域选择与地图区域选择双向高亮。

### Step 4: Run UI tests/build and commit

~~~bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui
npm test -- --run src/components/RightSidebar.test.tsx src/components/assistant/AssignmentPanel.test.tsx src/components/assistant/DirectiveComposer.test.tsx src/components/assistant/RegionTaskGraph.test.tsx
npm run build
~~~

提交信息：`feat: add collapsible regional tracking controls`。

## Task 6: Add smooth adversarial target behavior and fast blue-side replanning

**Files:**

- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/simulation/target.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/simulation/engine.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/agent/nodes/adversary.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/agent/nodes/strategy.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/tests/simulation/test_target.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/tests/simulation/test_engine.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/tests/agent/test_adversary_graph.py`
- Modify: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/tests/agent/test_runtime_master_slave_adversary.py`

### Step 1: Add failing smoothness tests

- [ ] 验证位置连续、速度/航向变化不超过加速度和转向率限制。
- [ ] 验证规避机动有界曲率，不出现瞬时航向跳变。
- [ ] 验证 escape/reacquisition 会改变 intent 和 prediction revision。
- [ ] 验证目标使用蓝方观测/对抗决策历史。

~~~bash
conda run -n lang_py310 python -m pytest tests/simulation/test_target.py tests/simulation/test_engine.py -q
~~~

### Step 2: Implement bounded maneuver primitives

- [ ] 在 `target.py` 增加 desired heading/speed、turn-rate、acceleration、expiry 的机动状态。
- [ ] 每步插值航向和速度，复用现有目标运动学和 `apply_evasive_maneuver(...)` 公共接口。
- [ ] 保证 RNG 下可复现，不破坏现有 target fixture。

### Step 3: Implement adversary hysteresis and blue rapid response

- [ ] 在 `engine.py`/现有 adversary graph 增加决策冷却和预测/意图/质量 revision 的显著变化阈值。
- [ ] 目标发生机动或区域质量下降时，触发蓝方区域计划快速 revision，并通过现有 runtime 安全门。
- [ ] 保留决策历史到 `OperationalFrame`，让 UI 可以展示响应原因。
- [ ] 敌方在 escape、增加不确定性、侧翼变化、短时静默间切换；蓝方更新预测区域和 LLM 成员集合，不写死团队数量。

### Step 4: Add event-chain tests

- [ ] 验证 `target maneuver -> prediction revision -> regional task revision -> group effect change -> blue replanning event`。
- [ ] 事件包含 target ID、旧/新 revision、受影响区域、原因和延迟。
- [ ] 专家反馈进入下一轮策略上下文；安全失败只产生 degraded/uncovered，不产生不安全分配。

~~~bash
conda run -n lang_py310 python -m pytest tests/simulation tests/agent -q
~~~

提交信息：`feat: model smooth adversarial maneuver response`。

## Task 7: End-to-end runtime, replay and browser verification

**Files:**

- Modify only if needed: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/main.py`
- Modify only if needed: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/services/assistantApi.ts`
- Create: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/tests/integration/test_main_runtime.py`
- Create: `/home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui/src/e2e/regionalTrackingFlow.test.ts`

### Step 1: Add runtime smoke test

- [ ] 以项目现有 headless/短时参数启动真实 `main.py` 子进程。
- [ ] 断言退出码为 0、无导入/LangGraph 错误，并至少产生含 `target_01` 区域计划的 `OperationalFrame`。
- [ ] 不从测试中留下第二个长驻服务。

~~~bash
conda run -n lang_py310 python -m pytest tests/integration/test_main_runtime.py -q
~~~

### Step 2: Add browser flow assertions

Against the existing Vite server on port 5184, verify:

- [ ] 初始探测范围隐藏，预测区域/区域接力显示。
- [ ] 底部回放条是仿真秒数时间轴，不再显示 `3/3` 帧计数。
- [ ] UUV/USV 平台环无需点击即可显示。
- [ ] 图谱包含方形区域、圆形实体、时间箭头和责任虚线。
- [ ] 右侧卡片可独立折叠，效果卡显示代理质量、覆盖和交接状态。
- [ ] 提交专家反馈后出现确认，不强制成员数量。
- [ ] 目标机动后地图/图谱 revision 更新，并出现蓝方响应事件。

~~~bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui
npm test -- --run src/e2e/regionalTrackingFlow.test.ts
npm run build
~~~

### Step 3: Run the real entry point

必须运行项目真实的 `main.py`，使用 `lang_py310`：

~~~bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking
conda run -n lang_py310 python main.py
~~~

使用项目已有 bind/port 配置。若 `main.py` 只启动 API，UI 使用远端 Node 22 并绑定 5184，不能停止无关服务：

~~~bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking/src/underwater_tracking/ui
export PATH=/home/shuixia/miniconda3/envs/auv_env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/bin:/usr/bin
npm run dev -- --host 0.0.0.0 --port 5184
~~~

用浏览器检查桌面和窄窗口：无未捕获 console error、无失败 frame/API 请求、初始地图焦点正确、图谱和效果卡存在、探测范围可选显示。

### Step 4: Full verification and handoff

~~~bash
cd /home/shuixia/users/houguoqiang/projects/Underwater-Tracking
conda run -n lang_py310 python -m pytest -q
cd src/underwater_tracking/ui
npm test -- --run
npm run build
git diff --check
git status --short
~~~

- [ ] 检查并保留工作区中已有的回放时间轴修复，不回滚或混入无关修改。
- [ ] 最终报告准确的 commit、测试结果、`main.py` 进程状态、浏览器 URL 和截图路径。
- [ ] 外部依赖/端口导致无法执行时，报告精确命令和错误，不将其标记为通过。


