# 自适应目标稳定跟踪闭环设计

## 目标

让运行中的 LangGraph 具备可验证的“预案 + 战场情报 + 资源/能力状态 -> 方案 -> 监控 -> 动态调整”闭环。预案和技侦情报是显式、可追溯的运行输入；UUV 的监视能力进入观测和编组优化；仿真产生的事件必须进入图的分级路由；周期复盘和低能量换班必须在生产适配器中触发；质量门槛必须影响分组和提交校验。

本轮不实现正式 Monte Carlo/evaluation runner；那是审计中明确分离的下一阶段范围。

## 方案

### 输入契约

- `OperationalScheme` 表示预定作战方案，包含版本、目标优先级、最低质量、有效期和可解释约束。它既可从场景 YAML 加载，也可通过运行时 API 更新。
- `IntelligenceReport` 表示经过来源标注的外部情报，包含来源类型（技侦/信号/电子/人工/声纳）、置信度、有效期、目标和结构化判断。它只描述观测或判断，不注入目标真实状态。
- `SurveillanceCapability` 随 `UUVState` 传播，至少覆盖被动/主动探测距离、方位方差、主动声纳可用性、最大速度和转向能力；缺省值保持旧场景兼容。
- `SituationSnapshot` 同时携带当前预案和仍在有效期内的情报，成为 LangGraph 的唯一规划快照输入；operational frame 以摘要形式展示预案、情报来源和能力，不携带 truth。

### 闭环和事件流

```text
YAML/API 预案或情报
          ↓
SimulationEngine -> SituationSnapshot -> CarrierRuntime.submit_events
          ↓                         ↓
     观测/质量/FIM             EventMonitor 分级
          ↓              strategic / tactical / informational
          └────────────── LangGraph snapshot -> strategy -> optimize -> commit
                                                        ↓
                                           PlanCommand / rotation / sensor mode
                                                        ↓
                                                  SimulationEngine
```

运行时保留事件原始 `event_id`，去重后再交给 EventMonitor；未知的生命周期、报告发布和质量 guard 事件有明确等级，不会因未分类而中断循环。适配器按照 `strategic_review_s` 产生周期复盘事件，并对低于轮换阈值的已部署成员产生带目标和能量信息的 `battery_rotation` 事件。

### 优化与安全边界

- 可行性按每艘 UUV 的能力矩阵计算，而不是使用全局单一传感器能力。
- 预案最低质量是硬下限；LLM proposal 只能提高，不能降低它。分组大小、候选排序和独立 commit 校验都使用目标级质量门槛。
- 换班只标记实际低能量成员，并优先用满足距离、能量和能力约束的健康备用 UUV 替代；没有备用资源时保留降级方案并记录问题。
- 策略提示词只消费估计、来源标注情报、预案摘要、资源/能力/质量因素和专家约束；不允许生成成员或航路，最终编组和航路仍由确定性优化器生成。

### 接口与展示

- `POST /api/intelligence` 接受结构化情报，排入下一轮快照。
- `PUT /api/operational-scheme` 更新预案并排入 strategic 事件；当前方案不会被 HTTP 直接替换，必须经过下一轮图提交。
- operational frame 增加预案摘要、有效情报摘要和能力统计；右侧态势栏显示方案约束/情报计数，底部事件时间线显示复盘与换班事件。

## 错误处理

- 过期或置信度越界的情报在入口拒绝；有效期到期后不进入规划摘要。
- 预案版本和目标键经过 Pydantic 校验；更新失败不改变当前已提交方案。
- 事件转发采用原始 ID 去重；某一事件分类失败会被记录为 carrier error，并由既有图错误节点结束当前周期，不修改活动方案。
- 质量门槛无法满足时不放宽硬约束，生成 `degraded`/rejected 结果并在验证问题与台账中留下依据。

## 测试策略

- 领域/配置测试覆盖默认兼容、输入校验、情报过期、能力序列化和 frame truth-isolation。
- 运行时回归测试先证明 engine 事件能够抵达 ActiveVerification/EventMonitor，再证明周期复盘与低能量换班事件触发。
- 优化器测试覆盖异构距离/方差/主动能力、预案最低质量、只旋转低能量成员和备用替代。
- API/UI 测试覆盖情报与预案的异步输入、帧连续刷新、事件展示和旧 JSONL 帧兼容。
- 完成后运行 Python（`lang_py310` 可用于执行；若项目环境实际 Python floor 不兼容则记录）、Ruff、Mypy、Vitest、build 和 Playwright。
