# 全局轨迹驱动的 IMM 四区域 UUV 执行闭环设计

日期：2026-08-26

状态：设计已确认，待实施计划审阅

范围：UUV-only 单目标场景的全局目标轨迹、IMM 预测、意图理解、四区域动态划分、八艇任务组、边界进出、LLM 增量优化、实时数据链、地图呈现、证据记忆、运行目录和进程生命周期。

## 1. 背景

当前系统已经具备 IMM/UIF 估计、轨迹预测、区域规划、MissionController、后台 LLM、分层记忆、实时操作帧和 React 地图，但完整运行暴露出以下结构性问题：

1. 启动骨架、LLM 规划和物理执行使用不同状态来源，LLM 延迟或输出非法时容易留下过期方案。
2. IMM 滤波已接入，但未来中心线主要由 B 样条或线性外推产生，IMM 模型概率只作为元数据展示。
3. UUV-only 场景中的目标轨迹已经全局可知，预测端却仍优先使用融合信念历史。
4. 四个执行任务区域不会稳定地随最新预测滚动，长时间运行后区域可能停留在初始位置。
5. 实时帧同时暴露底层候选网格和执行任务区域，前端按不同 ID 合并后产生几十个重叠多边形。
6. 多个区域按 target_id 共用航点历史，区域间会相互覆盖路径状态。
7. 旧航母、母舰、投放、回收和返航语义仍残留在运行测试、帧模型和进程链中。
8. 固定测试帧能够渲染理想效果，但真实后端帧的数据粒度、任务版本和区域状态并不等价。
9. HTTP、WebSocket、Replay、助手和记忆分别读取相关但不完全一致的状态，存在混合 revision 的风险。
10. 一次运行可能创建多个 run-* 或 serve-* 目录，并可能残留 UI 或工作线程进程。

本设计用一个权威执行快照统一这些边界。

## 2. 已确认决策

### 2.1 总体方案

采用“统一执行快照 + 可运行的确定性滚动骨架 + LLM 增量优化”。

确定性算法负责持续产生合法、可执行的目标预测、四区域链、UUV 分配和航点。LLM 在后台解释意图并优化策略，但不能阻塞物理执行，也不能直接修改航点或当前任务。

### 2.2 目标信息边界

UUV-only 场景中，目标潜艇的当前位置和历史运动轨迹对蓝方全局可知。目标未来动作和隐藏任务意图仍未知。

蓝方允许使用全局轨迹进行预测、区域划分和任务调度，但不得读取目标侧尚未执行的 LLM 决策或隐藏意图。

### 2.3 任务区域与资源

每个目标始终维护四个有序任务区域。每个区域配置一个两艇 task group，共八艘同时执行：

| task group | 区域 | 默认成员 | 主要职责 |
|---|---|---:|---|
| TG-01 | R01 | 2 | 当前跟踪或区域扫描 |
| TG-02 | R02 | 2 | 交接准备或区域扫描 |
| TG-03 | R03 | 2 | 预测路径预置监视 |
| TG-04 | R04 | 2 | 预测路径预置监视 |
| Reserve | 替补池 | 4 | 故障、能量和质量降级替换 |

每个 task group 优先包含一艘主动验证艇和一艘被动跟踪艇。

### 2.4 平台生命周期

UUV-only 执行不再包含航母、母舰、舰船投放、回收、返航或后勤路线。

不可用 UUV 驶向所属任务区域边界并消失；替补 UUV 从同一区域边界驶入并接替角色。

### 2.5 兼容边界

旧 TrackingPlan、航母字段和旧事件只允许用于历史回放读取或非 UUV-only 场景。它们不得重新进入新的 UUV-only 物理执行链，也不得出现在新的 UUV-only 地图主视图。

## 3. 方案比较

### 3.1 统一执行快照

这是选定方案。预测、意图、区域、task group、版本和证据在同一对象中原子发布。确定性算法保持可运行性，LLM 只提交经过验证的增量优化。

优点是能够根治混合版本、区域双重粒度、LLM 阻塞和真实帧与固定帧不一致。代价是需要调整领域模型、运行协调器、发布器和前端 store。

### 3.2 保留双计划并在前端过滤

只过滤底层网格并补充后台刷新，修改量较小，但 TrackingPlan、ExecutableMissionPlan、候选区域和执行区域仍可产生不同 revision。该方案无法解决根因。

### 3.3 全面事件溯源重建

所有当前状态都从事件流重建，一致性最强，但会扩大到全部运行时、记忆和回放基础设施，不符合当前范围。

## 4. 权威执行快照

新增 OperationalExecutionSnapshot，作为 UUV-only 场景唯一可进入 MissionController 和实时地图的执行契约。

### 4.1 顶层字段

- scenario_id
- target_id
- execution_revision
- source_snapshot_revision
- source_sim_time_s
- prediction_revision
- prediction_id
- intent_revision
- expert_request_version
- generated_at_s
- valid_from_s
- valid_until_s
- plan_source
- target_track
- prediction
- intent
- regions
- task_groups
- reserve_uuvs
- current_region_id
- next_region_id
- evidence_ids
- degradation

plan_source 只能是 deterministic、llm_optimized 或 human_revised。

### 4.2 版本不变量

1. execution_revision 严格递增。
2. 四个区域、四个 task group 和 MissionController 投影必须使用同一 execution_revision。
3. 四个区域必须引用同一 prediction_id。
4. source_snapshot_revision 不能晚于提交时的物理 revision。
5. current_region_id 和 next_region_id 必须存在于区域集合。
6. 同一 UUV 不能属于多个 task group 或同时属于执行组和替补池。
7. 所有 evidence_ids 必须能从持久化仓库解析。

### 4.3 执行边界

MissionController 只接受由 OperationalExecutionSnapshot 派生的 ExecutableMissionPlan。TrackingPlan 只是审计投影。

计划提交采用 compare-and-set。候选计划的 base_execution_revision 必须等于当前 execution_revision。过期候选只记录审计，不进入物理执行。

## 5. 全局目标轨迹

新增 GlobalTargetTrack：

- target_id
- track_revision
- sim_time_s
- position_xy
- velocity_xy
- heading_rad
- acceleration_xy
- turn_rate_rad_s
- bounded_history
- source_event_ids
- freshness_status

SimulationEngine 在每个物理步或配置采样边界更新全局轨迹。历史长度必须有明确上限，但必须覆盖 IMM 初始化和 1800 秒预测所需窗口。

所有 UUV-only 预测读取 GlobalTargetTrack。融合 UUV 观测继续产生跟踪质量、探测效果和博弈证据，但不作为“是否知道目标位置”的前置条件。

## 6. 完整 IMM 预测

### 6.1 模型状态

IMM 对 CV、CT_LEFT 和 CT_RIGHT 三个模型公开：

- state_mean
- state_covariance
- model_probability
- innovation
- likelihood
- source_observation_ids

不能只公开混合均值和概率。

### 6.2 多模型传播

三个模型分别传播到 1800 秒预测时域。每个采样点先得到各模型的二维位置均值与协方差，再按模型概率做矩匹配：

混合均值为各模型均值的概率加权和。

混合协方差同时包含模型内部协方差和模型均值相对混合均值的离散项。

### 6.3 输出契约

IMMPredictedTrack 包含：

- prediction_id
- prediction_revision
- target_id
- origin_sim_time_s
- times_s
- centerline_xy
- covariance_xy
- corridor_radius_m
- model_branches
- model_probabilities
- clipping_records
- source_track_revision
- source_observation_ids
- prediction_regime

prediction_regime 在正常路径中为 imm。IMM 状态不完整时回退 bspline；历史不足时回退 short_history。

### 6.4 预测差异

连续预测按相同绝对仿真时间对齐，计算：

- absolute_rms_m
- normalized_rms
- p90_displacement_m
- maximum_displacement_m
- maximum_displacement_time_s
- model_probability_js_distance
- leading_model_changed

显著预测变化立即触发确定性区域滚动，同时进入意图复核。区域滚动不等待 LLM 意图确认。

## 7. 意图理解

### 7.1 确定性意图

从全局轨迹和 IMM 状态提取：

- 平均速度
- 速度变化和冲刺
- 加速度
- 曲率
- 持续转向方向和时长
- 往返摆动
- 驻留半径和驻留时长
- 相对地图边界的接近或远离
- IMM 主模型和模型概率变化

输出标签限定为 transit、patrol、loiter、evade、approach、withdraw 或 unknown。

每个结论记录特征值、阈值、证据 ID 和确定性规则版本。进入和退出使用不同阈值，并要求连续周期确认。

### 7.2 LLM 修订

LLM 接收确定性意图、IMM 分支、预测差异和有界证据。只有标签合法、证据可解析、置信度达标、领先幅度达标、连续两次一致且仍对应当前预测版本时，才能形成已确认修订。

LLM 失败、超时、低置信度或输出过期时，保留确定性意图。

## 8. 四区域动态划分

### 8.1 时间窗

默认 1800 秒时域按以下窗口划分：

| 区域 | 默认时间窗 |
|---|---|
| R01 | 0 至 540 秒 |
| R02 | 450 至 990 秒 |
| R03 | 900 至 1440 秒 |
| R04 | 1350 至 1800 秒 |

相邻区域保留 90 秒交接重叠。具体窗口允许在受限范围内根据目标速度和 LLM 建议调整。

### 8.2 几何生成

区域沿 IMM 中心线的时间段和弧长生成，不再使用固定主轴大矩形。区域宽度由逐点协方差、UUV 机动余量和最小任务宽度共同决定。

区域必须：

1. 包含对应中心线样本。
2. 与预测时间顺序一致。
3. 只允许相邻区域受控重叠。
4. 非相邻区域不得重叠。
5. 裁剪到地图边界后仍满足最小宽度和面积。
6. 拥有完整前驱、后继和交接时间窗。
7. 始终恰好四个。

区域 ID 稳定为 target_id:task:01 至 target_id:task:04。滚动更新 geometry_revision，不堆积新 ID。

### 8.3 滚动触发

默认每 450 仿真秒检查，并在以下事件立即检查：

- 当前区域窗即将结束。
- 预测中心线离开当前区域链。
- IMM 主模型显著改变。
- 确定性意图改变。
- 目标冲刺或持续转向。
- 当前 task group 质量下降。
- UUV 失效或进入替换阈值。
- 人工反馈改变约束。

新方案校验成功后原子替换。失败时继续使用当前方案。

## 9. task group 与 UUV 机动

### 9.1 角色

每组两艇，优先分配一艘 active_verifier 和一艘 passive_tracker。资源不足时不得伪造成员，必须显示明确降级原因。

### 9.2 区域生命周期

PLANNED

PREPOSITIONING

ACTIVE_SCAN

PASSIVE_TRACK

HANDOFF_PENDING

HANDOFF_COMPLETED

MONITORING_COMPLETE

交接要求下一组两艇可用、当前周期存在有效观测、目标进入相邻重叠区、前后区域版本一致且质量达到硬阈值。

### 9.3 航点策略

- 当前区域组围绕全局已知目标建立受区域约束的测向几何。
- 下一交接组靠近预测进入点形成交接队形。
- 未来区域组执行稀疏蛇形或扇区覆盖。
- 航点始终位于所属区域内。
- 路径历史按 task_group_id 和 region_id 隔离。
- 每次更新限制最大位移、最大转向变化和最小艇间距。

### 9.4 边界替换

不可用 UUV 驶向所属区域最近边界，逐步降低透明度，到达后从空间集合消失。

替补 UUV 从同一区域适合的边界点出现，逐步增加透明度，驶向任务槽位并在形成有效观测后完成角色接管。

替换过程保留完整资源 episode、任务 revision 和事件证据。

## 10. LLM 增量优化

### 10.1 允许输出

LLM 只输出受限语义策略：

- 区域优先级
- 时间窗比例
- 宽度和交接重叠建议
- 声呐策略
- task group 角色建议
- 替补优先级
- 意图解释
- 方案保持或修订建议
- 理由和证据引用

LLM 不输出任意多边形、任意区域 ID、UUV 物理航点或未经允许的 UUV ID。

### 10.2 后台流程

当前执行快照被捕获为不可变输入。后台依次执行 LLM 分析、结构校验、证据校验、策略边界校验、确定性几何规范化、资源优化、物理可行性校验和 compare-and-set 提交。

物理线程、帧发布、助手查询和记忆流在 LLM 运行期间继续工作。

### 10.3 失败状态

规划健康状态至少包括 running、committed、invalid_output、provider_timeout、provider_unavailable、stale、resource_conflict、geometry_rejected 和 preserving_active_plan。

失败必须记录操作、模型、prompt 版本、请求响应摘要哈希、基础执行版本、失败字段、保留版本和重试条件。

失败不得清空区域、清空 task group、重置 plan_version、暂停物理线程或阻塞 API。

## 11. 操作帧与数据链

### 11.1 UUV-only 帧

OperationalFrame 增加 execution 投影，至少包含：

- execution_revision
- source_snapshot_revision
- prediction_revision
- intent_revision
- data_age_s
- data_status
- plan_source
- current_region_id
- next_region_id
- evidence_ids

实时帧只包含一个目标预测、一个意图、四个 regional_missions、四个 task_groups、八艘空间执行 UUV 和四艘非空间替补资源。

新 UUV-only 帧不得包含航母、母舰、舰船路线、投放回收任务、候选网格区域或过期区域。

### 11.2 原子发布

帧构建一次读取当前物理快照、当前 OperationalExecutionSnapshot、MissionController 快照、预测意图和审计事件尾部。

同一个完整帧对象发布到 OperationalHub、HTTP snapshot、WebSocket、JSONL 和 Replay。各通道不得重新拼装不同版本的数据。

### 11.3 一致性校验

发布前验证：

- frame_id 单调递增
- sim_time_s 单调不减
- execution revision 一致
- prediction ID 一致
- 四区域几何有效
- task group 引用可解析
- UUV 分配互斥
- 区域拓扑完整
- 证据 ID 可解析

如果语义状态暂时不一致，则发布最新物理状态和上一有效执行快照，并标记 degraded，不能发布半成品组合。

### 11.4 WebSocket

前端以 frame_id 为主序列。旧帧丢弃，帧跳跃时使用 HTTP snapshot 补偿，重连后先取完整 snapshot，再继续 WebSocket。页面卸载后必须回收接收、发送和心跳任务。

## 12. 地图与任务流

### 12.1 主地图

默认显示：

1. IMM 混合中心线。
2. 随时间变化的置信走廊。
3. 严格四个任务区域。
4. R01 到 R04 的交接箭头。
5. 当前区域和下一交接区域高亮。
6. 四个 task group。
7. 八艘空间执行 UUV。
8. 当前目标区域两艇的完整标签。
9. 其他区域的简洁组标识。
10. 短尾迹。

默认不显示候选网格、完整历史航线、航母舰船、回收连线或非空间替补 UUV。

### 12.2 相机与标签

默认相机包围目标、预测走廊、四区域和八艘执行 UUV，不使用目标探测范围扩大视野。

标签按缩放级别分层，并执行碰撞避让。固定格式元素使用稳定尺寸，动态文本不得挤压地图布局。

### 12.3 时间线

区域时间线只显示四行，包含区域、时间窗、task group、状态、质量和交接。底层候选只在审计详情中查询。

地图选择区域、task group 和时间线必须双向联动。

## 13. 助手、反馈与记忆

### 13.1 数据链

以下接口共享 execution_revision 和 frame_id：

- conversation messages
- assistant memory
- memory stream
- questions
- directives
- assignments
- sensor modes

证据查询只读。方案建议必须经人工确认后进入规划 mailbox。

### 13.2 “为何这样制定方案”

回答必须说明：

1. 当前执行目标。
2. 当前轨迹和 IMM 判断。
3. 当前意图判断。
4. 四区域划分理由。
5. 当前区域 task group 分配理由。
6. 其他组的当前职责。
7. 最近调整原因。
8. 确定性算法、LLM 和人工反馈各自贡献。
9. 可解析证据 ID。

缺失证据时明确返回缺失项，不生成推测性理由。

### 13.3 分层记忆

短期记忆保存会话、当前执行版本和最近事件。长期记忆保存目标机动模式、历史区域效果、交接成败、替换经验、人工偏好和已验证策略经验。

记忆后台失败不能影响执行。LLM 引用记忆时必须同时提供 memory_id 和原始证据 ID。

## 14. 双方博弈闭环

目标侧规则或 LLM 可以改变速度、航向、深度和规避策略。蓝方只根据已经进入全局轨迹的物理变化响应：

目标决策 -> 目标物理运动 -> 全局轨迹 -> IMM 预测 -> 意图与区域滚动 -> task group 机动 -> 跟踪和暴露效果 -> 目标下一次决策。

蓝方不得读取目标侧尚未执行的决策或隐藏意图。

## 15. 运行目录与进程生命周期

### 15.1 单目录

一次 main.py 调用只创建 outputs/run-<run_id>。

目录统一保存 manifest、agent.db、operational_frames、事件、日志、验收报告、证据链、预测快照、执行版本和桌面移动端截图。

新运行不创建 serve-*。

### 15.2 正式服务

正式前端构建为静态资源并由同一 FastAPI 服务提供。正式运行不启动独立 Vite 开发服务器。

开发 Vite 不创建运行目录、不启动第二个仿真，只连接已有 API。

### 15.3 关闭顺序

RunController 是唯一所有者。关闭时依次停止新请求、物理步进、LLM 工作、记忆工作、发布最终帧、写报告、关闭 WebSocket 和 API、关闭数据库文件，最后检查线程、子进程和端口。

超时必须写入 process-shutdown.json，不得静默残留进程。

## 16. 八分钟验收

正式验收运行 480 秒墙钟时间，默认 60 倍速度，对应 28800 秒仿真时间。

### 16.1 阶段

- 0 至 30 秒：启动、预测、四区域和八艇边界进入。
- 30 至 120 秒：当前区域跟踪、其他区域预置、助手反馈。
- 120 至 240 秒：目标机动、IMM 变化、区域滚动。
- 240 至 360 秒：区域交接和 UUV 替换。
- 360 至 450 秒：记忆提炼、证据问答和人工反馈。
- 450 至 480 秒：截图、回放一致性和关闭检查。

### 16.2 通过条件

- frame_id 持续递增。
- final sim_time_s 至少 28800。
- 始终存在有效执行版本。
- 地图始终只有四个任务区域。
- 正常情况下存在八艘执行 UUV。
- 每个区域恰好一个两艇 task group。
- 当前区域组持续跟踪目标。
- 至少一次区域滚动 revision。
- 至少一次区域交接。
- 至少一次目标机动和蓝方响应。
- IMM 中心线随目标运动改变。
- LLM 区域失败时 UUV 仍运动。
- 不出现航母投放回收事件。
- 助手反馈成功。
- 记忆可查询。
- 证据问答链完整。
- HTTP、WebSocket 和 Replay 版本一致。
- 桌面和移动端地图无空白、无重叠、无溢出。
- 本次调用只新增一个 run-*。
- 退出后无残留进程。

## 17. 测试策略

按以下顺序执行：

1. 领域模型和版本不变量。
2. GlobalTargetTrack 和历史 retention。
3. IMM 多模型传播和矩匹配。
4. 确定性意图特征、滞回和 LLM 修订。
5. 四区域几何属性和规范化。
6. 四个两艇 task group 分配。
7. 区域级航点缓存和边界替换。
8. LLM 超时、非法、过期和资源冲突。
9. 帧原子投影和多通道一致性。
10. 前端 reducer、地图和时间线。
11. 助手、记忆和证据查询。
12. 固定帧视觉基准。
13. 真实后端 Playwright。
14. 单运行目录和关闭检查。
15. 八分钟完整验收。

固定 mock 截图只能验证绘制规则，不能替代真实后端验收。

## 18. 迁移策略

1. 先增加新领域契约和只读投影，不改变物理执行。
2. 建立 GlobalTargetTrack 和完整 IMM 预测。
3. 建立确定性意图、四区域和 task group 纯函数。
4. 引入 ExecutionCoordinator，并让 MissionController 接受新快照。
5. 切换发布器和前端到权威执行投影。
6. 切换边界进出执行并停用 UUV-only 航母路径。
7. 更新助手、记忆和证据链。
8. 统一正式服务、运行目录和关闭流程。
9. 删除或隔离不再可达的旧 UUV-only 路径。

每一步必须保持旧回放可读取，并在下一步前通过定向测试。

## 19. 替代和冲突声明

本设计在 UUV-only 当前场景中替代以下旧约束：

- “母舰负责投放和回收”的执行约束。
- “蓝方规划不得使用全局目标轨迹”的约束。
- “B 样条是正常预测主路径”的约束。
- “只部署当前和下一任务批次”的资源约束。
- “候选网格可以与执行区域共同显示”的隐式行为。

旧文档仍可解释历史代码和回放，但不得作为新实现的验收依据。

## 20. 非目标

- 不重写目标侧物理模型。
- 不把目标隐藏意图直接暴露给蓝方。
- 不引入学习型轨迹分类器。
- 不为多目标场景设计跨目标资源拍卖。
- 不删除旧回放读取能力。
- 不在本阶段重构与 UUV-only 无关的非 UUV 平台算法。

## 21. LLM availability addendum (2026-08-27)

本节覆盖本文中与“LLM 可选或失败后继续执行”相冲突的旧表述：

- 正式 `agent-run`、`serve` 和 `main.py` 必须构造三个真实的 `RoleHTTPStructuredLLM`，分别对应 `master`、`slave` 和 `adversary`；禁止用 unavailable、mock 或确定性客户端替代真实角色。
- 认证信息从 `configs/.env` 的 `UNDERWATER_TRACKING_API_KEY` 读取。LongCat 使用 OpenAI 兼容根地址 `https://api.longcat.chat/openai/v1`，客户端向其 `/chat/completions` 发起结构化请求。
- 正式运行启动前必须对三个角色完成真实结构化探针，并在首个 LLM 规划提交成功前禁止推进物理时间。
- LLM 配置错误、连接中断、超时、限流或服务端错误在传输重试耗尽后均为终止性错误：必须抛出 `LLMError`，运行阶段置为 `failed`，停止物理步进，不得切换确定性基线、静默保留并继续或自动重连。
- 语义输出错误同样不得在正式首个规划阶段启动确定性算法；只有测试替身可以通过显式注入参与单元测试，不能改变正式入口的真实 LLM 门禁。
