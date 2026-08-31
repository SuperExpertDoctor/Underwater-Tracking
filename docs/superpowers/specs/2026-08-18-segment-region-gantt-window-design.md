# 分段区域跟踪甘特图设计

## 1. 目标

在现有水下跟踪指挥台中增加“分段跟踪”可视化小窗口，以区域泳道甘特图展示意图理解和轨迹预测产生的未来行动区域，以及 UUV/USV 在时间和空间上的覆盖、接力、通信和降级逻辑。

本功能必须满足：

- `main.py` 仍是唯一启动入口，并能同时启动后端和 React/Vite 界面。
- 实时模式随 WebSocket 操作帧更新。
- 回放模式随 PlaybackBar 拖动同步更新。
- 时间轴采用相对时间，当前操作帧为 `T+0`。
- 区域任务是唯一事实来源，旧的目标组和分段字段只作为兼容视图。
- UI 只展示估计、预测、计划和平台状态，不引入目标真值。

## 2. 已确认的交互决定

### 2.1 主视角

采用“区域泳道”布局，而不是“平台泳道”。每行代表一个预测区域，区域内部展示负责该区域的 UUV/USV 角色和 relay 链路。这种布局优先回答“每个预测区域如何完成接力覆盖”。

### 2.2 时间轴

时间轴采用相对当前帧的偏移：

- `T+0` 表示当前 `sim_time_s`。
- 右侧表示未来预测窗口。
- 后端保留绝对仿真时间，并为前端计算 `start_offset_s` 和 `end_offset_s`。
- 已经开始的任务可以从负偏移开始，但 UI 将当前线左侧的历史部分裁剪到可视窗口边界。

### 2.3 运行模式

- 实时：收到新操作帧后按 `region_id` 保留选择状态并更新条形、当前时间线和平台状态。
- 回放：每次选中历史帧时，使用该帧自己的 `sim_time_s` 重算 `T+0`。
- 无区域数据：保留现有地图、连接、方案和事件视图，“分段跟踪”显示空状态，不影响其它界面。

### 2.4 交互窗口

不启动独立桌面进程；在现有 BottomDrawer 中增加“分段跟踪”标签。抽屉保持可拖拽高度和移动端响应式布局。点击区域后在同一抽屉内展示区域详情，避免新增第二套窗口生命周期和数据连接。

## 3. 数据契约

### 3.1 操作帧新增字段

在 `OperationalFrame` 增加可选字段：

```text
region_timeline: RegionTimelineView[]
```

缺省值为空数组，旧 frame JSON 和旧前端仍然可反序列化。

### 3.2 区域条目

`RegionTimelineView` 至少包含：

- `region_id`、`target_id`
- `center` 和方形 `bounds`
- `start_offset_s`、`end_offset_s`
- `status`: planned / active / handed_off / degraded / uncovered
- `coverage_mode`
- `priority`、`occupancy_likelihood`
- `uuv_assignments`
- `usv_assignments`
- `communication_links`
- `handoff_from`、`handoff_to`
- `sonar_mode`
- `evidence_ids`
- `degraded_reasons`
- `plan_revision`

UUV/USV assignment view 至少包含平台 ID、平台类型、区域角色和本区域内的相对起止偏移。通信链路 view 包含源 ID、目标 ID、介质、状态和是否为 relay。所有 ID 都来自已生成的区域任务或操作帧平台状态；模型不得产生平台 ID。

### 3.3 后端派生规则

frame builder 从 `TrackingPlan.regional_plans` 和 `TrackingPlan.region_tasks` 生成 `region_timeline`：

1. 按 `start_offset_s`、`region_id` 稳定排序。
2. 使用区域单元的中心和边界作为空间摘要。
3. 使用 `RegionTask.active_window` 减去当前 `sim_time_s` 计算相对时间。
4. 使用任务的 assigned IDs、roles、sonar policy、communication links、status 和 degraded reasons。
5. 保留区域任务中定义的 predecessor/successor 作为 handoff 连接。
6. 当区域任务缺少平台或链路时，仍输出区域条目，并保留 degraded/uncovered 原因。

派生过程是纯函数，不读取目标真值，不改变 TrackingPlan，不修改旧 `PlanView` 和 `SegmentOverlay` 字段。

## 4. 前端组件设计

建议新增：

- `RegionTimelinePanel.tsx`：区域泳道和详情状态管理。
- `RegionTimelineRow.tsx`：单区域行、时间条、平台标签和 handoff 节点。
- `regionTimeline.ts`：相对时间、百分比、状态颜色和稳定排序的纯函数。
- 对应 CSS 和 Vitest 测试。

BottomDrawer 新增“分段跟踪”标签。组件只读取 `OperationalFrame.region_timeline`，不直接依赖 Python 领域模型。

### 4.1 视觉编码

- 当前覆盖：青绿色。
- USV relay：橙色。
- handoff reserve：紫色。
- degraded：琥珀色。
- uncovered：红色。
- standby：灰色。
- 当前 `T+0`：竖直强调线。
- 区域之间的交接：连接线加圆形节点。

### 4.2 响应式行为

桌面端使用可拖拽底部抽屉和横向时间轴。窄屏保持区域泳道，但时间轴允许横向滚动；区域详情改为紧凑卡片，确保平台 ID、角色、状态和降级原因仍可读。

## 5. main.py 启动设计

保留现有 `main.py` 的职责：

1. 将 `src` 加入 Python path。
2. 检查 `npm` 和 UI `node_modules`。
3. 启动 Vite 子进程。
4. 输出 Web UI、API/WS 地址。
5. 启动 `underwater_tracking.cli serve`。
6. 处理 Ctrl+C 和异常退出，确保 Vite 子进程被关闭。

本次改动不新增第二个 GUI 进程。启动冒烟测试使用有限步数运行 `main.py`，确认 UI 进程和 API 进程都已启动；交互展示使用同一 Web UI 地址。

## 6. 错误和兼容性

- `region_timeline` 缺失、为空或某一条区域数据非法时，前端显示局部空状态，不阻塞地图和旧时间线。
- 旧 frame 没有 `region_timeline` 时，TypeScript 类型使用可选字段并降级为“暂无区域任务”。
- 区域计划降级不转换成目标覆盖成功；区域条仍显示 degraded/uncovered 和原因。
- 后端序列化失败应被记录为 frame 构建错误，不允许用目标真值或随机构造区域替代。
- 不改变现有 API 路由、不改变旧 PlanView 语义、不移除旧 Segment/SegmentPlan 兼容字段。

## 7. 测试和验收

### Python

- 区域 timeline view 的严格 schema、排序和相对时间计算。
- 单元格、任务、平台角色、relay、handoff 和 degraded reason 的 frame round trip。
- 旧 frame 无区域数据时仍能构造并反序列化。
- `build_operational_frame` 在实时和回放时间点产生正确的相对偏移。

### TypeScript

- 区域排序、时间百分比和当前线计算。
- active / handoff / degraded / uncovered 状态颜色和标签。
- 区域点击详情显示平台角色、声纳、relay 和原因。
- 新 frame 更新时保留选中的 `region_id`。
- 空数据和旧 frame 兼容。

### 启动和端到端

- `python main.py --steps <有限步数>` 返回可控退出状态，并打印 Web UI/API 地址。
- Vite build 通过。
- 实时模式收到新帧后甘特图更新。
- 回放拖动后相对时间轴和区域条同步变化。
- 使用真实运行入口打开浏览器，展示至少一个区域、一个 UUV、一个 USV relay 和一个 handoff。

## 8. 非目标

- 本次不新增独立桌面 GUI。
- 本次不改变预测、意图理解、区域生成或分配算法本身。
- 本次不把目标真值、敌方真实轨迹或评估帧带入操作 UI。
- 本次不删除旧目标组、Segment、PlanView 和旧 API 字段。
