# 前端改动与联调说明

> 用途：后续将当前前端与原项目后端合并时的快速核对清单。前端工作目录：`src/underwater_tracking/ui`。

## 本轮前端改动

- 地图、右侧状态栏、底部抽屉均直接消费后端返回的同一份 `OperationalFrame`；开发、测试和验收都通过真实后端接口获取数据。
- 底部抽屉新增/完善：事件流累积展示、唯一的区域分段跟踪甘特图、LLM 思考过程横向历史；右侧“预测与接力”仅保留区域图谱与列表，避免重复时间线。
- 右侧栏“当前态势”上方新增只读 2×2 作业状态矩阵：任务执行、事件触发、人工反馈、动态调整；高亮完全由态势帧决定，前端不可点击切换。
- 右侧“当前态势、预测与接力、智能助理”三个面板首次打开时均默认收起，由操作员按需展开。
- “当前态势”中的目标潜艇脑已并入“智能节点”的对手脑卡片；点击对手脑展开/收起目标潜艇脑的决策摘要、暴露节点与反跟踪历史。
- 右侧 `智能助理` 使用真实会话接口，提供“方案调整”和“证据回溯”两个模式。方案调整只展示后端返回的方案预览，并通过后端返回的 turn/proposal 显式应用；证据回溯只展示只读回答、记忆版本和已验证来源。
- 智能助理下方的 `记忆窗口` 直接读取真实 SQLite 记忆快照，分为短期、情景、语义和程序记忆；长期记忆展示 version、importance、access count、来源，版本展开和删除都回到后端接口。后端返回空、错误或 degraded 时，界面保留对应状态，不填充演示数据。
- 地图默认展示短、渐隐的 UUV 航迹尾线，使用帧内 `uuvs[].breadcrumb`，不需要新增接口；选中编队的尾线加强显示。地图仍不展示全网通信线、通信/母舰支援圆或 UUV 航点线。仅当选中 UUV 时，突出该艇及同组 UUV，并显示该组的被动测向线和必要中继链路；返航时仍显示母舰—返航 UUV 连线。网格、预测走廊与协方差椭圆使用低对比度样式，确保区域与当前目标仍是视觉重点。
- 右侧作业状态矩阵的非选中与选中状态使用更明显的层级样式；其高亮完全由后端帧字段决定。
- 任务详情抽屉仍以覆盖方式停靠在左侧地图工作区底部，不会改变地图高度，也不会覆盖右侧态势栏。回放模式下抽屉与回放控制条共享边界，形成连续工作台；LLM 思考过程为空时抽屉自动采用紧凑高度。以上均为前端布局行为，不需要后端调整。
- LLM 思考卡片的内部结构固定为：

  ```text
  触发因素 v* + 触发因素内容
  方案 v* + 原有思考正文
  ```

  新记录追加在右侧，旧记录保留在左侧并用箭头连接。

## 实时与回放：后端 → 前端

### 1. 实时首帧与持续推送

| 通道                            | 前端行为                                         | 后端要求                                                                                                                                    |
| ------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET /api/operational/snapshot` | WebSocket 连通后获取快照；在快照完成前暂存 WS 帧 | 返回一条完整 `OperationalFrame` JSON                                                                                                        |
| `WS /ws/operational`            | 接收实时帧；前端最多以约 60 FPS 刷新             | 连续推送完整 `OperationalFrame` JSON；可接收前端文本 `ping` 并回复 `pong`；也可推送 `{ "type": "heartbeat", "sim_time_s": number \| null }` |

帧顺序必须单调：优先递增 `frame_id`，`sim_time_s` 不应倒退。同一事件在后续帧中可重复出现，但 `event_id` 必须稳定，前端会按该 ID 去重并累积最近 300 条实时事件。

### 2. 回放

`GET /api/replay?start_s=<number>&end_s=<number>`

返回：

```json
{
  "frames": ["OperationalFrame", "..."],
  "count": 2
}
```

`end_s` 可选。帧应按 `sim_time_s` 升序返回；回放界面会从已播放的所有帧累积事件和 LLM 思考记录。

### 3. `OperationalFrame` 的关键约定

完整 TypeScript 契约在 `src/types/frames.ts` 的 `OperationalFrame` 及其引用类型中，后端应以它为准。其核心字段为：

```ts
{
  schema_version: string;
  scenario_id?: string | null; // authoritative SituationSnapshot scope; absent on legacy frames
  frame_id: number;
  sim_time_s: number;
  physics_step_s?: number;
  plan_version: number;
  map_bounds: MapBounds;
  uuvs: UUVView[];
  target_estimates: TargetEstimateView[];
  bearing_rays: BearingRayView[];
  groups: GroupView[];
  events: EventView[];
  plans: PlanView[];
  ledger: LedgerView[];
  metrics: MetricView[];
  carrier: CarrierView | null;
  usvs?: USVView[];
  communication_links?: CommunicationLinkView[];
  regional_plans?: Record<string, RegionalPlanView>;
  region_timeline?: RegionTimelineView[];
  plan_timeline?: PlanTimelineView[];
  // 其余可选：brains、adversaries、scheme、intelligence、suggestions 等
}
```

为支持本轮新增的 LLM 抽屉，建议在 **上述三个帧来源**（快照、WS、回放）统一增加以下两个可选字段：

```ts
llm_thinking?: string | null;          // 面向操作员的一段 LLM 思考说明
llm_thinking_trigger?: string | null;  // 引发该思考/方案变化的因素
```

两者以同一帧的 `sim_time_s` 和 `plan_version` 归属。内容未变化时可重复返回，前端会去重；没有新思考时返回 `null` 或省略字段即可。无需额外新增 LLM 接口，也无需返回模型原始逐 token 推理。

右侧 2×2 状态矩阵同样不需要新接口；请在快照、WS、回放帧中返回一个可多选的标识字段：

```ts
operational_stage_flags?: Array<
  "task_execution"     // 任务执行
  | "event_trigger"    // 事件触发
  | "human_feedback"   // 人工反馈
  | "dynamic_adjustment" // 动态调整
>;
```

数组中的每个值对应一个高亮框，允许同时存在多个值。例如目标机动触发重规划时可返回 `["task_execution", "event_trigger", "dynamic_adjustment"]`；人工确认期间可再加入 `"human_feedback"`。空数组或省略字段表示四项均未选中。该字段是**后端判别结果**，不是前端命令，前端不会向后端写回其状态。

### 4. 地图选择与收敛图层

地图选择不新增接口：点击地图 UUV 或右侧“UUV 资源”条目都会把 `selectedUuvId` 交给地图。为使“同组高亮”和按需线条正确，实时/回放帧需保持以下现有字段的 ID 一致：

- `uuvs[].uuv_id`、`uuvs[].group_id`；
- `groups[].group_id`、`groups[].member_ids`；
- `bearing_rays[].uuv_id`（选中组时才绘制）；
- `communication_links[].source_id/target_id/relay/status`（选中组时只绘制已连通的相关中继链路）；
- `carrier.returning_uuv_ids` 或 `uuvs[].deployment_state = "returning"`（返航连线）。

未选中 UUV 时不画任何平台高亮圆圈，也不画全网通信线；因此无需后端返回新的地图开关字段。

### 5. 单目标展示名称

当前指挥界面按单目标运行，区域、预测和证据派生 ID 与展示名称统一使用 `target`。真实后端即使保留其他稳定内部 ID，前端也只显示 `target`；后端只需保证区域、群组、事件和证据中的关联 ID 前后一致。

## 人机操作：前端 → 后端

以下接口已经由前端调用；请求均为 JSON，前端请求超时为 15 秒。涉及方案的写入请求必须校验 `expected_plan_version`，避免覆盖较新的调度方案。

| 接口                                      | 请求体关键字段                                                                 | 用途/期望响应                                                                                                     |
| ----------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `POST /api/directives`                    | `text`, `author`, `expected_plan_version`, `target_ids?`                       | 提交专家自然语言指令；返回 `{ request_id, status }`                                                               |
| `GET /api/directives/{request_id}`        | —                                                                              | 查询指令处理状态，返回 `request_id`, `status`, `expected_plan_version?`, `directive?`, `error?`                   |
| `POST /api/directives/{request_id}/apply` | —                                                                              | 确认应用预览指令，返回 `{ request_id, status }`                                                                   |
| `POST /api/assignments`                   | `target_id`, `uuv_ids`, `expected_plan_version`                                | 人工调整 UUV 分配，返回 `{ request_id, status }`                                                                  |
| `POST /api/sensor-modes`                  | `uuv_id`, `mode: "passive" \| "active"`, `target_id?`, `expected_plan_version` | 切换传感器模式，返回 `{ status, passive_continuous }`                                                             |
| `POST /api/questions`                     | `text`, `counterfactual?`                                                      | 问答/反事实查询；返回 `answer?`, `evidence_ids?`, `counterfactual_plan_id?`, `counterfactual_summary?`, `status?` |
| `POST /api/conversation/messages`         | `conversation_id`, `user_id`, `assistant_mode`, `text`, `expected_plan_version`, `target_ids?`, `region_ids?` | 智能助理的唯一会话写入接口；`assistant_mode` 为 `plan_revision` 或 `evidence_query`，响应包含真实 `memory_context` 和 evidence trace |
| `POST /api/conversation/{conversation_id}/apply` | `user_id`, `turn_id`, `expected_plan_version` | 只应用后端返回的方案预览；前端不自行修改方案 |
| `GET /api/assistant/memory`               | `user_id`, `conversation_id`, `scenario_id`, `query?`, `memory_type?`, `limit?` | 读取当前场景的短期上下文、情景/语义/程序记忆、检索命中、版本和 `memory_status` |
| `GET /api/assistant/memory/{family}/versions` | `user_id`, `scenario_id` | 读取当前场景同一 memory family 的历史版本 |
| `DELETE /api/assistant/memory/{memory_id}` | `user_id`, `scenario_id`, `conversation_id` | 请求后端标记当前场景的整个记忆族删除；成功后刷新快照 |
| `GET /api/assistant/memory/stream`        | `user_id`, `conversation_id`, `scenario_id`, `after_cursor`, `limit` | 读取当前场景 Memory Stream 增量事件，使用 `next_cursor` 推进，不与 LLM thinking 合并 |

会话响应中的 memory 状态来自真实后端；若记忆 worker 或 Embedding 不可用，后端返回 `degraded` 和原因，前端只显示降级状态。若指令引发调度变化，后端仍应通过实时帧/快照发布更新后的 `plan_version`、方案、分段区域、资源状态与事件。前端不以 POST 响应直接替换态势帧。

### 5.1 记忆数据流与回放边界

- App 在启动时生成一次稳定的 `conversation_id`，`user_id` 默认是 `operator`；live 和 replay 从当前 `OperationalFrame.scenario_id` 取得权威场景，并显式传给真实 `/api/assistant/memory*` 接口。旧帧缺少场景时只显示等待/不可用状态，不查询默认场景。
- 页面加载、智能助理发送成功后和周期刷新都会读取记忆快照；Memory Stream 使用 `after_cursor`/`next_cursor` 增量读取，并在前端限制为最近 300 条。
- 短期记忆是当前会话上下文；情景、语义和程序记忆只作为后端筛选后的长期素材。摘要不是原始证据，证据回溯必须显示后端返回的 `MemoryEvidenceTrace` 来源链。
- LLM thinking history 是帧中的操作员可见思考说明，仍由 `llm_thinking`/`llm_thinking_trigger` 渲染；Memory Stream 是后台 worker 的状态审计流，单独位于任务详情的 `Memory Steam` 标签。两者不互相填充。
- `PlaybackBar` 的回放选项固定为 `1x`、`4x`、`10x`；它只改变视频回放帧的播放间隔。常规仿真默认时钟仍是 60 倍，不能把回放倍速传给仿真调度器。

## 真实后端联调

在 `ui` 目录执行：

```powershell
npm run dev
```

Vite 默认将 `/api` 和 `/ws` 代理到 `http://127.0.0.1:8000` / `ws://127.0.0.1:8000`；如后端端口不同，在启动前设置：

```powershell
$env:UNDERWATER_TRACKING_API_PORT = "你的端口"
npm run dev
```

生产部署时，建议让 Web 服务器把同域 `/api`、`/ws` 转发到后端，避免跨域和 WebSocket 地址不一致。

## 复制与部署前端

当前前端是独立的 React/Vite 工程，可直接复制整个 `src/underwater_tracking/ui` 目录到融合后的项目中；不要只复制 `src/`，还需保留：

- `package.json` 与锁文件；
- `vite.config.ts`、`tsconfig*.json`；
- `public/`（如存在）；

首次使用：

```powershell
cd src\underwater_tracking\ui
npm install
```

所有前端开发、测试和验收均使用真实后端数据链路：`npm run dev` 请求同源 `/api` 和 `/ws`，`npm run build` 只生成真实后端模式的静态资源。

开发时，`vite.config.ts` 把 `/api` 和 `/ws` 代理至 `127.0.0.1:8000`。后端端口不同则在启动前设置：

```powershell
$env:UNDERWATER_TRACKING_API_PORT = "8001"
npm run dev
```

`vite` 代理只在开发服务器生效。生产环境应托管 `dist/` 静态文件，并由 Nginx、FastAPI 或其他网关将同域 `/api`、`/ws` 分别反向代理到真实后端。

## 集成验收清单

1. `GET /api/operational/snapshot`、`WS /ws/operational`、`GET /api/replay` 三处都返回同版本的 `OperationalFrame`。
2. 帧携带稳定的 `frame_id`、`event_id`、`plan_version`；区域、群组、USV 和链路引用的 ID 前后一致。
3. 若启用 LLM 思考栏，三处帧来源都提供 `llm_thinking` 与 `llm_thinking_trigger`；作业状态矩阵则提供 `operational_stage_flags`；均不需要新增接口。
4. 人工操作接口对 `expected_plan_version` 做冲突处理，并在后续实时帧中反映最终结果。
5. 使用真实后端启动 UI，验证实时、回放、人机操作和记忆快照/Memory Stream；测试不得通过替代帧流或本地接口实现绕过后端。

## 关键实现位置

- 帧类型/接口契约：`src/types/frames.ts`
- 实时 WebSocket 与快照：`src/hooks/useWebSocket.ts`
- 回放：`src/hooks/useReplay.ts`
- 人机操作 API：`src/services/assistantApi.ts`、`src/services/memoryApi.ts`
- 事件与 LLM 历史累积：`src/App.tsx`
- 底部抽屉（方案、事件、甘特图、LLM）：`src/components/BottomDrawer.tsx`
- 智能助理与记忆窗口：`src/components/assistant/SmartAssistantPanel.tsx`、`src/components/assistant/MemoryWindow.tsx`
- 记忆快照/Memory Stream：`src/hooks/useMemory.ts`、`src/components/BottomDrawer.tsx`
