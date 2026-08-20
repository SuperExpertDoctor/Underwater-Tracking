# 前端改动与联调说明

> 用途：后续将当前前端与原项目后端合并时的快速核对清单。前端工作目录：`src/underwater_tracking/ui`。

## 本轮前端改动

- 增加可独立运行的 **MOCK 实时流与回放流**；MOCK 覆盖母舰、UUV 投放/返航、USV 中继、通信链路、被动测向线、不确定性椭圆、预测走廊、分段接力区域、主动声呐、方案修订、交接失败/恢复等连续过程。
- 地图、右侧状态栏、底部抽屉均直接消费同一份 `OperationalFrame`，因此真实后端只要返回同结构帧即可替换 MOCK。
- 底部抽屉新增/完善：事件流累积展示、唯一的区域分段跟踪甘特图、LLM 思考过程横向历史；右侧“预测与接力”仅保留区域图谱与列表，避免重复时间线。
- 右侧栏“当前态势”上方新增只读 2×2 作业状态矩阵：任务执行、事件触发、人工反馈、动态调整；高亮完全由态势帧决定，前端不可点击切换。
- 右侧“当前态势、预测与接力、LLM Client”三个面板首次打开时均默认收起，由操作员按需展开。
- “当前态势”中的目标潜艇脑已并入“智能节点”的对手脑卡片；点击对手脑展开/收起目标潜艇脑的决策摘要、暴露节点与反跟踪历史。
- 右侧 `LLM Client` 已精简为一个多行输入框和一个发送按钮；不展示对话历史、证据、分类、方案预览或“确认应用”按钮。
- 地图默认展示短、渐隐的 UUV 航迹尾线，使用帧内 `uuvs[].breadcrumb`，不需要新增接口；选中编队的尾线加强显示。地图仍不展示全网通信线、通信/母舰支援圆或 UUV 航点线。仅当选中 UUV 时，突出该艇及同组 UUV，并显示该组的被动测向线和必要中继链路；返航时仍显示母舰—返航 UUV 连线。网格、预测走廊与协方差椭圆使用低对比度样式，确保区域与当前目标仍是视觉重点。
- 顶部 `MOCK 数据` 仅为前端数据源状态标记，真实后端无需提供字段；真实模式不显示该标记。右侧作业状态矩阵的非选中与选中状态使用更明显的层级样式，但其数据契约不变。
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

当前指挥界面按单目标运行，前端 MOCK、区域/预测/证据派生 ID 与展示名称统一使用 `target`。真实后端即使保留其他稳定内部 ID，前端也只显示 `target`；后端只需保证区域、群组、事件和证据中的关联 ID 前后一致。

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
| `POST /api/conversation/messages`         | `conversation_id`, `text`, `expected_plan_version`, `target_ids?`              | 右侧 LLM 输入框的唯一写入接口；前端发送后清空输入框，不渲染该接口的正文、证据或方案预览                           |

LLM 接口的响应只需表示请求已接收/处理完成即可；若指令引发调度变化，后端仍应通过实时帧/快照发布更新后的 `plan_version`、方案、分段区域、资源状态与事件。前端不以 POST 响应直接替换态势帧。

## MOCK 与真实后端切换

### MOCK（仅前端开发）

在 `ui` 目录执行：

```powershell
npm run dev:mock
```

该命令使用 `.env.mock`，其中 `VITE_MOCK_MODE=true`。此时：

- 实时态势来自 `src/mocks/mockData.ts`，每 500 ms 产生一帧；
- 回放由同一份预生成帧提供；
- 人工指令、分配、传感器、问答、会话接口全部在浏览器内模拟，不请求后端；
- 页面顶部会显示 `MOCK 数据` 标识。

MOCK 场景时间步长为 5 秒，主数据与连续性规则维护在 `src/mocks/mockData.ts`；修改该文件后应运行 `npm test` 和 `npm run build:mock`。

### 真实后端联调

在 `ui` 目录执行：

```powershell
npm run dev
```

不要设置 `VITE_MOCK_MODE=true`。Vite 默认将 `/api` 和 `/ws` 代理到 `http://127.0.0.1:8000` / `ws://127.0.0.1:8000`；如后端端口不同，在启动前设置：

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
- `.env.mock`（仅供前端开发使用）。

首次使用：

```powershell
cd src\underwater_tracking\ui
npm install
```

模式切换无需修改前端业务代码：

| 目的 | 命令 | 数据来源 |
| --- | --- | --- |
| 独立前端开发 | `npm run dev:mock` | 加载 `.env.mock` 中的 `VITE_MOCK_MODE=true`；浏览器内生成 MOCK 实时/回放帧，不请求后端 |
| 真实后端联调 | `npm run dev` | 不设置 `VITE_MOCK_MODE=true`；请求同源 `/api` 和 `/ws` |
| 生产构建 | `npm run build` | 生成 `dist/`；默认是真实后端模式 |

开发时，`vite.config.ts` 把 `/api` 和 `/ws` 代理至 `127.0.0.1:8000`。后端端口不同则在启动前设置：

```powershell
$env:UNDERWATER_TRACKING_API_PORT = "8001"
npm run dev
```

`vite` 代理只在开发服务器生效。生产环境应托管 `dist/` 静态文件，并由 Nginx、FastAPI 或其他网关将同域 `/api`、`/ws` 分别反向代理到真实后端；不要把 `.env.mock` 作为生产构建模式使用。

## 集成验收清单

1. `GET /api/operational/snapshot`、`WS /ws/operational`、`GET /api/replay` 三处都返回同版本的 `OperationalFrame`。
2. 帧携带稳定的 `frame_id`、`event_id`、`plan_version`；区域、群组、USV 和链路引用的 ID 前后一致。
3. 若启用 LLM 思考栏，三处帧来源都提供 `llm_thinking` 与 `llm_thinking_trigger`；作业状态矩阵则提供 `operational_stage_flags`；均不需要新增接口。
4. 人工操作接口对 `expected_plan_version` 做冲突处理，并在后续实时帧中反映最终结果。
5. 分别验证 `npm run dev:mock` 与 `npm run dev`；前者不依赖后端，后者应不显示 MOCK 标识。

## 关键实现位置

- 帧类型/接口契约：`src/types/frames.ts`
- 实时 WebSocket 与快照：`src/hooks/useWebSocket.ts`
- 回放：`src/hooks/useReplay.ts`
- MOCK 数据与场景：`src/mocks/mockData.ts`
- MOCK 实时/回放适配：`src/hooks/useMockStream.ts`、`src/hooks/useMockReplay.ts`
- 人机操作 API：`src/services/assistantApi.ts`
- 事件与 LLM 历史累积：`src/App.tsx`
- 底部抽屉（方案、事件、甘特图、LLM）：`src/components/BottomDrawer.tsx`
