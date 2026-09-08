# 前端验收所需的跨组配合

本文档只记录 UI 前端（李业昊）在落实 `F-01`～`F-05` 时发现的接口依赖；本次没有修改 UUV 控制、世界模型或 LLM 实现。

## 当前前端策略

前端已经对新字段做了向后兼容：字段存在时直接展示后端证据，字段缺失时显示“不可用/未知”，不会由路线几何、航迹、UUV 数量、旧 `effect.coverage_ratio` 或页面历史推导扫描/接力结论。普通 `OperationalFrame` 仍不包含 truth/evaluation 字段。

## 需要骆（UUVs 控制相关）补充的同帧证据

请在 HTTP、WebSocket、JSONL replay 使用相同字段名和语义发布：

1. 每个 `execution.regions[]` 增加 `scan_telemetry`：`route_progress`、`scan_round`、`active_coverage_ratio`、`source_backed_ping_count`（可兼容 `active_ping_count`）、`scan_completed`、`scan_completion_threshold`、`evidence_ids`。`active_coverage_ratio` 必须是本轮主动扫描的 source-backed 覆盖，不得用路线完成或编组存在替代。
2. 在 `execution` 发布 `region_entry_evidence`（或等价的 `region_entry_probabilities`、`entry_confirmation_counts`、`entry_confirmation_required_cycles`、`entry_blocking_reasons`），确认计数必须可识别轮次重置和被打断。
3. 发布 `handoff_evidence`：前任/后继编组、同一观测周期、`required_uuv_ids`、`valid_observation_uuv_ids`、状态、阻塞原因和证据 ID。owner 只有在后端交接状态允许时切换；旧组在 `DISAPPEARED` 之前保持 `EXITING`。
4. `task_groups[]` 可选发布 `deployed_member_uuv_ids`、`passive_observation_uuv_ids`，使 UI 能显示 `3/3` 部署与有效被动观测，而不是猜测。
5. 替补必须同时保留 `replacements[]` 的 outgoing/incoming/revision/geometry/batch 信息（当前 UI 兼容可选 `batch_id`），以及对应生命周期编组。

## 需要余（世界模型相关）补充的估计新鲜度

请在每个 `target_estimates[]` 发布 `estimate_freshness`：`status`、`estimate_time_s`、`valid_until_s`、`data_age_s`、`track_revision`、`source_observation_ids`、`reason`。同一版本也可以暂时发布扁平别名。`prediction.health` 需要继续保持与预测走廊同一 revision；缺失 health 时前端只显示 `UNKNOWN`，不会默认为实时。

## 需要 LLM 负责人配合的只读语义

LLM 助理/思考摘要若引用目标状态、owner 或阻塞原因，应携带当前 `frame_id`、`execution_revision`、相关 `evidence_ids`，并明确“等待刷新/等待交接证据”等状态。前端不会把自然语言摘要当作状态机输入，也不会据此改变控制。

## 联调验收

后端补字段后，请使用同一 `seed=42` 正常实时运行（不要只喂 fixture），至少验证：

- 无 ping 的区域为 `0`/`0%`，路线完成不等于扫描完成；新一轮计数明确从 0 开始。
- owner、`EXITING`、`DISAPPEARED` 和 `2/3 → 3/3` 在同一帧可追溯。
- estimate 超过 `valid_until` 时 UI 显示过期并等待刷新，同时仿真时钟继续推进。
- 正式运行没有 truth；只有显式评估模式和醒目标识才可读取真值接口。
