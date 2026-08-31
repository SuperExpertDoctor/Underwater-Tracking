
# 实施计划：智能助理后台记忆处理与 60 倍仿真演示链路

> 本计划依据 docs/superpowers/specs/2026-08-20-smart-assistant-memory-design.md 编写，执行分支为 fix/long-running-replay-acceptance。
> 当前阶段只保存计划和设计修订，不启动 8 小时完整仿真验收。

## 目标与不可违反的约束

本次实现形成一条可长期运行的生产链路：

1. LLM 思考过程是前台同步链路。每次首次制定方案、事件触发动态调整方案、专家反馈触发方案预览时，都由真实方案 LLM 进行思考，并继续遵循预览、专家确认、应用的边界。
2. Memory Worker 是独立后台线程。它持续读取环境观测、事件、决策和人机交互历史，使用真实 LLM 完成记忆筛选、记忆提炼和短期摘要压缩，不阻塞物理仿真 tick 或方案思考请求。
3. 短期记忆是最终推理上下文。长期记忆只能作为经过 Top-K、过滤和重排的有限素材，不能直接替代短期上下文或作为未经验证的事实。
4. 记忆处理任务进入 SQLite 持久化工作队列，支持重启恢复、租约超时重领、重试、降级状态和 Memory Steam 增量审计。
5. 生产记忆链路不得使用本地哈希向量、关键词规则、静态摘要或 mock 结果替代真实 LLM/Embedding。语义检索必须使用本地 `sentence-transformers` 模型；缺少真实聊天服务、Python 包或本地模型权重时必须暴露 degraded、保留任务并重试或明确失败。
6. 常规 UI 演示默认仿真时钟为 60 倍：仿真 1 秒约对应真实 1 分钟，8 小时仿真目标墙上时间约 8 分钟。实际时间仍受真实 LLM 网络延迟、重试和本地 I/O 影响。
7. 视频回放倍速是另一套控制，只保留 1x、4x、10x，不得把回放倍速传给仿真调度器。
8. 不恢复或保留旧的 ConversationPanel/LLM Client 生产实现，不新增前端记忆 mock API。现有用户未提交的 pyproject.toml、tests/api/test_frame_pipeline.py、tests/integration/test_uuv_only_8h_replay_acceptance.py 和 node_modules 相关改动必须保留并逐项核对。

## 现有边界

后端复用以下边界：

- SQLite 统一由 src/underwater_tracking/persistence/sqlite.py 管理，所有新增表通过 schema migration 创建。
- _AgentLoop 负责把 SimulationEngine、CarrierRuntime、OperationalHub 和 SQLite 组装在一起；Memory Worker 在这里启动和停止，但不进入物理循环。
- CarrierRuntime.conversation_message 是统一智能助理入口；process_conversation_message 负责会话分类、方案预览和证据回答。
- StructuredLLM 是已有真实聊天 LLM 端口，记忆筛选/提炼/压缩复用同一审计体系。
- EventRepository、PlanRepository、DecisionLedger 是事件、方案、决策和 LLM 审计的原始来源，证据回溯必须回到这些仓库校验。
- FastAPI create_app 只负责 HTTP/WebSocket 适配，运行时资源由 RunController/_AgentLoop 所有。
- React 主界面由 App、RightSidebar、BottomDrawer、assistantApi 和 useReplay 组成；真实数据行为在 FastAPI + SQLite + Playwright 链路验证。

## 执行顺序

按以下顺序执行，每一步先写失败测试，再实现最小行为，再运行该层验证：

1. 契约、配置和仿真时钟。
2. SQLite schema、记忆仓库和持久化工作队列。
3. 真实 Embedding、结构化记忆 LLM 操作和审计。
4. MemoryService、MemoryWorker、后台源读取和 Memory Stream。
5. Conversation/CarrierRuntime/证据链集成。
6. FastAPI 记忆接口与运行时生命周期。
7. 智能助理、右侧记忆窗口和底部 Memory Steam。
8. 60 倍调度、1x/4x/10x 回放和文档。
9. 分层验收，最后保留 8 小时完整测试供后续显式执行。

---

## Task 1：增加领域契约和配置

### 先写测试

新增或扩展：

- tests/memory/test_models.py
- tests/config/test_models.py
- tests/domain/test_conversation_models.py
- tests/runtime/test_run_controller.py

测试内容：

- user_id 缺省为 operator，空字符串和超长值被拒绝。
- assistant_mode 只允许 auto、plan_revision、evidence_query。
- memory_type 只允许 episodic、semantic、procedural。
- 记忆版本模型校验 version、importance_score、status、supersedes_memory_id 和 user_id。
- 短期上下文包含摘要版本、最近消息、估计 token 数、压缩状态和更新时间。
- MemoryWorkItem 能表示 observation、conversation_turn、maintenance，且 status 只能在 pending、processing、completed、degraded、failed 之间转换。
- MemoryStreamEvent 具有 cursor、event_id、user_id、status、type，事件 payload 不允许未经限制的原始请求内容。
- MemoryContext 明确区分 short_term_context、long_term_material、retrieved_memory_ids、memory_status 和 evidence_trace。
- TimingConfig.demo_time_scale 默认 60，必须大于 0；0 仅保留给 CLI speed=0 的无节流覆盖，不写入配置。
- MemoryConfig 的阈值、队列轮询、租约、重试次数、Top-K、token 上限、衰减和 Embedding 配置通过严格 Pydantic 校验；本地 provider 强制 `embedding_local_files_only=true`。

### 实现

新增 src/underwater_tracking/domain/memory_models.py，定义：

- MemoryType、MemoryStatus、MemoryWorkType、MemoryWorkStatus、MemoryStreamStatus。
- ShortTermMessage、ShortTermContext。
- MemoryVersion、MemoryRetrievalHit、MemoryContext。
- MemoryWorkItem、MemoryEvidenceTrace、MemoryStreamEvent。
- MemoryFilterDecision、MemoryExtractionResult、ShortTermCompressionResult。

扩展 src/underwater_tracking/domain/conversation_models.py：

- ConversationMessage 增加 user_id 和 assistant_mode。
- ConversationAnswer 增加 memory_ids、memory_status 和可展开的 evidence_trace。
- ConversationTurnResult 增加 user_id、assistant_mode、memory_context、memory_stream_cursor 和 queued_memory_work_id。
- 方案消息和 Memory Stream 的字段分开，LLM 思考文本不能被序列化成 MemoryStreamEvent。

扩展 src/underwater_tracking/config/models.py：

- TimingConfig 增加 demo_time_scale: float = 60.0。
- 新增 MemoryConfig，至少包含 enabled、poll_interval_s、maintenance_interval_s、short_term_message_threshold、short_term_token_threshold、short_term_compress_interval_s、recent_message_limit、context_token_budget、retrieval_top_k、retrieval_candidate_limit、min_importance_score、archive_threshold、decay_half_life_s、work_lease_timeout_s、max_attempts、retry_backoff_s、embedding_provider、embedding_model、embedding_local_files_only、embedding_device、embedding_normalize、embedding_vector_version；HTTP 兼容路径额外使用 embedding_base_url、embedding_api_key_env 和 embedding_timeout_s。
- AppConfig 增加 memory: MemoryConfig | None，未配置时由运行时构造显式的 degraded 配置，不使用本地替代实现。
- 保持 StrictModel extra=forbid，避免错拼配置静默生效。

扩展 configs/agent.yaml 或新增 configs/memory.yaml，并在 config/loader.py 加入加载路径。发版配置使用本地 `sentence-transformers` 权重；聊天 LLM 的 API key 仍由 LongCat 配置负责，不能把聊天 endpoint 当作 embedding endpoint。

### 验证

    pytest -q tests/memory/test_models.py tests/config/test_models.py tests/domain/test_conversation_models.py tests/runtime/test_run_controller.py

---

## Task 2：升级 SQLite schema 和记忆仓库

### 先写测试

新增 tests/persistence/test_memory_repository.py：

- 新数据库自动创建 short_term_contexts、long_term_memories、memory_work_items、memory_stream_events、memory_source_cursors。
- 旧 schema v3 打开时可幂等迁移到新版本；更新前的 runtime_events、plans、llm_calls 数据不丢失。
- 同一 user_id 的短期上下文可以更新，不同 user_id 永远隔离。
- 长期记忆第一次创建为 v1；更新在同一事务中创建 v2 并将 v1 标记 superseded；并发或重复写入不能产生两个 active 最新版本。
- 版本链按 v1、v2、v3 稳定返回；删除记忆族后 active 检索为空，但原始事件和决策仍存在。
- work item 可以 enqueue、claim、complete、fail；processing 超过 lease 后可以重领；attempts 和 last_error 正确累积。
- 同一 source_key 不重复入队；cursor 按 user/scenario/source_type 稳定推进。
- stream cursor 只返回当前用户、当前 conversation 的事件，after_cursor 不重复返回旧事件，limit 受到硬上限约束。
- access_count、last_accessed_at、importance 的更新只作用于最终命中版本。

### 实现

修改 src/underwater_tracking/persistence/sqlite.py：

- SCHEMA_VERSION 从 3 升级到 4。
- 用迁移表定义创建 short_term_contexts、long_term_memories、memory_work_items、memory_stream_events、memory_source_cursors。
- 为 user_id/status/memory_type/created_at、work status/available_at、stream user/conversation/cursor 建立索引。
- payload、来源 ID、检索理由和审计元数据使用 json_dumps，设定大小上限。
- 记忆表保存 memory_id、memory_family_id、version、user_id、memory_type、summary、importance_score、embedding、embedding_version、status、supersedes_memory_id、source IDs、change_reason、created_at、last_accessed_at、access_count、sim_time_s。

新增 src/underwater_tracking/persistence/memory.py，提供 LongTermMemoryRepository 和 ShortTermContextRepository：

- get_short_term(user_id, conversation_id)
- append_messages(user_id, conversation_id, messages)
- save_compressed_context(expected_summary_version, summary, retained_messages)
- create_memory_version(memory, expected_previous_version)
- list_active(user_id, filters, limit)
- list_versions(user_id, memory_family_id)
- mark_deleted(user_id, memory_id)
- record_access(user_id, memory_ids)
- enqueue_work(item, source_key)
- claim_work(worker_id, now, lease_timeout_s)
- complete_work(work_id, worker_id)
- fail_work(work_id, worker_id, status, error, retry_at)
- get_source_cursor / advance_source_cursor
- append_stream_event / list_stream_events

所有会改变版本、短期摘要、队列状态的操作使用 transaction(conn) 和校验后的 user_id。Repository 的连接独立于 CarrierRuntime 的 checkpointer，允许后台线程安全关闭和重启。

### 验证

    pytest -q tests/persistence/test_memory_repository.py tests/persistence

---

## Task 3：接入真实 Embedding 和结构化记忆 LLM

### 先写测试

新增：

- tests/memory/test_embeddings.py
- tests/memory/test_reasoner.py
- tests/memory/test_real_llm_memory.py

测试规则：

- 本地 `SentenceTransformerEmbeddingProvider` 始终使用 `local_files_only=true`；缺少包或权重时明确返回配置错误或 degraded，绝不联网下载、切换 HTTP、返回哈希或零向量。
- Embedding 响应维度和模型版本会被校验并持久化。
- 响应格式、超时和 HTTP 错误进入已有 LLM 错误分类，不泄漏 API key。
- MemoryReasoner 的结构化结果校验候选 memory_id、source IDs、user_id、importance 和摘要长度，拒绝模型凭空引用不存在的来源。
- 不允许用关键词匹配决定 should_store，不允许用静态摘要通过生产测试。
- 标记 real_llm 的测试只在 UNDERWATER_TRACKING_API_KEY 和本地 sentence-transformers 权重可用时运行；测试真实的本地向量、memory_filter、memory_extract、short_term_compress 调用和 LLM audit 记录。没有凭据或权重时只跳过真实测试，不把 degraded 当成功。

### 实现

新增 src/underwater_tracking/memory/embeddings.py：

- 定义 EmbeddingProvider 协议。
- 实现 `SentenceTransformerEmbeddingProvider`，懒加载配置的本地模型并调用真实 `encode()`；HTTPEmbeddingProvider 只保留为显式兼容 provider。
- 使用 api_key_env 在调用时读取凭据，复用现有超时、重试和脱敏规则。
- 只返回真实 provider 向量；provider 不可用时抛出可分类错误，由上层返回 degraded。
- 记录 operation=memory_embedding 的 hash、模型、耗时、token 元数据，不记录正文和凭据。

新增 src/underwater_tracking/memory/reasoner.py：

- MemoryReasoner 依赖 StructuredLLM、LongTermMemoryRepository 的候选范围和 MemoryConfig。
- memory_filter 使用 operation=memory_filter、版本化 prompt 和 MemoryFilterDecision。
- memory_extract 使用 operation=memory_extract、版本化 prompt 和 MemoryExtractionResult。
- short_term_compress 使用 operation=short_term_compress、版本化 prompt 和 ShortTermCompressionResult。
- prompt 只提供有限短期上下文、当前来源和候选版本，不把整个长期库发送给 LLM。
- 服务端验证 LLM 返回内容：来源必须属于输入集合，更新目标必须属于同 user_id，摘要不能引入输入没有的事实，不能突破 token/长度预算。
- LLM 失败只产生错误状态，不生成规则替代记忆。

新增 src/underwater_tracking/memory/retriever.py：

- query 向量化后按 user_id、status、memory_type、时间和最低重要性过滤。
- 计算语义相似度，再合并重要性、墙上时间衰减、访问频次做重排。
- 最终 Top-K 和记忆 token 预算硬截断，只有最终命中版本更新 access_count。
- Embedding 不可用时返回空 long_term_material 和 degraded，不阻塞当前方案 LLM。

### 验证

    pytest -q tests/memory/test_embeddings.py tests/memory/test_reasoner.py
    pytest -q -m real_llm tests/memory/test_real_llm_memory.py

---

## Task 4：实现 MemoryService 和持久化后台 MemoryWorker

### 先写测试

新增：

- tests/memory/test_service.py
- tests/memory/test_worker.py
- tests/memory/test_source_reader.py

必须验证：

- prepare_context 只读取短期上下文并调用 retriever，不调用 MemoryReasoner 的筛选/提炼/压缩方法。
- prepare_context 将长期命中标记为 long_term_material，不把长期素材覆盖到 short_term_context。
- accept_turn 先落原始短期消息和 queue work item，再返回 queued；请求线程不等待真实记忆 LLM。
- worker.poll_once 能按 work type 依次执行 filter、extract、version transaction、short-term compression。
- short-term 消息数/token/周期达到阈值才排入压缩；压缩成功递增 summary_version，失败保留旧摘要和有界最近消息。
- 一次性问候/感谢是否过滤完全由真实 memory_filter 结果决定，服务层不加入关键词规则。
- observation 和 conversation_turn 都能保存来源 ID；重复 source_key 不产生重复记忆。
- worker 暂时遇到 transient LLM error 会退避重试，超过上限进入 degraded/failed，数据库中仍可回溯。
- stop 会唤醒并在 bounded timeout 内退出；重启后 lease 超时 item 可继续处理。
- worker 处理 memory_stream_event 时区分 queued、processing、completed、degraded，事件不会混入 LLM thinking 事件。

### 实现

新增 src/underwater_tracking/memory/service.py：

- MemoryService.prepare_context(user_id, conversation_id, query, filters, scenario_id) 返回 MemoryContext。
- MemoryService.accept_turn(turn, result, source_refs) 写入短期消息、conversation_turn work item 和 queued stream event。
- MemoryService.enqueue_observation(source_ref, payload) 只写受限观测和 source_key，不在 simulation tick 内调用 LLM。
- MemoryService.snapshot、versions、delete、stream 为 API 使用的只读/管理接口。
- Service 将 user_id 作为所有读写和检索的强制边界；长期记忆只能进入有限 context payload。

新增 src/underwater_tracking/memory/worker.py：

- MemoryWorker 构造参数包括 repository、service、reasoner、source_reader、config、worker_id 和 stop_event。
- 提供 start、stop、poll_once、run_forever，线程名固定为 underwater-memory-worker。
- 主循环先领取队列，再按低频 interval 读取 source cursor；无任务时用 Event.wait，不使用阻塞 sleep。
- conversation_turn 处理交互历史；observation 处理事件/决策/环境观测；maintenance 处理到期压缩、访问衰减、冲突版本和归档。
- 内容理解动作必须调用真实 MemoryReasoner；时间衰减、访问频次和阈值归档可以是确定性维护规则，但不能替代筛选/提炼/压缩 LLM。
- 所有版本写入和 work item 状态变更使用短事务，不持有仿真锁，不调用 CarrierRuntime 的同步锁。
- 维护任务设置 bounded batch，防止长时间运行中队列无限增长；记录 queue backlog、oldest item age、last success 和 degraded reason。

新增 src/underwater_tracking/memory/source_reader.py：

- 从 EventRepository、DecisionLedger、PlanRepository 和已落库短期对话读取新增来源。
- 使用 memory_source_cursors 去重，按 runtime event/decision/plan 的稳定 ID 推进。
- 读取失败只记录 degraded stream event，下一次轮询可重试；不伪造观测。
- 每个 source payload 只携带允许的摘要字段和原始来源 ID，完整证据仍留在原有仓库。

### 验证

    pytest -q tests/memory/test_service.py tests/memory/test_worker.py tests/memory/test_source_reader.py

---

## Task 5：接入 Conversation、CarrierRuntime 和证据回溯

### 先写测试

扩展：

- tests/agent/test_conversation.py
- tests/agent/test_runtime.py
- tests/api/test_conversation.py
- tests/agent/test_questions.py

测试内容：

- plan_revision 的每次请求都产生独立 LLM thinking 结果和 conversation turn；不等待 memory worker。
- evidence_query 只读，不应用方案；返回 memory_ids 与经仓库验证的 source IDs。
- 记忆检索命中但来源不存在时，返回“记忆线索存在、原始证据不足”，不能把摘要当作事实。
- 长期记忆不在短期上下文中时，方案 LLM 仍能基于当前态势/当前方案/短期上下文工作。
- evidence_ids 和 memory_ids 不能超出本次提供/命中的候选集合。
- 方案预览仍必须经过显式 apply，记忆 worker 不能直接产生 plan commit。
- 同一 conversation/user 的短期消息被后台保存；不同 user 的问题不能命中对方长期记忆。
- Worker 处理交互结果发生在 conversation_message 返回之后，测试用 queue/stream 状态断言异步边界。
- runtime close 时先停止 worker，再关闭 LLM client、runtime、repositories；重复 close 安全。

### 实现

修改 src/underwater_tracking/agent/nodes/conversation.py：

- ConversationContext 增加 memory_service、user_id、assistant_mode 和 MemoryContext。
- build_classification_payload 只注入短期上下文和有限 long_term_material，明确标注素材不是事实。
- process_conversation_message 先 prepare_context，再进行现有 conversation_classification 和方案/证据分支；完成后调用 accept_turn 入队。
- 保留 LLM 思考过程的独立输出和事件，Memory Stream 只记录记忆处理状态。
- 证据回溯先通过 MemoryRetriever 找线索，再从 EventRepository、DecisionLedger、PlanRepository、Knowledge 查询验证来源，最后只把验证通过的原始证据交给回答 LLM。

修改 src/underwater_tracking/agent/runtime.py：

- conversation_message 接收新的 user_id/assistant_mode，构造含 memory service 的 context。
- 保留现有 _lock 对方案状态的保护，但 MemoryService.accept_turn 只提交短事务，不能在锁内等待后台 LLM。
- apply_conversation 继续只应用已保存预览，并将应用结果写入可回溯来源。
- close 先由 _AgentLoop 停 worker，再执行现有 payload/checkpointer 关闭。

修改 src/underwater_tracking/agent/graphs/central.py：

- CarrierDependencies 增加可选 MemoryService 只用于会话入口/来源登记，不把 MemoryWorker 放入 carrier graph。
- 事件、UUV 状态、任务轮转、资源轮转继续按现有算法运行，不依赖记忆处理完成状态。

修改 src/underwater_tracking/domain/conversation_models.py 的序列化兼容逻辑，确保旧 API consumer 缺少 memory 字段时仍能显示空状态，但后端始终返回真实 memory_status。

### 验证

    pytest -q tests/agent/test_conversation.py tests/agent/test_runtime.py tests/agent/test_questions.py tests/api/test_conversation.py

---

## Task 6：暴露真实 FastAPI 记忆接口并接入运行时生命周期

### 先写测试

新增或扩展：

- tests/api/test_memory_routes.py
- tests/api/test_app_lifespan.py
- tests/integration/test_memory_api_real_sqlite.py
- tests/integration/test_agent_loop.py

测试内容：

- POST /api/conversation/messages 接收 user_id、assistant_mode，并返回 memory_context、queued work ID 和 stream cursor。
- GET /api/assistant/memory 返回短期记忆、三类长期记忆、当前命中和版本信息。
- GET /api/assistant/memory/{memory_family_id}/versions 只允许同 user_id 访问。
- DELETE /api/assistant/memory/{memory_id} 标记整个逻辑族 deleted，随后默认检索和快照不再显示。
- GET /api/assistant/memory/stream 支持 user_id、conversation_id、after_cursor、limit，增量请求不重复。
- 任何跨 user_id、非法 limit/cursor/memory type、非法版本操作返回 4xx。
- HTTP 接口通过真实临时 SQLite 持久化，重建 app/worker 后数据仍存在。
- FastAPI lifespan 不启动第二个 worker，也不在 app close 前截断队列；controller close 能关闭 worker。
- 无 Embedding/Memory LLM 时 API 返回 degraded 和可见错误原因，不返回假命中/假摘要。

### 实现

修改 src/underwater_tracking/api/app.py：

- ConversationMessageRequest 增加 user_id、assistant_mode。
- 新增 query/request models：MemorySnapshotQuery、MemoryVersionQuery、MemoryDeleteRequest、MemoryStreamQuery。
- 新增 GET /api/assistant/memory、GET /api/assistant/memory/{memory_family_id}/versions、DELETE /api/assistant/memory/{memory_id}、GET /api/assistant/memory/stream。
- 读写操作通过 asyncio.to_thread 调用 runtime memory port，避免阻塞事件循环。
- 对响应数量、cursor、user_id 和来源范围做服务端校验，返回真实 memory_status。

修改 src/underwater_tracking/api/dependencies.py：

- 新增 MemoryPort 协议：snapshot、versions、delete、stream。
- RuntimePort 或其 adapter 暴露 memory service，避免 API 直接导入 SQLite/LLM 实现。

修改 src/underwater_tracking/cli.py：

- _AgentLoop 在创建真实 master LLM、repositories 后按配置创建本地 SentenceTransformerEmbeddingProvider、MemoryReasoner、MemoryService、MemoryWorker；HTTP provider 仅可显式选择。
- attach(engine) 后启动 worker；worker 使用同一 agent.db 的独立连接。
- _deps() 将 MemoryService 注入 CarrierRuntime 所需 context，但不将 MemoryWorker 注入仿真 graph。
- close 顺序为 worker.stop、memory provider/LLM close、runtime close、knowledge/repository close；异常也要走同一清理路径。
- 无 memory 配置时不偷偷创建本地向量回退；构建 degraded provider，使接口状态诚实可见。

修改 src/underwater_tracking/agent/llm.py 或新增共享 HTTP helper，保证 Embedding 和 chat LLM 的 API key、请求 hash、响应 hash、错误类别和关闭语义一致。

### 验证

    pytest -q tests/api/test_memory_routes.py tests/api/test_app_lifespan.py tests/integration/test_memory_api_real_sqlite.py
    pytest -q tests/integration/test_agent_loop.py

---

## Task 7：实现真实数据驱动的智能助理和右侧记忆窗口

### 先写测试

新增或扩展：

- src/underwater_tracking/ui/src/services/memoryApi.test.ts
- src/underwater_tracking/ui/src/components/assistant/SmartAssistantPanel.test.tsx
- src/underwater_tracking/ui/src/components/assistant/MemoryWindow.test.tsx
- src/underwater_tracking/ui/src/components/RightSidebar.test.tsx
- src/underwater_tracking/ui/src/e2e/visualCommandCenterFlow.test.ts

测试要求：

- 只验证 API 类型解析、空状态、错误状态和组件布局，不在 production app 中注入 memory mock response。
- SmartAssistantPanel 标题/可见文本为 智能助理；提供 方案调整、证据回溯两个模式。
- plan_revision 显示后端返回的预览和显式应用按钮；evidence_query 显示只读回答、memory version 和验证后的来源。
- MemoryWindow 显示短期记忆、情景记忆、语义记忆、程序记忆；显示 version、importance、access count、来源和 degraded 状态。
- 版本展开请求真实 /api/assistant/memory/{family}/versions；删除请求真实 DELETE，并刷新快照。
- 页面加载、发送消息后和定时刷新都使用真实 API；后端为空时显示空状态，失败时显示错误，不回填固定演示数据。
- E2E 使用运行中的 FastAPI、同一 SQLite 和真实配置 LLM，断言真实响应可被渲染；不通过 vi.mock 或静态 fixture 响应绕过 API。

### 实现

新增 src/underwater_tracking/ui/src/services/memoryApi.ts：

- 定义 ShortTermContextView、MemoryVersionView、MemoryRetrievalHitView、MemoryStreamEventView、MemorySnapshotView、MemoryEvidenceTraceView。
- 提供 getMemorySnapshot、getMemoryVersions、deleteMemory、getMemoryStream，统一处理 cursor、timeout、4xx/5xx。
- 扩展 src/underwater_tracking/ui/src/services/assistantApi.ts 的 request/response 类型，发送 user_id、assistant_mode，读取 memory_context 和 evidence_trace。

新增或替换 src/underwater_tracking/ui/src/components/assistant/SmartAssistantPanel.tsx：

- 替换 ConversationPanel.tsx 的生产入口，不保留 LLM Client 文案和无结果吞错逻辑。
- conversation_id 由 App 稳定持有，user_id 默认 operator，模式切换直接传后端。
- 展示发送中、真实错误、方案预览、应用结果、证据回溯和 memory_status。
- 只能通过后端返回的 proposal 调用 apply，不在前端自行修改计划。

新增 src/underwater_tracking/ui/src/components/assistant/MemoryWindow.tsx：

- 用 tabs/segmented control 切换短期、情景、语义、程序记忆。
- 当前版本和历史版本使用可展开行；删除使用图标按钮并提供 aria-label/tooltip。
- 命中项显示相似度、版本、类型、来源数量和访问时间；空状态、加载状态、错误和 degraded 独立显示。
- 记忆摘要正文不承担原始证据功能，证据回溯使用 MemoryEvidenceTrace 展示来源链。

修改 src/underwater_tracking/ui/src/components/RightSidebar.tsx：

- 将 LLM Client 标题改为 智能助理。
- 将 assistant panel 与 MemoryWindow 置于同一真实 API 数据边界，默认收起/展开行为沿用现有界面。
- 更新 aria-label、空状态和测试期望。

修改 src/underwater_tracking/ui/src/App.tsx：

- 在 App 层创建稳定 conversationId 和 userId，传给智能助理。
- live 模式通过 hook 周期读取 memory snapshot/stream；不通过 frame 字段伪造记忆。
- replay 模式读取当前运行数据库产生的 memory snapshot/stream，明确展示历史状态；没有数据时显示真实空状态。
- 将 memory stream events 传给 BottomDrawer，和 thinkingHistory 分开保存。

修改 src/underwater_tracking/ui/src/FRONTEND_INTEGRATION.md：

- 说明智能助理两种模式、真实接口、短期/三类长期记忆、版本/删除和 Memory Steam 的 cursor。
- 明确 LLM 思考过程与 Memory Steam 是两个独立数据流，记忆后台异步完成。
- 明确回放 1x/4x/10x 与仿真默认 60x 不同。

### 验证

    cd src/underwater_tracking/ui
    npm test -- --run src/services/memoryApi.test.ts src/components/assistant/SmartAssistantPanel.test.tsx src/components/assistant/MemoryWindow.test.tsx src/components/RightSidebar.test.tsx
    npm run build

---

## Task 8：增加底部 Memory Steam 并保留独立 LLM 思考流

### 先写测试

新增或扩展：

- src/underwater_tracking/ui/src/components/BottomDrawer.test.tsx
- src/underwater_tracking/ui/src/components/MemorySteam.test.tsx
- src/underwater_tracking/ui/src/e2e/visualCommandCenterFlow.test.ts

测试内容：

- LLM 思考过程和 Memory Steam 是相邻但独立的 tab。
- LLM tab 只显示方案思考历史，不显示 memory_filter、memory_extract、compression 事件。
- Memory Steam 显示 context_loaded、retrieval、filtered、extracted、compression、version、access、archive、evidence_trace 和 degraded。
- queued -> processing -> completed 的事件顺序稳定；同一 cursor 刷新不重复。
- 事件中能展开因果链 问题 -> 记忆版本 -> 来源事件/决策/知识 -> 方案版本。
- 无事件时显示真实空状态，接口错误时显示错误状态，不显示固定假事件。

### 实现

新增 src/underwater_tracking/ui/src/components/MemorySteam.tsx：

- 接收 MemoryStreamEventView[]、loading、error、cursor 和 onLoadMore。
- 使用统一时间/状态/类型标签，当前专家消息关联的 event 高亮。
- 记忆素材和原始证据采用不同标签，证据来源可以回调现有 onSelectEvidence。
- 对长事件文本和来源 ID 做截断/展开，保持固定行高和滚动窗口，避免长时间运行造成 DOM 无限增长。

修改 src/underwater_tracking/ui/src/components/BottomDrawer.tsx：

- 在 LLM 思考过程旁增加 Memory Steam tab，使用 lucide 图标。
- 将 compact empty 逻辑、active tab 索引、tab accessibility 和高度计算更新为新 tab。
- 不从 frame.llm_thinking 推导 Memory Stream；Memory Stream 只接受 App 传入的真实接口事件。

修改 App.css：

- 增加 Memory Steam 状态、时间线、版本链和 degraded 样式。
- 使用现有设计 token，确保移动端不出现事件文本覆盖、按钮溢出和不可滚动区域。

### 验证

    cd src/underwater_tracking/ui
    npm test -- --run src/components/BottomDrawer.test.tsx src/components/MemorySteam.test.tsx
    npm run build

---

## Task 9：实现 60 倍仿真调度和 1x/4x/10x 视频回放

### 先写测试

扩展：

- tests/runtime/test_run_controller.py
- tests/cli/test_cli.py 或现有 CLI 测试文件
- src/underwater_tracking/ui/src/components/PlaybackBar.test.tsx
- src/underwater_tracking/ui/src/hooks/useReplay.test.ts

测试内容：

- TimingConfig.demo_time_scale 默认 60 且配置验证拒绝非正数。
- RunController 未提供 speed 时采用 config.timing.demo_time_scale。
- 指定 speed=0 时保持无节流语义；指定正数时使用 deadline pacing，不会按每 tick 固定 sleep 累积漂移。
- 以可控 monotonic/Event 测试模拟时间 8h 的目标 wall interval 约为 480s，不实际运行 8h。
- serve --speed 未提供时不覆盖配置，明确传值时覆盖配置。
- PlaybackBar 只渲染 1x、4x、10x；不存在 20x、50x、100x。
- useReplay 传递 1、4、10 时延迟按回放速度变化，其他倍速不能通过 UI option 产生。

### 实现

修改 src/underwater_tracking/config/models.py：

- TimingConfig.demo_time_scale 默认 60.0，并保持 physics_step_s、observation_step_s 等物理仿真时间定义不变。

修改 src/underwater_tracking/runtime/run_controller.py：

- 构造参数 speed 改为 float | None；None 解析为 config.timing.demo_time_scale。
- _start_worker 用 monotonic wall_origin 和 sim_origin 计算目标 deadline：
  target_wall = wall_origin + (current_sim_time - sim_origin) / effective_speed。
- 每步执行结束后只等待到目标 deadline，避免 LLM 延迟后再次累计一整步 sleep。
- 保留 speed=0 的无节流覆盖和 stop Event 中断语义。
- RunSummary/manifest 增加 effective demo speed，便于 UI 和验收区分仿真速度与回放速度。

修改 src/underwater_tracking/cli.py：

- serve --speed 默认 None，help 说明默认使用 config timing.demo_time_scale。
- _serve 只有用户显式提供 speed 时才覆盖配置，拒绝负数。
- agent-run/simulate 保持按 steps 的离线语义，不因为 UI 演示时钟改变其确定性 step 结果。

修改 src/underwater_tracking/ui/src/components/PlaybackBar.tsx：

- SPEEDS 固定为 [1, 4, 10]。
- 保持 useReplay 的数值 speed 和 frame timeline 行为，不把它映射到后端 simulation speed。

### 验证

    pytest -q tests/runtime/test_run_controller.py tests/cli
    cd src/underwater_tracking/ui && npm test -- --run src/components/PlaybackBar.test.tsx src/hooks/useReplay.test.ts

---

## Task 10：移除旧实现、更新真实集成测试和文档

### 先写测试/检查

- rg 检查生产 src/underwater_tracking/ui 和后端中不存在 LLM Client、ConversationPanel 作为入口、local hash/n-gram embedding、memory keyword fallback。
- 检查 UI 测试中不存在对 memoryApi/assistantApi 的 vi.mock、固定 Memory Stream 成功响应或为了通过而注入的 mock 后端。
- 检查 API 响应中的 memory 数据来自 SQLite，不来自 frame 常量或组件默认数据。
- git diff --check、ruff、mypy、TypeScript build 都必须通过。

### 实现

- 删除或停止导出旧的 src/underwater_tracking/ui/src/components/assistant/ConversationPanel.tsx；App 只引用 SmartAssistantPanel。
- 清理旧的 LLM Client 文案、LLM Client 输入 aria-label 和对应测试期望。
- 清理任何本地 hash embedding 或规则 memory filter 实现；MemoryRetriever 只能依赖真实 `EmbeddingProvider`，发版配置必须选择 `SentenceTransformerEmbeddingProvider`。
- 保留无数据时的空状态，不把空状态改成静态演示数据。
- 更新 FRONTEND_INTEGRATION.md、必要的 README/运行文档，写明真实 LLM/Embedding 凭据、后台 worker、队列状态、60 倍时钟和回放档位。
- 继续保留用户现有未提交改动；只修改与本计划相关的文件，若同一文件有用户改动则先逐段合并，不使用 reset/checkout 覆盖。

### 验证

    rg -n "LLM Client|ConversationPanel|hash embedding|n-gram|mock memory|vi\\.mock" src/underwater_tracking/ui src/underwater_tracking/memory src/underwater_tracking
    git diff --check
    ruff check src tests
    mypy src
    cd src/underwater_tracking/ui && npm run build

---

## Task 11：分层验收和后续 8 小时完整运行

### 本轮实现收口

- 已完成本地 `SentenceTransformerEmbeddingProvider`、严格的 local-only 配置、真实向量审计和 `_AgentLoop`/MemoryWorker 双 provider 接入；LongCat 不再作为 embedding 服务。
- 已完成 Memory Steam 结构化来源、版本、方案版本和证据回溯事件；证据开始/完成事件采用确定性 ID、原子事务和并发幂等写入。
- 已将区域策略/候选区域的单次真实 LLM 请求限制为最多 4 个区域，适配 LongCat `max_tokens=4096` 下的 reasoning 和嵌套 JSON 输出；超时、截断、非法 JSON 不生成伪造策略。
- 已完成记忆/API/会话/运行时/UI 的分层回归验证。当前机器缺少发版默认多语言模型权重时，链路按设计显示 `degraded`，不联网下载或伪造向量。

完整 8 小时仿真验收（以 100x 演示速度运行）按操作员后续安排单独执行；它不属于本轮代码收口的阻塞项。

### 当前实现阶段执行

先执行不依赖 8 小时的快速验收：

    pytest -q tests/memory tests/persistence/test_memory_repository.py tests/api/test_memory_routes.py
    pytest -q tests/agent/test_conversation.py tests/agent/test_runtime.py tests/runtime/test_run_controller.py
    pytest -q -m "not long_running"
    cd src/underwater_tracking/ui && npm test && npm run build

启动真实后端和前端进行浏览器验收：

    python -m underwater_tracking.cli serve --config configs/scenario/uuv_only_single_target.yaml --seed 42 --host 127.0.0.1 --port 8000
    cd src/underwater_tracking/ui && npm run dev

使用 Playwright 真实 API 验收：

- 首屏能看到智能助理，能切换方案调整/证据回溯。
- 提交专家反馈后先出现真实方案思考和预览，点击应用前方案版本不变，点击应用后版本变化。
- 同一消息之后 Memory Steam 先显示 queued/processing，后台完成后显示筛选、提炼或压缩事件。
- 右侧记忆窗口能看到短期和三类长期记忆，展开版本链能看到 v1 -> v2，删除后不再出现在默认快照。
- 证据回溯能展示记忆素材与经仓库验证的原始事件/决策/知识/方案来源。
- 断开或错误配置真实 LLM 时显示 degraded，不显示假成功。
- 回放控制只显示 1x、4x、10x；仿真状态仍按 60x 运行。
- 运行期间检查 SQLite 队列 backlog、processing lease、failed/degraded、Memory Stream cursor、LLM audit operation、sim_time 单调性和 operational frame 数量，确认后台记忆未阻塞 UUV 任务/资源轮转。

### 后续显式执行的完整验收

本轮不启动完整 8 小时测试。实现和短验收通过后，再显式执行：

    UNDERWATER_TRACKING_RUN_8H=1 pytest -q tests/integration/test_uuv_only_8h_replay_acceptance.py

该测试必须使用默认 60 倍仿真时钟跑满 sim_time_s=28800，目标墙上时间约 8 分钟，并验证：

- UUV 能力监视、任务组动态调整和航程耗尽资源轮换持续发生。
- 事件触发的方案动态调整保持 LLM thinking 与 Memory Stream 分离。
- 环境观测和人机交互进入 durable work queue，worker 不丢任务、不重复版本。
- 短期摘要周期性更新，三类长期记忆可检索、版本可追踪、低权重可归档。
- 证据回溯只引用存在的原始来源；长期记忆不能替代短期上下文。
- SQLite 文件大小、WAL、内存队列、UI stream 窗口和前端 DOM 都有界增长。
- 真实 LLM 延迟、重试和 degraded 事件均可审计；没有规则/哈希/mock 伪造成功。

---

## 文件清单

新增：

- src/underwater_tracking/domain/memory_models.py
- src/underwater_tracking/memory/__init__.py
- src/underwater_tracking/memory/embeddings.py
- src/underwater_tracking/memory/reasoner.py
- src/underwater_tracking/memory/retriever.py
- src/underwater_tracking/memory/service.py
- src/underwater_tracking/memory/source_reader.py
- src/underwater_tracking/memory/worker.py
- src/underwater_tracking/persistence/memory.py
- src/underwater_tracking/ui/src/services/memoryApi.ts
- src/underwater_tracking/ui/src/components/assistant/SmartAssistantPanel.tsx
- src/underwater_tracking/ui/src/components/assistant/MemoryWindow.tsx
- src/underwater_tracking/ui/src/components/MemorySteam.tsx
- tests/memory/test_models.py
- tests/memory/test_embeddings.py
- tests/memory/test_reasoner.py
- tests/memory/test_retriever.py
- tests/memory/test_service.py
- tests/memory/test_worker.py
- tests/memory/test_source_reader.py
- tests/persistence/test_memory_repository.py
- tests/api/test_memory_routes.py
- tests/integration/test_memory_api_real_sqlite.py
- 对应 UI 单测和真实 Playwright 验收用例

修改：

- docs/superpowers/specs/2026-08-20-smart-assistant-memory-design.md
- src/underwater_tracking/config/models.py
- src/underwater_tracking/config/loader.py
- configs/agent.yaml 或 configs/memory.yaml
- src/underwater_tracking/persistence/sqlite.py
- src/underwater_tracking/domain/conversation_models.py
- src/underwater_tracking/agent/nodes/conversation.py
- src/underwater_tracking/agent/runtime.py
- src/underwater_tracking/agent/graphs/central.py
- src/underwater_tracking/cli.py
- src/underwater_tracking/api/app.py
- src/underwater_tracking/api/dependencies.py
- src/underwater_tracking/runtime/run_controller.py
- src/underwater_tracking/ui/src/App.tsx
- src/underwater_tracking/ui/src/components/RightSidebar.tsx
- src/underwater_tracking/ui/src/components/BottomDrawer.tsx
- src/underwater_tracking/ui/src/components/PlaybackBar.tsx
- src/underwater_tracking/ui/src/hooks/useReplay.ts
- src/underwater_tracking/ui/src/services/assistantApi.ts
- src/underwater_tracking/ui/src/App.css
- src/underwater_tracking/ui/FRONTEND_INTEGRATION.md
- 相关后端、前端和集成测试文件

## 自审清单

- 方案思考和 Memory Worker 是否有独立接口、线程、审计事件和 UI tab。
- 每个 Memory Worker 的语义处理是否都调用真实 LLM；是否完全移除了 hash/keyword/static/mock fallback。
- 短期记忆是否按周期/阈值由后台压缩，而不是在 conversation API 请求中同步等待。
- 长期记忆是否始终只作为受限素材，最终推理是否在短期上下文中完成。
- work item 是否可重试、可恢复、可去重，SQLite 事务是否不会持有物理仿真锁。
- user_id 是否覆盖所有查询、版本、删除、stream 和证据回溯路径。
- Memory Stream 是否使用真实 cursor，是否和 LLM thinking 完全分离，是否有界。
- 证据链是否从记忆版本回到真实事件/决策/知识/方案版本，是否拒绝不存在的来源。
- 仿真 60x 和视频 1x/4x/10x 是否没有混用。
- UI 是否没有 mock 后端数据，是否在真实 FastAPI + SQLite 下通过空、错误、降级和成功状态。
- 当前用户未提交文件是否未被覆盖，完整 8 小时测试是否仍只在后续显式执行。
