# 世界模型公开数据契约修复与交接（2026-09-05）

## 结论与范围

本轮完成世界模型负责的公开观测、估计、预测和事件之间的来源校验与失效处理，可提交个人模块复核；**不代表整个任务流程或联合验收已经通过**。

基线为 `master@2c444bb347c17fa25aa7ac136c70d3a234bf5fc6`，工作分支为 `fix/world-model-data-contract-20260905`。验收依据是本仓库 `docs/three-uuv-tracking-team-remediation-and-acceptance.md` 的 W-01～W-04，以及 `docs/three-uuv-tracking-backend-state-machine-overview.md`。主场景保持 **12 艘 UUV、4 个三艇组、一个目标**。未修改控制状态机、接管/交接门限、任务区方案或前端布局；少量跨层改动只用于贯通字段和拒绝失效数据。

通俗地说：过去同一份旧测量可能不断被盖上“刚更新”的章；现在区分“什么时候测到”“估计推算到了什么时候”和“画面什么时候刷新”。没有新测量，可以有限外推，但不能伪装成新的测量；过期后给出“无法判断”，不能说“一切正常”。

本轮由 Codex 本地执行，没有使用 DeepSeek，没有训练、模型下载或共享环境修改。世界模型仍为 `rule_demo（按规则推演的演示模型）`，`control_authority=false（没有控制权）`，不是经过训练的世界模型，也没有证明规则分数等于真实事件发生概率。

## 1. 修改了什么

| 条目 | 本轮实现 | 主要文件 |
| --- | --- | --- |
| W-01：持续且真实的公开估计 | 合格观测去重；拒绝旧时序、未来、无效与异常观测；按观测时刻更新；无新观测只外推，不更新来源版本和最后观测时间 | `groups/nodes.py`、`groups/state.py`、`tracking/public_estimate.py` |
| W-02：可审计的区域概率 | 沿用二维高斯区域积分，增加来源、时钟、协方差健康、有效期与确认资格；无法计算返回 `null`，不冒充概率 0 | `tracking/region_probability.py`、`simulation/engine.py` |
| W-03：同一帧、同一预测、当前执行上下文 | 预测、事件贯通来源版本；发布前按当前执行快照重算只读事件；校验 owner、区域、几何版本；过期/混版事件撤下 | `world_model/adapter.py`、`world_model/rules.py`、`agent/runtime.py`、`api/frame_builder.py` |
| W-04：公开与评估分离 | 同一公开帧经 HTTP、WebSocket、JSONL、Replay 保持一致；固定公开输入改变评估真值，公开结果与捕获的意图分析输入不变 | `tests/world_model/test_truth_transport_contract.py` |

表中文件路径均相对于 `src/underwater_tracking/`，测试路径除外。

### 1.1 观测融合

- 主动和被动观测都经过统一 `BearingObservation（方向测量）` 输入。
- 增加 `observer_position_xy（测量发生时传感器的位置）`，避免延迟到达的角度与 UUV 后来的位置错误配对。实际仿真生产端已经填写；旧数据没有此字段时仍兼容当前成员位置，不能据此证明异步真实传感器定位正确。
- 三角定位初始化使用同一时刻的角度；初始化用过的观测不再重复进入滤波更新。不会用零时长预测人为增加过程噪声。
- 非初始化更新沿用现有创新检验门限 `NIS <= 6.635（测量与预测的偏差是否可接受）`。至少一个 IMM 子模型接受，才计为新的来源证据。此条件不是事件概率校准。
- 记录已处理观测 ID（有界保留 4096 条）与来源观测 ID（有界保留 256 条）；只保留有界历史，不承诺对任意长时间、伪造时间戳的重放永久去重。
- UUV-only 流程去除同周期重复融合；没有观测时仍推进已有估计的状态时间，保留最后真实观测的时间。

### 1.2 三种时间和两个版本

| 字段 | 含义与约束 |
| --- | --- |
| `sim_time_s / state_time_s` | 状态被推算到的时间；可以随仿真前进，不代表新测量 |
| `last_observed_at_s` | 最近一次被接受的真实公开观测时间；不能被画面刷新覆盖 |
| `generated_at_s` | 当前预测或事件这份结果的生成时间；不是最近测量时间 |
| `valid_until_s` | 这份输入/结果还能使用到何时；达到该时刻即失效 |
| `track_revision / source_track_revision` | 合格新观测推动的估计版本及其引用；一个周期多条观测可只推动一次版本 |
| `prediction_revision` | 预测版本；允许无新观测时更新外推，但它仍引用旧的 source track revision |
| `accepted_observation_ids_this_cycle` | 仅本轮新接受的观测；不能拿累计历史代替本轮证据 |

估计 TTL（有效时长）取现有 `prediction_health.hard_stale_s`，本场景为 900 秒；执行候选同时受原有 450 秒有效期及各来源有效期约束，取更早到期者，不借重新发布续期。`source_track_age_s = 当前帧时间 - last_observed_at_s`，不知道最近观测时间时返回 `null`。

### 1.3 健康状态与区域概率

`assess_public_estimate` 检查来源元数据、时间先后、有限数值以及位置协方差的对称性和正定性。

- `current`：当前时刻有有效的观测来源。
- `degraded`：来源仍在有效期内，但已有数据年龄，当前结果属于外推。
- `expired`：有效期已到。
- `unavailable`：缺少可靠来源、时间不合理、协方差无效或无法计算。

`public_region_probability` 返回上述健康信息、`probability`、`polygon_xy` 和 `eligible_for_confirmation（是否具备本轮新证据的确认资格）`。世界模型只提供输入，**不修改**控制层的概率门限、连续计数或状态切换。

例如同一份 revision=8 的估计连续刷新三次，即使区域概率都为 0.8，也不能被说成三次独立测量。控制负责人需在消费端检查确认资格，并按不同来源版本/观测周期去重。当前 `mission_controller.py` 仍消费旧数值接口，尚未消费本轮新增的资格字段；这一点列为明确的联调项，不宣称 W-02 的“成功入区接管”端到端验收通过。

### 1.4 预测、事件与前端

1. 公开估计进入现有 IMM / B-spline 预测器，预测记录来源版本与有效期。
2. 世界模型优先读取本帧权威执行快照中的预测曲线与当前 owner、任务区。无执行快照的兼容路径读取已接受的预测。
3. 若执行快照提供 B-spline，则使用实际发布的 B-spline 点；否则使用 IMM 或明确标注的短历史/边界恢复结果，不把所有来源都写成 B-spline。
4. 规则输出当前条件下的事件假设，并给每条事件附上版本、时钟、owner、区域和来源组。
5. 发布前统一校验预测与事件身份及有效期。旧 owner 的卡片不能混入新 owner 的同一帧；事件内部元数据与外层结果不一致也拒绝。
6. HTTP、WebSocket、落盘和回放共用这一公开帧。前端仅补充空值兼容和“过期/暂不可用”的文案，不改布局或重算规则。

注意：`source_group_id（提供观测的小组）` 与 `owner_group_id（当前负责跟踪的小组）` 不一定相同。没有 owner 时不借用旧计划伪造当前归属。执行快照没有带时间的 UUV 未来航迹，因此覆盖/几何类推演仍使用当前公开航向、速度外推，并明确告警；同步仿真帧的 `state_time_s` 有依据，不等于已经支持真实通信延迟。

## 2. 验证结果与证据

本仓库 `docs/verification/world-model-contract-20260905/` 保存便于复核的摘要；原始 JUnit 位于本地交接包的 `evidence/`。最终数值以 `gate-summary.json` 和对应原始测试报告为准。

- 专项回归：`tests/groups tests/tracking tests/world_model`，106 项通过，包括初始化只使用一次、观测时刻位置、重复/乱序/过期/恢复、概率空值、owner 变更、预测混版及真值隔离。
- 前端：143 项通过；TypeScript 类型检查和 Vite 生产构建通过。未执行浏览器实景/视频验收。
- 完整回归筛选：2275 通过、2 失败、46 跳过、25 未选入，耗时 273.57 秒。排除 `real_llm / long_running / live_acceptance` 标记，以及主分支缺少驱动的两个 acceptance 文件；两个失败已在干净 master 相同环境复现，均属本地记忆提供器。**这不是“全仓库所有测试全绿”。**
- 两次相同 seed=42、正常 5 秒物理步长、每次 3600 仿真秒复跑；真实运行引擎和公开融合链路，LLM 使用确定性测试替身，没有注入目标路线、观测、owner 或门限。两次公开摘要一致，接受 9 条来源观测、最大估计版本 8，状态经历不可用→降级→过期，正常关闭，无 carrier 异常。
- 该复跑没有出现首次 owner 或交接。出现的“异常低速”是同一类规则卡片在多帧中的重复出现，不是多次真实事故，也不是预测准确率。没有做事件标签对齐、校准或真实潜艇事件验证。
- 新增/改动行 Ruff 检查无新增告警；相关旧文件仍有 25 项存量告警，未扩大修改范围。

### 2.1 可重复命令

在分支根目录、已配置项目依赖的 Python 环境执行（没有训练）：

```powershell
python -m pytest -q tests/groups tests/tracking tests/world_model --junitxml=contracts.xml
python -m pytest -q -m "not real_llm and not long_running and not live_acceptance" --ignore=tests/acceptance/test_default_live_acceptance.py --ignore=tests/acceptance/test_default_live_acceptance_driver.py --junitxml=regression.xml
python scripts/run_world_model_contract_audit.py --output audit-results-new --seconds 3600 --seed 42
```

复跑脚本依赖仓库中的 `tests.integration.test_uuv_only_production_acceptance.FixedSeedUUVLLM`，需保留 tests 目录；它是离线诊断入口，不是正式 `main.py` 验收替代。输出目录必须不存在，避免覆盖旧证据。前端在 `src/underwater_tracking/ui` 运行 `npm test` 和 `npm run build`。

当前环境的两个记忆相关失败为：

```text
tests/agent/test_runtime_master_slave_adversary.py::test_agent_loop_uses_real_memory_provider_chain_when_configured
tests/agent/test_runtime_master_slave_adversary.py::test_local_memory_provider_does_not_require_embedding_api_key
```

两项测试期待 SentenceTransformer 提供器，当前环境降级为替代检索器；基线同样失败。本轮没有安装额外模型或改动全局依赖。另两个 acceptance 文件导入的 `tools.run_default_live_acceptance` 不在基线 Git 树中，最终联合验收需先由主线补齐正式驱动。

测试维护说明：旧夹具补齐显式来源字段；旧 8 小时稳定性测试改为检查过期后预测/事件撤下，而不是强求一直存在预测。没有降低安全距离、概率门限或确认次数。曾遇到一次 Showcase WebSocket 退出时 `CancelledError`，独立复跑通过；保留该次记录，不作为预测逻辑修复成果。

## 3. 下一轮联调清单

1. UUV 控制负责人：消费 `region_probability_evidence`，按新观测周期/来源版本累计确认；确认高概率但不可用/无新证据时的阻塞原因。不要直接累计本轮新增的历史来源 ID。
2. LLM 与控制负责人：共同闭合持续产生候选→准入→更新执行快照的链路。世界模型拒绝过期来源后，不得用旧快照重新盖章规避。
3. 前端负责人：按新增健康状态、空值和执行信息展示，本轮仅做最低兼容；正式演示仍需浏览器联调。
4. 世界模型跟进：消费其他负责人合并后的接口，复查来源版本、事件有效期和 owner/区域一致性；不接管他们的策略实现。
5. 团队联合验收：补齐普通 `main.py` 驱动，使用实际 provider、正常物理步长，观察首次接管和真实交接/替换/专属模式。未出现的流程不能用强制夹具补成 PASS。

允许评审本分支；禁止直接宣称联合验收完成、真实预测准确率达标、已训练世界模型，或将本分支自动合并 `master`。
