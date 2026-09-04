# 2026-08-30 世界模型实时链路与 UUV 控制审计

## 结论先行

本轮以远端 `master` 的 `791ade4235247781b16462d88a2bcc847912c7eb` 为基线，工作在隔离分支
`audit/world-model-uuv-control-20260830`，没有修改 `master`，也没有把当前主场景改回历史上的
3-UUV 演示。

当前主场景保持为：12 艘 UUV、1 个目标、4 个双 UUV 任务组，其中 8 艘执行任务、4 艘作为
各任务组的储备资源。世界模型仍然是只读模块，只计算并展示未来事件，不向 UUV 下控制命令。

本轮完成了三类工作：

1. 世界模型接入当前执行快照与连续两帧公开接触信息；
2. 修正 UUV 航迹中“终点安全但途中相交”、编队成员交换位置、急转弯不减速、缺少地图与禁行区
   保护等问题；
3. 把 300 米间距从一个没有真正验收的描述值，改为按当前任务组成员动态检查的硬性门槛。

在同一 Python 依赖环境下，30 分钟确定性仿真重复两次并得到相同摘要：最终状态为 `PASS`，
物理违规为 0，四个任务组在建立编队间距后均未跌破 300 米。跟踪位置误差的 RMSE 从
3101.64 米降至 455.82 米，下降 85.3%。

## 1. 范围与边界

### 1.1 本轮负责的范围

- 世界模型使用公开态势、IMM 结果和 B-spline 未来轨迹，生成规则式未来事件预测；
- 世界模型结果进入运行时状态，供前端读取；
- UUV 仍采用“给航点—到航点”的执行形式；
- 检查并修正航点规划、编队重新散开、UUV 动力学执行和地图安全边界；
- 建立可重复的 30 分钟双跑验收。

### 1.2 明确不做的事项

- 不改变主分支的 12-UUV、4 个双 UUV 任务组设定；
- 不让世界模型控制 UUV，所有预测结果的 `control_authority` 仍为 `false`；
- 不读取目标真值来做在线规划；真值只进入离线验收误差计算；
- 不训练模型。本轮全部是确定性规则、几何规划和动力学约束，因此没有训练登记；
- 不为了让现有失败测试变绿而修改主任务生命周期。

## 2. 世界模型实时链路

### 2.1 修改前的问题

原有规则世界模型已经可以根据一组输入计算未来事件，但在线链路存在以下断点：

- 上一帧目标数量没有进入规则，无法可靠判断“新目标或诱饵突然出现”；
- 目标关联置信度和关联混乱程度没有来自实时公开数据；
- 世界模型记录的计划版本来自旧的 `TrackingPlan`，而当前主流程以
  `OperationalExecutionSnapshot` 为权威执行快照，两者可能不同步；
- 测试只证明函数可以运行，没有证明在线运行时把连续帧历史和当前执行版本传了进去。

### 2.2 当前实现

新增 `ContactAssociationSnapshot`，每个目标保存三项公开信息：

- `contact_count`：当前公开接触数量；
- `target_confidence`：目标接触在全部公开接触证据中所占的权重；
- `normalized_entropy`：证据在多个接触之间分散得有多均匀，越接近 1 表示越难区分。

权重来自公开观测：有声呐方位线时使用公开的探测置信度；没有方位线时，按公开分类使用固定且
可解释的权重。该值只是规则算法的“关联混乱代理量”，不是训练得到的概率，前端和文档不得把它
描述成真实概率。

`CarrierRuntime` 现在保存上一帧的公开接触摘要，并在下一帧计算时一并传入规则。世界模型因此能
比较“上一帧只有一个可靠目标”和“这一帧新增一个未确认接触”，从而产生
`DECOY_OR_NEW_CONTACT_AMBIGUITY` 等事件。

世界模型的 `source_plan_revision` 优先使用当前
`OperationalExecutionSnapshot.execution_revision`。只有没有执行快照时，才退回旧计划版本。

### 2.3 仍然保留的限制

- `SituationSnapshot.uuvs` 是同一快照中的当前状态，但单个 `UUVState` 没有独立时间戳，因此
  `state_age_s` 仍只能填写 0。不能凭空编造每艘 UUV 的状态延迟；后续如要启用“状态过旧”规则，
  应先在公开数据契约中增加每艘 UUV 的观测时间。
- 当前执行快照没有把可执行 UUV 航迹直接暴露给世界模型；旧 `TrackingPlan` 也不一定带航点。
  因此世界模型在没有计划轨迹时会明确降级为按当前位置和速度外推，不伪造未来 UUV 路线。
- 该模块仍是规则展示版，不等同于已经训练并校准过的学习型世界模型。

## 3. UUV 控制与规划修正

### 3.1 两艘 UUV 分配不同扫描线

执行快照转任务计划时，原代码把同一个区域多边形直接交给同组两艘 UUV。两艘 UUV 因而可能
收到相同路线，观测角度接近，既影响定位，也容易在同一路线上靠得过近。

当前实现使用已有的蛇形覆盖算法为同组成员分配不同扫描线。主任务区、任务组、角色和生命周期
均保持不变，只改变组内航迹的具体走法。

### 3.2 检查整段路线，而不只检查终点

新增连续航段安全检查：

- 计算两艘 UUV 从当前点同时前往下一航点时的最小距离；
- 终点相隔 300 米但中途交叉的方案会被拒绝；
- 初始部署点较近时允许向外散开，但距离必须从一开始就不再缩小，并在终点建立规定间距；
- 通用航点规划器、双 UUV 任务组规划器和修复候选都使用同一检查方法。

### 3.3 修正“互换座位”导致的迎面穿越

跟踪质量不稳定时，编队会在当前中心周围重新散开。原实现按角度排序后固定分配圆环位置；角度
在 `-π` 和 `π` 的边界处会因为很小的数值差异翻转，出现左侧 UUV 去右侧、右侧 UUV 去左侧的
“互换座位”命令。

当前实现枚举最多 4 个编队成员的圆环位置分配，只保留整段路线满足组内间距的方案，再选择总
移动距离最短的方案。如果边界裁剪后不存在安全方案，则保持当前位置，不发出已知会相交的命令。

### 3.4 航点前减速和转弯速度约束

UUV 到达一个航点后，如果下一个航点在身后，原实现可能以较高速度冲过航点，再画大圆返回。
当前速度上限同时考虑：

- 距离航点还有多远；
- 按最大减速度是否来得及停下；
- 当前朝向与目标方向的夹角。

这仍是“给点—到点”控制，只是把到点过程改得更符合有限转向率和有限加减速度的动力学约束。

### 3.5 地图边界与禁行区

UUV 运动层现在接收主场景的地图边界和导航禁行多边形。命令执行前会按停止距离、最小转弯半径
和安全余量提前减速转向；每个实际运动段完成后再次检查是否合法。无法保持合法时抛出明确的
`NavigationInvariantError`，不再静默穿越边界。

潜艇原有边界行为保持原兼容模式，新的提前保护只在 UUV 边界对象上开启，避免改变潜艇既有测试
和任务行为。

## 4. 验收口径

### 4.1 为什么不能使用“全局任意两艘 UUV 相隔 300 米”

300 米来自任务组航点规划约束。当前四个任务区域存在空间重叠，但主任务没有配置全舰队统一的
碰撞净空半径。把 300 米直接套在不同任务组之间，会把“组内编队间距”和“全局物理防碰距离”
混为一谈，也会擅自改变主任务设定。

因此当前口径为：

- 全舰队最小距离继续作为描述指标和风险提示；
- 每一帧读取当时真实的任务组成员，而不是只使用开局成员；
- 同组成员允许从共同部署边界向外散开；
- 一旦建立 300 米间距，后续任何帧都不得再次跌破；
- 没有观测到已分配成员，或始终没有建立规定间距，均判定失败。

### 4.2 30 分钟确定性双跑

统一条件：

- 场景：`configs/scenario/uuv_only_single_target.yaml`；
- 随机种子：42；
- 物理步长：5 秒；
- 步数：360，即 1800 秒；
- 重复次数：2；
- 禁止网络 LLM；
- 对照与最终版本均使用 NumPy 2.3.5，避免依赖版本造成不公平比较。

复现最终结果：

```powershell
.\.venv\Scripts\python.exe scripts/run_uuv_tracking_coverage_audit.py `
  --config configs/scenario/uuv_only_single_target.yaml `
  --seed 42 `
  --steps 360 `
  --repeat 2 `
  --work-dir outputs/audit-20260830-final-py311-compatible `
  --evidence-dir outputs/evidence-20260830-final-py311-compatible
```

两次最终摘要哈希均为：

`663a348b77b6892531ca9bf4e77db0154bc3076a4c8dfe76c0ff76a214747e08`

### 4.3 指标对比

| 指标 | `master` 基线 | 本轮最终 | 变化 |
| --- | ---: | ---: | ---: |
| 跟踪误差 RMSE | 3101.64 m | 455.82 m | 下降 85.3% |
| 跟踪误差中位数 | 3084.11 m | 479.65 m | 下降 84.4% |
| 跟踪误差 P95 | 5171.68 m | 620.44 m | 下降 88.0% |
| 最大跟踪误差 | 6272.05 m | 636.70 m | 下降 89.8% |
| 全舰队最小距离 | 0.00 m | 117.77 m | 描述指标，不作为 300 m 组内门槛 |
| 物理违规数 | 0 | 0 | 保持为 0 |

旧基线虽然也显示 `PASS`，但旧硬性检查中没有 UUV 间距项目，因此不能把旧 PASS 理解为“旧版
已经满足间距要求”。

最终四个任务组建立间距后的最小值：

| 任务组 | 成员 | 最小组内距离 |
| --- | --- | ---: |
| task:01 | uuv_00 / uuv_01 | 485.33 m |
| task:02 | uuv_02 / uuv_03 | 467.42 m |
| task:03 | uuv_04 / uuv_05 | 467.42 m |
| task:04 | uuv_06 / uuv_07 | 467.42 m |

最终硬性检查全部通过：真值隔离、融合估计可用、分配路线存在、路线几何合法、存在控制命令、
观察到实际运动、动态任务组间距合格、物理审计完整、指标有限、重复运行一致。

## 5. 测试情况

### 5.1 通过项

- 世界模型、规划、仿真、运行时、审计和相关 CLI 的扩大回归：754 项通过；
- Agent 测试排除两项已知本地记忆模型依赖问题后：484 项通过、44 项跳过；
- 其余模块排除一项实际 8 小时仿真后：769 项通过、3 项跳过；
- 审计模块新增动态成员检查测试：通过；
- 修改涉及的 6 个核心源文件按项目 Python 3.11 目标执行严格 mypy：通过；
- 针对本轮文件执行 Ruff 检查，在明确排除同文件中 13 条既有告警后：通过；
- `git diff --check`：没有新增空白错误。

把全部改动源文件一次性交给严格 mypy 时仍会报告 59 条历史类型问题；相同命令在未修改的
`master` 上报告 60 条。本轮修正了其中 1 条 `dict_values` 与 `Sequence` 不兼容问题，没有把其余
历史债务混入本轮范围。Ruff 默认检查显示的 13 条告警也能在本轮未改动的旧代码段中定位到。

### 5.2 已知基线问题

完整测试在当前本地环境中仍有三类现存问题：

1. 两项本地记忆测试期望 `SentenceTransformerEmbeddingProvider`，但审计环境没有安装对应本地
   模型依赖，运行时按设计降级为 `DegradedMemoryRetriever`。修改前后失败相同。
2. `test_real_uuv_default_timeline_local_perception_and_periodic_memory` 期望 3400 秒内出现边界退出，
   但当前 `master` 和本分支都会先把区域置为 `DEGRADED`，没有完成 `handoff_completed`，因此按
   主任务规则不会执行“交接后轮换”。该问题属于主分支生命周期流程，不是本轮控制修改的回归。
3. `test_tracking_pipeline_remains_bounded_and_executable_for_eight_hours` 实际运行时间很长，却没有
   标记为 `long_running`。本轮普通回归在该项等待后中止，并在排除它后完成其余测试。建议后续
   把它纳入专门的长任务测试登记或 CI 队列。

## 6. 当前风险与后续路线

1. 全舰队最小距离仍只有 117.77 米，发生在不同任务组的 `uuv_02` 与 `uuv_05`。当前没有统一
   防碰净空配置，不能擅自用 300 米判失败。后续应由任务设计方给出 UUV 尺寸、定位误差和安全
   裕度，再建立全局防碰门槛与跨任务组协调器。
2. 世界模型的 UUV 状态年龄缺少数据源，应先扩展公开状态时间戳，再启用过期状态事件。
3. 世界模型目前没有权威的未来 UUV 执行航迹输入。后续前端若要展示“目标未来轨迹与 UUV 未来
   覆盖是否相交”，应把当前执行快照中的按 UUV 航迹作为只读契约发布。
4. 同一代码在不同 NumPy 版本下会出现末位浮点差异，虽然各自环境内重复运行完全一致，但摘要
   哈希不同。若要求跨机器字节级一致，应增加依赖锁文件或固定验收镜像；本轮未擅自修改依赖策略。
5. 主分支的区域降级—交接—轮换流程需要单独立项，不应混入本轮只读世界模型和 UUV 底层控制
   修正中。

## 7. 训练登记

本轮没有训练任务：未调整学习模型、未生成训练数据、未启动 GPU 作业，因此无需训练登记。

## 8. Three-UUV Runtime Acceptance (2026-09-04)

This addendum records the new UUV-only runtime contract. The historical
two-UUV-per-task-group plus four-reserve projection above is retained as an
audit baseline only; it is obsolete for the live UUV-only implementation and
does not drive the current state machine.

Source of truth:

- Design: `docs/superpowers/specs/2026-09-02-three-uuv-tracking-modes-design.md`
- Implementation plan: `docs/superpowers/plans/2026-09-02-three-uuv-tracking-modes-implementation-plan.md`
- Backend acceptance: `tests/acceptance/test_three_uuv_tracking_modes.py`
- Live visual acceptance: `src/underwater_tracking/ui/e2e/three-uuv-tracking-modes.spec.ts`

The authoritative execution contract was verified with four 2000 m square
regions, exactly three UUVs per task-group instance, a 1000 m target radius,
and 600 m active/passive UUV radius. The runtime sequence covered active scan,
passive tracking, regional ownership handoff, dedicated tracking, the 7000 m
remaining-mileage release threshold, regional restore, and parallel replacement.
Observed acceptance metrics were:

```text
region_side_m=2000.0
target_detection_radius_m=1000.0
uuv_detection_radius_m=600.0
task_group_size=3
max_coverage_gap_area_m2=0.0
active_ping_count_during_passive=0
tracking_owner_gap_frames=0
max_visible_uuv_count=24
```

The Python acceptance test passed. The real `main.py` WebSocket/replay
acceptance passed at 1440x900, 1280x720, and 390x844, including canvas pixel
checks, layout bounds, strict frame fields, event ordering, replay validation,
and browser console checks. The live run used the `underwater-tracking`
Python 3.11 environment and the explicit acceptance fixture; it did not inject
hand-authored frames into the browser.
