# Underwater Tracking 水下目标跟踪指挥系统

确定性多 UUV 纯方位(仅测向)目标跟踪基础平台:无头仿真循环驱动群组跟踪、轨迹预测、
规划与任务分配,LLM 智能体(LangGraph 载体)在闭环中做高层决策,前端提供实时/回放的
命令中心 Web 界面。相同种子、相同配置,运行结果逐字节一致。

![Python](https://img.shields.io/badge/Python-3.11--3.12-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-agent-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-teal)
![React](https://img.shields.io/badge/React-18%20%2B%20Vite-blueviolet)

## 特性

- **确定性可复现**:相同 seed + 配置,帧日志逐字节一致,便于实验对照与论文复现
- **真值安全**:运行帧(`OperationalFrame`)绝不包含目标真值;真值只经引擎评估通道输出,
  杜绝"用真值做决策"的泄漏
- **完整算法链路**:仿真 → 被动测向 → IMM/UIF 跟踪 → 群组管理 → B 样条预测 →
  FIM 规划与分配 → LLM 智能体决策,一条 `main.py` 跑通
- **LangGraph 信息闭环**:"观测 → 事件 → 决策 → 执行 → 再观测"持续闭环,
  状态可检查点化、可断点续跑,相同输入逐字节一致
- **智能体闭环**:LangGraph 状态机载体,LLM 故障自动降级(出错计数、群组循环不中断),
  计划命令在下一观测周期回流到群组管理器
- **命令中心**:FastAPI/WebSocket 实时推送 + React Canvas 态势地图,支持回放、指令
  下发、任务指派与提问端口
- **对抗要素**:主动声呐探测模型(能量代价、被截听概率)与诱饵漂移模型

## 快速开始

一条命令启动完整算法链路 + Web 命令中心,并打印界面地址:

```powershell
python main.py
```

```
Underwater tracking command center:
  Web UI:  http://127.0.0.1:5173
  API/WS:  http://127.0.0.1:8000  (docs: http://127.0.0.1:8000/docs)
```

`main.py` 会同时拉起后端(FastAPI + 仿真线程)与前端(Vite dev server),
`Ctrl+C` 干净退出。可选参数:

```powershell
python main.py --config configs/scenario/default.yaml --steps 100 --seed 42 --host 127.0.0.1 --port 8000
```

`--steps 0`(默认)表示运行直到 `Ctrl+C`;`--steps N` 跑满 N 步后后端仍持续服务。

前置条件:

1. Python 3.11 或 3.12 及已安装的后端依赖(`pip install -e ".[dev]"` 即可;`main.py`
   本身不要求 editable install,会自动处理 `src/` 路径)
2. Node.js(用于前端),安装依赖:`npm --prefix src/underwater_tracking/ui install`
3. LLM API key:配置到 `configs/.env`(git 忽略)或环境变量 `UNDERWATER_TRACKING_API_KEY`

## 安装

```powershell
python -m pip install -e ".[dev]"
npm --prefix src/underwater_tracking/ui install
```

要求 Python 3.11 或 3.12。

## 算法链路

```
┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌─────────────┐
│  仿真引擎   │──▶│  跟踪       │──▶│  群组       │──▶│  预测       │──▶│  规划/分配  │──▶│ LLM 智能体   │
│ 纯方位观测  │   │ IMM + UIF  │   │ 组图/报告   │   │ B 样条     │   │ FIM + 航路  │   │ LangGraph   │
└────────────┘   └────────────┘   └────────────┘   └────────────┘   └────────────┘   └─────────────┘
       ▲                                                                                  │
       └──────────────────────────── 计划命令回流(下一观测周期) ───────────────────────────┘
```

| 阶段 | 模块 | 说明 |
| --- | --- | --- |
| 仿真 | `simulation/` | 确定性世界推进:UUV 编队、目标潜艇(巡航/冲刺)、诱饵漂移、主动声呐探测、纯方位观测生成;观测帧不含真值 |
| 跟踪 | `tracking/` | 交互多模型(IMM)转弯率模型库 + 尺度无迹信息滤波(UIF,信息形式纯方位量测)、航迹初始化、跟踪质量 EWMA 评估 |
| 群组 | `groups/` | 每目标一个 LangGraph 群组状态机,产出群组报告驱动载体周期 |
| 预测 | `prediction/` | 协方差加权三次平滑 B 样条航迹预测,输出快照预测器供智能体使用 |
| 规划 | `planning/` | 纯方位 Fisher 信息(FIM)度量、鲁棒滚动时域航路点规划、UUV 分配、区域预留、计划校验 |
| 智能体 | `agent/` | LangGraph 载体状态机:事件监控、策略生成、方案优化、主动验证协议、提交决策;LLM 走 OpenAI 兼容 LongCat 接口 |
| 持久化 | `persistence/` | 每次运行一个 SQLite:计划库、事件库、决策台账、LangGraph checkpoint |
| 传输 | `api/` | FastAPI + WebSocket 实时帧推送、回放、指令/指派/提问端口;`ui/` 为 React 18 + Vite + Canvas 2D 命令中心 |

## LangGraph 信息闭环

整个系统以 LangGraph 载体图为核心,构成"观测 → 事件 → 决策 → 执行 → 再观测"的
持续信息闭环。每个观测周期(默认 30 s)结束、群组报告产出后,引擎调用载体钩子,
`runtime.tick()` 推进一个完整的图循环:

```
仿真观测 → 群组报告 → 事件监控(三级分类) → 规划快照 → 分级路由 → 计划提交
    ▲                                                                    │
    └────────── 计划命令/验证命令回流(下一观测周期生效) ◀─────────────────┘
```

闭环的关键机制:

- **引用而非原始数据**:高频原始状态(群组报告、资源状态)不写入 checkpoint,
  以确定性引用寻址——实时情况 `{scenario_id}:live`、不可变规划快照
  `{scenario_id}:snapshot:{revision}`、候选计划 `{scenario_id}:candidate:{index}:{revision}`;
  checkpoint 只持久化状态通道,大载荷存注入的 store
- **不可变规划快照**:每次规划基于自包含的 `PlanningSnapshot`(实时情况 + 待处理事件 +
  当前广播计划 + 已应用指令),后续验证/提交只对照该快照,版本号一致才允许提交
- **一体化持久化**:计划库、事件库、决策台账、LangGraph checkpoint 同在本次运行的
  SQLite 中;`resume` 可不推进时钟地重放一个图循环,支持断点续跑与事后审计
- **确定性可重放**:相同 seed + 配置,图循环的每步输出逐字节一致,便于实验对照
- **错误降级不中断**:节点错误路由到 `handle_error` 记录后正常结束循环;载体异常
  仅计数,群组跟踪循环永不中断

## 智能助手执行逻辑

智能助手以两条互补路径对跟踪方案做动态调整:**事件触发机制**(自主路径)和
**人机交互**(专家回路)。

### 事件触发机制

`EventMonitor` 把每次观测分类为三个层级,按层级执行不同链路:

| 事件层级 | 触发条件(示例) | 执行链路 | 是否调用 LLM |
| --- | --- | --- | --- |
| STRATEGIC 战略 | 初始化、目标新增/移除/丢失、意图变化确认、重大故障、不可行修复、专家指令已应用 | 意图分析 → 轨迹预测 → 策略生成(三个候选概念)→ 验证子图 → 资源优化 → 计划校验 → 提交 | 是 |
| TACTICAL 战术 | 跟踪质量预警、几何退化、电池轮换 | 预测 + 确定性优化(沿用现有策略,不重规划) | 否 |
| INFORMATIONAL 信息 | 常规报告、专家提问 | 记录并报告 | 否 |

分级规则由配置门限驱动:

- **质量事件需持续触发**:Q_EWMA < 0.65 持续 2 分钟才发预警、< 0.40 持续 30 秒
  才发临界事件(`agent.yaml` 的 `quality_*_persist_s`)
- **同类事件合并**:同一实体同类型事件在冷却窗口(300 s)内去重
- **意图变化双重确认**:意图分析置信度 ≥ 0.70、领先次优 ≥ 0.15、且连续两次分析
  一致,才确认为意图变化
- **策略三候选**:战略事件生成 `quality_first` / `balanced` / `resource_saving`
  三个候选概念,由验证子图裁定;战术事件只生成 `hold_current`
- **主动验证协议**:对可疑接触启动确定性声呐验证状态机
  `idle → verifying(空闲 UUV ping)→ classified_submarine → in_position(几何门限)`,
  验证命令同样回流引擎生效

### 人机交互(专家回路)

| 通道 | 流程 | 对方案的影响 |
| --- | --- | --- |
| 专家指令 | 前端/API 提交自然语言 → LLM 解析为结构化指令 → 校验 → 结构化预览 → 专家确认应用 | 作为**硬约束**进入下个周期的战略重规划;当前计划保持生效直至新计划提交 |
| 专家提问 | 只读查询,答案取自不可变规划快照,带证据链引用 | 无影响;反事实推演在克隆快照上 `dry-run:`,绝不触碰在线计划 |
| 任务指派 | 前端/API 提交目标-UUV 指派,携带期望计划版本号 | 版本不匹配返回 409,防止基于过期方案做决策 |

所有人工入口都遵循**版本化计划**约束:每个已提交计划带版本号,指令/指派必须携带
期望版本,方案在编辑期间被更新时要求人工复核,避免"对着旧方案下指令"。指令、提问、
指派分别经 `/api/directives`、`/api/questions`、`/api/assignments` 进入,前端命令中心
(右侧栏/助手抽屉)提供同样的操作入口。

## 使用方式

除 `main.py` 一键入口外,也保留细粒度 CLI:

**无头仿真**(基础算法,不调 LLM):

```powershell
python -m underwater_tracking.cli simulate --config configs/scenario/default.yaml --steps 360 --seed 42
```

**智能体运行**(完整链路 + LLM 决策,落盘 manifest 与帧日志):

```powershell
python -m underwater_tracking.cli agent-run --config configs/scenario/default.yaml --steps 540 --seed 42
```

**命令中心服务**(LangGraph 运行时 + FastAPI/WebSocket;另开终端起前端):

```powershell
python -m underwater_tracking.cli serve --config configs/scenario/default.yaml --seed 42
npm --prefix src/underwater_tracking/ui run dev
```

## 项目结构

```
├── main.py                 # 一键入口:完整链路 + Web 界面
├── configs/
│   ├── scenario/default.yaml   # 场景与节拍配置
│   ├── tracking.yaml           # 跟踪/群组/传感器/运动模型参数
│   ├── agent.yaml              # 智能体行为参数
│   └── llm.yaml                # LongCat 提供商配置(密钥仅环境变量或 .env)
├── src/underwater_tracking/
│   ├── cli.py              # simulate / agent-run / serve 命令
│   ├── simulation/         # 确定性仿真引擎
│   ├── tracking/           # IMM、UIF、初始化、质量评估
│   ├── groups/             # 群组状态机与管理器
│   ├── prediction/         # B 样条航迹预测
│   ├── planning/           # FIM、航路点、分配、预留、校验
│   ├── agent/              # LangGraph 载体(图、节点、LLM 客户端、运行时)
│   ├── domain/             # 领域模型(真值/运行帧/UI 视图)
│   ├── persistence/        # SQLite 计划/事件/台账/checkpoint
│   ├── api/                # FastAPI 应用、帧构建、日志、回放、中心
│   └── ui/                 # React 命令中心(Vite dev / build)
├── tests/                  # 单元/集成/性质/端到端测试
├── assets/                 # 场景素材(复制到 ui/public 使用)
└── docs/superpowers/       # 设计规格、实施计划、审计
```

## 配置说明

### `configs/scenario/default.yaml` — 场景与节拍

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `scenario.uuv_count` | 12 | UUV 数量 |
| `scenario.initial_target_count` | 2 | 初始目标数 |
| `scenario.max_target_count` | 4 | 最大目标数 |
| `scenario.duration_s` | 28800 | 场景时长(8 小时) |
| `scenario.seed` | 42 | 确定性种子 |
| `scenario.initial_decoy_count` | 0 | 初始诱饵数 |
| `timing.physics_step_s` | 10 | 物理步长 |
| `timing.observation_step_s` | 30 | 观测周期 |
| `timing.group_report_s` | 300 | 群组报告周期 |
| `timing.prediction_horizon_s` | 1800 | 预测时域 |

### `configs/tracking.yaml` — 跟踪与传感器

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `group_min_size` / `group_max_size` | 2 / 4 | 群组规模上下限 |
| `quality_warning` / `quality_critical` / `quality_release` | 0.65 / 0.40 / 0.75 | 跟踪质量 EWMA 阈值 |
| `quality_window_s` / `release_hold_s` | 300 / 600 | 质量窗口 / 释放保持 |
| `covariance_reference_m2` | 10000 | 协方差基准 |
| `fim_min_eigenvalue_reference` / `fim_condition_reference` | 0.001 / 100 | FIM 度量基准 |
| `uuv_max_speed_mps` / `uuv_max_turn_rate_rad_s` | 4.0 / π/60 | UUV 运动约束 |
| `submarine_cruise_speed_mps` / `submarine_sprint_speed_mps` | 8.0 / 14.0 | 潜艇巡航/冲刺速度 |
| `sensor_active_range_m` | 3000 | 主动声呐作用距离 |
| `sensor_ping_interval_s` / `sensor_ping_energy_cost` | 30 / 0.0002 | 声呐 ping 周期/能量代价 |
| `sensor_ping_heard_probability` | 0.6 | 被截听概率 |
| `decoy_drift_speed_mps` | 0.5 | 诱饵漂移速度 |

### `configs/agent.yaml` — 智能体行为

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `transport_retries` | 3 | 传输失败指数退避重试次数 |
| `semantic_repairs` | 2 | LLM 输出语义修复最大轮数 |
| `quality_warning_persist_s` / `quality_critical_persist_s` | 120 / 30 | 质量事件触发需持续的时长 |
| `event_cooldown_s` | 300 | 同类事件合并窗口 |
| `history_token_threshold` | 6000 | 触发历史压缩的 token 估算 |
| `intent_change_confirmation` | 0.70 / 0.15 / 2 | 意图切换确认门限(置信度/领先幅度/连续次数) |

### `configs/llm.yaml` — 提供商

OpenAI 兼容的 LongCat 接口(`model: LongCat-2.0`、`base_url: https://api.longcat.chat/openai/v1`)、
超时/重试/退避参数,接口文档:https://longcat.chat/platform/docs/zh/。
**API key 只来自环境变量 `UNDERWATER_TRACKING_API_KEY` 或
git 忽略的 `configs/.env`(环境变量优先),绝不提交到仓库**;算法运行时
`load_app_config` 自动加载 `.env`。无 key 时
`agent-run` / `serve` / `main.py` 会在启动时明确报错退出。

## 输出

每次运行写入 `outputs/run-<seed>-<id>/`(`serve`/`main.py` 为 `outputs/serve-<seed>-<id>/`,
目录已 git 忽略):

- `frames.jsonl` — 每仿真步一行运行帧(不含真值)
- `operational_frames.jsonl` — 经智能体链路发布的操作帧
- `manifest.json` — 运行摘要:carrier 错误数、决策数、LLM 调用数、活跃计划等
- `agent.db` — 计划/事件/台账/checkpoint 的 SQLite

相同 seed 与配置重跑,日志逐字节一致(run id 与输出路径除外)。

## 测试与验证

```powershell
python -m pytest -q
```

基础门禁(全部应退出码 0):

```powershell
python -m pytest tests/config tests/domain tests/simulation tests/tracking tests/prediction tests/planning tests/groups tests/integration tests/property -q
python -m ruff check src/underwater_tracking tests main.py
python -m mypy src/underwater_tracking
python -m underwater_tracking.cli simulate --config configs/scenario/default.yaml --steps 360 --seed 42
```

## 技术栈

| 层 | 技术 |
| --- | --- |
| 语言/运行时 | Python 3.11–3.12、Node.js ≥ 20 |
| 数值计算 | NumPy、SciPy |
| 智能体 | LangGraph(状态机图、SQLite checkpoint)、OpenAI 兼容 LongCat 接口 |
| 服务 | FastAPI、Uvicorn、WebSocket |
| 前端 | React 18、TypeScript、Vite、Canvas 2D |
| 数据 | SQLite、Pydantic v2、PyYAML |
| 质量 | pytest、Hypothesis(性质测试)、ruff、mypy(strict)、Playwright(端到端) |
