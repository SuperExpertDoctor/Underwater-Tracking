# 前端 mock 验收

mock 只在 UI 层运行，使用固定 `three-uuv-mock-seed-42` 帧序列，不连接 WebSocket，也不会调用控制接口。

## 启动

在仓库根目录执行：

```powershell
cd src\underwater_tracking\ui
npm install
npm run dev
```

打开 Vite 输出的地址并追加 `?mock=1`，例如：

```text
http://localhost:5173/?mock=1
```

也可以通过环境变量启用（不需要 query 参数）：

```powershell
$env:VITE_MOCK_MODE = "true"
npm run dev
```

## 验收路径

- 默认“实时”会每 1.2 秒循环 8 个帧：R01 路线完成但主动覆盖只有 25% 且 ping 为 3；随后出现入区 `1/2 → 2/2`、被动 owner、交接 `2/3 → 3/3`。
- 切换到“回放”后，可用底部播放条逐帧检查同一批数据；第 4/5 帧展示替补 `OUT → IN`、几何修订和旧组 `EXITING` 保留，下一帧才 `DISAPPEARED`。
- 最后一帧故意让目标估计超过 `valid_until=300s`：右侧固定目标状态板显示 `EXPIRED / 已过期`、等待刷新；仿真时间仍继续，地图上的旧目标/走廊变暗并使用虚线/冻结标记。
- 右侧“区域任务审查”和区域时间线分别显示路线进度、source-backed ping、主动覆盖与扫描完成状态；缺字段时显示“不可用”，不会从路线、航迹或编组数量推导扫描完成。

mock 横幅明确标记“仅前端契约演示，不参与在线控制”。如需验收真值，必须显式设置 `VITE_EVALUATION_ENABLED=true` 使用评估模式；不要把真值字段加入普通 `OperationalFrame`。

