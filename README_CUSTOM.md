# Bench2Drive-VL 自定义场景运行手册

本文档记录在当前仓库中，使用自定义 OpenDRIVE 地图（.xodr）与 DriveStudio 轨迹数据，跑通 Bench2Drive-VL 闭环评测的完整步骤。

## 1. 目标与默认约定

- 工作目录：仓库根目录（必须在根目录执行命令）。
- CARLA：外部宿主机已启动，监听 127.0.0.1:2000。
- 地图文件：放在仓库根目录，命名为 <town>_local.xodr。
- 路线文件：仓库根目录 custom_route.xml。
- 严格控制：默认启用严格 VLA 控制（脚本中已开启）。

## 2. 前置准备

### 2.1 依赖与环境

1. 激活 conda 环境（示例）：

```bash
conda activate b2dvla
```

2. 确认 CARLA PythonAPI 可用：

```bash
python -c "import carla; print('carla ok')"
```

3. 仓库根目录需存在：

- singapore-onenorth_local.xodr（或你的 town 对应的 _local.xodr）
- custom_route.xml
- my_checkpoint.json（可自动生成）

### 2.2 外部 CARLA

请在宿主机先启动 CARLA 0.9.15（端口 2000），再执行本仓库流程。

## 3. 生成或更新路线文件

你可以使用两种方式生成 custom_route.xml。

### 3.1 基于当前地图自动找可连通起终点（推荐）

```bash
python scripts/gen_valid_custom_route.py \
  --host 127.0.0.1 \
  --port 2000 \
  --town singapore-onenorth \
  --out custom_route.xml \
  --min-distance 80
```

### 3.2 从 DriveStudio 轨迹直接转换

先把 DriveStudio 的 scene.json 与 trajectory.json 放到脚本默认读取目录，再运行：

```bash
python scripts/convert_drivestudio_route.py
```

说明：该脚本会输出 custom_route.xml 到仓库根目录。

## 4. 启动流程（推荐 3 个终端）

必须都在仓库根目录执行。

### 终端 A：启动 VLA 接口服务

```bash
conda run -n b2dvla python B2DVL_Adapter/web_interact_app.py --config ./configs/vlm_config_alpamayo.json
```

### 终端 B：启动背景交通（可选）

```bash
conda run -n b2dvla python scripts/spawn_drivestudio_agents.py
```

说明：脚本默认读取固定路径下的 agents.json，请按你的数据位置修改脚本中的文件路径。

### 终端 C：启动闭环评测

```bash
conda run -n b2dvla bash scripts/start_inference_custom.sh
```

## 5. 关键配置说明

start_inference_custom.sh 已包含以下关键设置：

- 使用外部 CARLA：EXTERNAL_CARLA=1
- OpenDRIVE 网格参数（减小墙体/异常碰撞）
- 严格 VLA 控制：
  - ENABLE_VLA_DIRECT_ACTION=1
  - STRICT_VLA_CONTROL=1
  - VLA_DIRECT_ACTION_MAX_DEVIATION_DEG=20
- 低速安全上限：CUSTOM_MAX_SPEED_MPS=1.5

如果你需要调参，直接编辑 scripts/start_inference_custom.sh。

## 6. 成功判定

运行中建议检查：

1. VLA 服务日志中有 /interact 200 返回。
2. 评测日志进入 Running the route / RouteScenario。
3. 不出现连续 Cannot spawn actor / Traceback。
4. 输出目录 eval_v1 下产生结果文件。

## 7. 常见问题与处理

### 7.1 直接退出或返回码 255

优先检查：

1. 外部 CARLA 是否已在 2000 端口启动。
2. custom_route.xml 是否可解析、起点是否落在可驾驶区域。
3. <town>_local.xodr 文件名是否和 route 的 town 对应。
4. VLA 服务是否已先启动。

### 7.2 车辆不动、冲出道路、掉落

优先检查：

1. custom_route.xml 首 waypoint 的 x/y/z/yaw 是否合理（避免半悬空）。
2. 使用 scripts/gen_valid_custom_route.py 重新生成可连通路线。
3. 地图本身是否存在几何碰撞异常（OpenDRIVE 网格问题）。

### 7.3 背景车脚本可运行但无车辆

1. 检查 agents.json 路径是否存在。
2. 检查蓝图名称在当前 CARLA 是否可用。
3. 检查 traffic manager 端口是否与评测端一致（默认 8000）。

## 8. 一次性最小可复现命令清单

在仓库根目录，按顺序执行：

```bash
# 1) 生成可用 route
conda run -n b2dvla python scripts/gen_valid_custom_route.py --host 127.0.0.1 --port 2000 --town singapore-onenorth --out custom_route.xml --min-distance 80

# 2) 启动 VLA 服务（新终端）
conda run -n b2dvla python B2DVL_Adapter/web_interact_app.py --config ./configs/vlm_config_alpamayo.json

# 3) 启动评测（新终端）
conda run -n b2dvla bash scripts/start_inference_custom.sh
```

如果需要背景车，再额外开一个终端执行：

```bash
conda run -n b2dvla python scripts/spawn_drivestudio_agents.py
```

## 9. 相关文件

- scripts/start_inference_custom.sh
- scripts/gen_valid_custom_route.py
- scripts/convert_drivestudio_route.py
- scripts/spawn_drivestudio_agents.py
- configs/vlm_config_alpamayo.json
- custom_route.xml
- singapore-onenorth_local.xodr
