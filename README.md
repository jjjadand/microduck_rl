# Microduck on Jetson：强化学习训练与 MuJoCo 推理 Demo

本 Demo 在 **Seeed Jetson Orin NX 16GB** 上部署 Microduck 强化学习训练环境，支持使用官方任务训练动作、加载官方 ONNX 策略进行 MuJoCo 可视化和键盘控制，并提供新增自定义动作任务的完整开发流程。

当前环境已基于 **JetPack 7.2 / Ubuntu 24.04 / CUDA 13.2 / Python 3.12** 验证。

## Demo 能做什么

- 在 Jetson GPU 上并行训练 Microduck 行走、起身、坐站、拾取、翻滚、踢球和轮滑动作。
- 使用 MuJoCo Native Viewer 或浏览器 Viewer 查看仿真效果。
- 加载官方 ONNX 策略，通过键盘实时控制速度并切换动作。
- 从自己训练的 `.pt` checkpoint 导出带观测归一化的 ONNX。
- 以现有任务为模板，开发鞠躬、点头、下蹲等自定义动作。

## 1. 部署环境与准备工作

### 1.1 已验证环境

| 项目 | 配置 |
|------|------|
| Jetson | Seeed reComputer，NVIDIA Jetson Orin NX 16GB |
| 操作系统 | Ubuntu 24.04 LTS，aarch64 |
| JetPack / L4T | JetPack 7.2 / L4T R39.2 |
| CUDA | 系统 CUDA 13.2 |
| Python | 3.12 |
| PyTorch | 2.9.1+cu130 |
| MuJoCo | 3.10.0 |
| Warp | 1.12.0 |

建议使用 NVMe 并预留至少 25GB 空间，同时准备主动散热、稳定电源和可访问 Python/CUDA 软件源的网络。

### 1.2 工作目录

```text
/home/seeed/microduck-jetson/
├── deploy_microduck_jetson.sh          # JetPack 7.2 一键部署脚本
├── microduck_rl/                        # 训练、导出和仿真项目
│   ├── src/mjlab_microduck/tasks/       # 任务环境、奖励和任务注册
│   ├── scripts/                         # ONNX 导出与推理脚本
│   ├── pretrained/pollen-robotics/      # 官方 ONNX 动作模型
│   └── logs/rsl_rl/                     # 本地训练日志与 PT checkpoint
├── microduck_jetson_startup.md          # 环境、训练、推理完整手册
├── microduck_jetson_training_guide.md   # 训练与可视化专题
└── microduck_custom_action_training.md  # 自定义动作进阶篇
```

### 1.3 首次部署或重建环境

```bash
ssh seeed@192.168.88.77

SUDO_PASSWORD=<JETSON_PASSWORD> \
TARGET_DIR=$HOME/microduck-jetson/microduck_rl \
bash $HOME/microduck-jetson/deploy_microduck_jetson.sh
```

进入项目。后续训练命令必须在项目目录内执行：

```bash
cd ~/microduck-jetson/microduck_rl
export MUJOCO_GL=egl
```

验证 CUDA：

```bash
uv run --no-sync python3 - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

value = torch.randn(512, 512, device="cuda")
print("CUDA matmul:", (value @ value).device)
PY
```

> 使用 `uv run --no-sync` 可以避免锁文件重新同步后覆盖 Jetson 使用的 CUDA 版 PyTorch。

## 2. 快速开始：官方动作训练与推理

### 2.1 查看官方任务

```bash
cd ~/microduck-jetson/microduck_rl
uv run --no-sync list-envs | grep MicroDuck
```

常用任务包括：

| 动作 | Task ID |
|------|---------|
| 行走 | `Mjlab-Velocity-Flat-MicroDuck` |
| 行走与跌倒恢复 | `Mjlab-VelStand-Flat-MicroDuck` |
| 从地面起身 | `Mjlab-StandUp-Flat-MicroDuck` |
| 坐下/站起 | `Mjlab-SitStand-Flat-MicroDuck` |
| 低头触地拾取 | `Mjlab-GroundPick-Flat-MicroDuck` |
| 前滚翻 | `Mjlab-Roulade-Flat-MicroDuck` |
| 踢球 | `Mjlab-BallKick-Flat-MicroDuck` |

### 2.2 运行训练冒烟测试

先用 64 个并行环境和 5 次迭代确认训练链路：

```bash
cd ~/microduck-jetson/microduck_rl
export MUJOCO_GL=egl

uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.logger tensorboard \
  --agent.max_iterations 5
```

训练输出保存在 `logs/rsl_rl/`。正式训练可将环境数提高到 `2048` 或 `4096`；如果内存不足则逐级降低。

```bash
uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 2048 \
  --agent.logger tensorboard
```

### 2.3 查看自己训练的 PT

通过 SSH 使用浏览器 Viewer：

```bash
export MUJOCO_GL=egl

uv run --no-sync play Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file /完整路径/model_XXXX.pt \
  --num-envs 1 \
  --viewer viser
```

浏览器打开 `http://192.168.88.77:8080`。

Jetson 连接显示器时，可在本地桌面终端使用 Native Viewer：

```bash
export DISPLAY=:0
export MUJOCO_GL=glfw

uv run --no-sync play Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file /完整路径/model_XXXX.pt \
  --num-envs 1 \
  --viewer native
```

### 2.4 使用官方 ONNX 键盘推理

官方 ONNX 位于 `pretrained/pollen-robotics/`。它们可直接用于推理，但不能恢复 PPO 优化器或继续训练；需要续训时，应先运行对应任务生成自己的 `.pt`。

在 Jetson 本地桌面终端启动全部双足动作：

```bash
cd ~/microduck-jetson/microduck_rl
export DISPLAY=:0
export MUJOCO_GL=glfw

uv run --no-sync python3 scripts/infer_policy.py \
  --walking pretrained/pollen-robotics/alpha_walking.onnx \
  --standing pretrained/pollen-robotics/alpha_stand.onnx \
  --sitstand pretrained/pollen-robotics/alpha_sitstand.onnx \
  --ground-pick pretrained/pollen-robotics/alpha_ground_pick.onnx \
  --roulade pretrained/pollen-robotics/roulade.onnx \
  --kick-left pretrained/pollen-robotics/ball_kick_left.onnx \
  --kick-right pretrained/pollen-robotics/ball_kick_right.onnx \
  --new-cmd-obs
```

主要按键：

- 方向键：前后移动和横移。
- `A` / `E`：左右转向。
- `G`：触发拾取。
- `Y`：切换坐下/站起。
- `R`：触发前滚翻。
- `K` / `L`：左脚/右脚踢球。
- `Space`：速度指令归零。
- `Q`：退出。

### 2.5 导出自己的 ONNX

```bash
uv run --no-sync python3 scripts/export.py \
  Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file /完整路径/model_XXXX.pt \
  --onnx-file walking_custom.onnx
```

必须使用项目的 `scripts/export.py`，以便把训练使用的观测归一化器一并写入 ONNX。

## 3. 自定义动作训练

自定义动作不是直接编辑 ONNX，而是新增或复制一个训练任务，定义动作目标、奖励、终止条件和 PPO 配置，再注册新的 Task ID 进行训练。

推荐流程：

1. 根据动作类型选择最接近的官方模板。
2. 复制环境配置文件并设计动作时间线或阶段命令。
3. 定义目标关节姿态和动作结果奖励。
4. 在 `src/mjlab_microduck/tasks/__init__.py` 注册新任务。
5. 先用随机策略检查 Viewer，再运行 64 环境冒烟训练。
6. 分阶段增加训练长度、并行环境和域随机化。
7. 使用训练得到的 PT 可视化，最后导出 ONNX。

例如开发 Bow 鞠躬动作，可从相位动作模板开始：

```bash
cd ~/microduck-jetson/microduck_rl

cp src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py \
  src/mjlab_microduck/tasks/microduck_bow_env_cfg.py
```

注册 `Mjlab-Bow-Flat-MicroDuck` 后，依次运行：

```bash
# 确认任务已注册
uv run --no-sync list-envs | grep Mjlab-Bow

# 随机策略检查场景、重置和 Viewer
export MUJOCO_GL=glfw
uv run --no-sync play Mjlab-Bow-Flat-MicroDuck \
  --agent random --num-envs 1 --viewer native

# 训练冒烟测试
export MUJOCO_GL=egl
uv run --no-sync train Mjlab-Bow-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.logger tensorboard \
  --agent.max_iterations 5
```

开发时必须保持项目模型契约：Actor 观测为 61 维、策略动作为 14 维、未使用的 13 维命令字段必须补零，并避免硬编码关节索引。详细的 Bow 示例、奖励函数、课程学习、Backlash 任务和测试方法见：

```text
~/microduck-jetson/microduck_custom_action_training.md
```

## 4. 详细文档

- 环境部署、日常训练、TensorBoard、MuJoCo 和键盘推理：`microduck_jetson_startup.md`
- 训练参数、可视化方式和实体机器人部署：`microduck_jetson_training_guide.md`
- 自定义动作任务开发与 Bow 完整示例：`microduck_custom_action_training.md`

仓库内模型说明见 `models/README.md`。官方 ONNX 位于
`pretrained/pollen-robotics/`；Jetson 训练产生的 PT checkpoint 位于
`models/checkpoints/rsl_rl/velocity/`。官方项目没有提供可续训 PT，因此这里的
PT 是本次 Jetson 行走训练结果，不是官方发布模型。

Jetson 实机训练截图和 MuJoCo 推理录屏位于 `docs/media/`，包括并行训练、
GPU 监控、前进/后退推理以及键盘触发踢球演示。

## 5. 最短复现路径

```bash
ssh seeed@192.168.88.77
cd ~/microduck-jetson/microduck_rl
export MUJOCO_GL=egl

# 训练链路验证
uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.logger tensorboard \
  --agent.max_iterations 5

# Jetson 本地桌面运行官方 ONNX Demo
export DISPLAY=:0
export MUJOCO_GL=glfw
uv run --no-sync python3 scripts/infer_policy.py \
  --walking pretrained/pollen-robotics/alpha_walking.onnx \
  --standing pretrained/pollen-robotics/alpha_stand.onnx \
  --sitstand pretrained/pollen-robotics/alpha_sitstand.onnx \
  --ground-pick pretrained/pollen-robotics/alpha_ground_pick.onnx \
  --roulade pretrained/pollen-robotics/roulade.onnx \
  --kick-left pretrained/pollen-robotics/ball_kick_left.onnx \
  --kick-right pretrained/pollen-robotics/ball_kick_right.onnx \
  --new-cmd-obs
```

完成以上步骤，即可复现 Microduck 在 Jetson 上的训练、MuJoCo 可视化和官方多动作键盘推理 Demo。
