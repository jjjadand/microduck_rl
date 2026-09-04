# Microduck Jetson 训练指南（含可视化篇）

> **目标**：在 NVIDIA Jetson（Orin Nano / Orin NX / AGX Orin / Thor）上完成 Microduck 强化学习策略的端到端训练、可视化验证与 ONNX 导出。
>
> **适用平台**：Jetson Orin Nano (8GB/4GB)、Jetson Orin NX、Jetson AGX Orin、Jetson Thor 等所有 aarch64 + CUDA 设备。
>
> **最后更新**：2026-09-01

---

## 目录

1. [硬件与系统要求](#1-硬件与系统要求)
2. [Jetson 系统准备](#2-jetson-系统准备)
3. [训练环境搭建](#3-训练环境搭建)
4. [首次训练（行走策略）](#4-首次训练行走策略)
5. [训练可视化与验证](#5-训练可视化与验证)
6. [训练其他行为](#6-训练其他行为)
7. [策略导出与验证](#7-策略导出与验证)
8. [Sim 验证（键盘测试）](#8-sim-验证键盘测试)
9. [部署到实体机器人](#9-部署到实体机器人)
10. [常见问题排查](#10-常见问题排查)
11. [参考资源](#11-参考资源)

---

## 1. 硬件与系统要求

### 最低配置

| 项目 | 要求 |
|------|------|
| **设备** | Jetson Orin Nano 8GB / Orin NX / AGX Orin / Thor |
| **JetPack** | 6.2+（推荐 JetPack 7.x + CUDA 13.x） |
| **存储** | 64GB+ microSD 或 NVMe SSD（强烈推荐 NVMe） |
| **散热** | 主动散热（风扇+散热片），训练时 GPU 满载 |
| **网络** | 有线以太网或稳定 Wi-Fi（下载约 2GB CUDA 依赖） |
| **电源** | 官方电源适配器（Orin Nano Super 需 25W 模式） |
| **显示器**（可选） | HDMI/DP 显示器 + 键盘鼠标（用于交互式可视化） |

### 关键提醒

- **Jetson Orin Nano 4GB 版本**：内存极其紧张，建议关闭桌面环境（`sudo systemctl set-default multi-user.target`），纯 SSH 操作，并减少并行环境数（如 `--env.scene.num-envs 1024` 或更低）。
- **Orin Nano Super (25W)**：在 `nvpmodel` 中切换到 25W 模式可获得最佳训练性能。
- **Thor / DGX Spark / GB10**：算力更强，但 CUDA Compute Capability 可能较新（如 sm_110），需要特别注意 PyTorch 版本兼容性。
- **可视化需要显示器**：交互式 3D Viewer 需要 Jetson 连接显示器并运行桌面环境；无头（headless）模式只能使用离屏视频录制或浏览器 Sandbox。

---

## 2. Jetson 系统准备

### 2.1 确认 CUDA 环境

```bash
# 检查 JetPack 和 CUDA 版本
sudo apt update
sudo apt install -y nvidia-jetpack

# 验证 CUDA
nvcc --version
# 期望输出：Cuda compilation tools, release 13.x

# 验证 GPU 可用性
python3 -c "import torch; print(torch.cuda.is_available())"
# 期望输出：True
```

如果 `nvcc` 未找到，添加环境变量：

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 2.2 设置高性能模式（重要）

```bash
# Orin Nano Super 切换到 25W 模式
sudo nvpmodel -m 1

# 查看当前模式
sudo nvpmodel -q

# 监控温度与功耗（训练时建议另开终端观察）
sudo tegrastats
```

### 2.3 安装基础工具（含可视化依赖）

```bash
sudo apt update && sudo apt install -y     git curl wget build-essential     libgl1 libgles2 libegl1     libglfw3 libglfw3-dev     libudev-dev libgstreamer1.0-dev     libgstreamer-plugins-base1.0-dev     libgstreamer-plugins-bad1.0-dev     ffmpeg  # 视频编码
```

> **libglfw3-dev** 是交互式 MuJoCo Viewer 的必需依赖，无头环境可省略。

---

## 3. 训练环境搭建

### 3.1 安装 uv（Python 包管理器）

Microduck 官方使用 `uv` 管理依赖，比 pip 更快更可靠。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 验证
uv --version
```

### 3.2 克隆官方训练仓库

```bash
cd ~
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl

# 查看当前 commit（用于复现）
git log --oneline -1
```

### 3.3 ⚠️ ARM 平台关键：同步依赖

**这是 Jetson 上最容易踩坑的一步。**

官方 `uv.lock` 锁定的 PyTorch 在 x86_64 上没问题，但在 aarch64 上 `uv sync` 会拉取 **CPU 版 torch**（约 104MB），`torch.cuda.is_available()` 甚至也会返回 `True`，但一跑真实 CUDA 算子就崩溃。

#### 正确做法：

```bash
# 第一步：延长 uv 超时（ARM 下载 CUDA 轮子约 2GB，默认 30 秒会中断）
export UV_HTTP_TIMEOUT=600

# 第二步：先按官方 lock 同步基础依赖
uv sync

# 第三步：手动覆盖安装 NVIDIA 官方 aarch64 CUDA 版 PyTorch
# 注意：根据你的 JetPack/CUDA 版本选择正确的索引
# JetPack 6.x (CUDA 12.x) -> cu128
# JetPack 7.x (CUDA 13.x) -> cu130

uv pip install torch torchvision torchaudio     --index-url https://download.pytorch.org/whl/cu130

# 如果你需要特定版本（如 Jetson Thor 的 sm_110），可能需要：
# uv pip install torch==2.11.0+cu130 torchvision torchaudio #     --index-url https://download.pytorch.org/whl/cu130
```

#### 验证 PyTorch CUDA 是否真正可用

**不要只信 `torch.cuda.is_available()`！** 必须跑真实 matmul：

```bash
uv run python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))

# 真实 CUDA 算子测试
a = torch.randn(512, 512, device='cuda')
b = torch.randn(512, 512, device='cuda')
c = torch.matmul(a, b)
print('Matmul OK, result device:', c.device)
print('PyTorch version:', torch.__version__)
"
```

如果 matmul 通过，说明 CUDA 内核真正可用。

### 3.4 设置渲染后端环境变量

根据使用场景选择渲染后端：

```bash
# 交互式可视化（需要显示器 + 桌面环境）
export MUJOCO_GL=glfw

# 离屏渲染（无头/SSH 环境，用于视频录制）
export MUJOCO_GL=egl

# 纯软件渲染（无 GPU 显示支持时的兜底）
export MUJOCO_GL=osmesa
```

建议将常用设置写入 `~/.bashrc`：

```bash
# 默认使用 EGL（离屏），需要交互式时再临时切换 GLFW
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
source ~/.bashrc
```

---

## 4. 首次训练（行走策略）

### 4.1 冒烟测试（必做）

在长跑之前，先用小规模环境验证整个链路：

```bash
# 64 环境 × 5 迭代，约 1-2 分钟
uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck     --env.scene.num-envs 64     --agent.max_iterations 5
```

> **注意**：所有 `uv run` 命令在 Jetson 上必须加 `--no-sync`，否则 uv 会把手动装好的 CUDA 版 torch 悄悄还原回 `uv.lock` 里的 CPU 版。

如果看到 `iteration 5/5` 正常结束且没有 CUDA 错误，说明环境就绪。

### 4.2 正式训练行走策略

```bash
# 4096 并行环境，约 1-2 小时出可用步态
uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck     --env.scene.num-envs 4096
```

训练过程中会输出：
- 当前 iteration / 总 iteration
- 平均 episode length
- 平均 reward
- 策略熵、价值损失等

**注意：训练过程中没有实时 3D 可视化窗口**，所有环境并行在 GPU 上仿真，不渲染。可视化请在训练完成后通过 `play` 命令进行。

### 4.3 断点续训

```bash
# 从最新 checkpoint 恢复
uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck     --env.scene.num-envs 4096     --agent.run-name resume     --agent.load-checkpoint model_29999.pt     --agent.resume True
```

checkpoint 默认保存在 `logs/rsl_rl/velocity/` 下。

### 4.4 后台持久训练（tmux）

Jetson 训练时间较长，建议使用 tmux 防止 SSH 断开：

```bash
sudo apt install -y tmux

tmux new -s microduck-train

# 在 tmux 会话中执行训练
export UV_HTTP_TIMEOUT=600
uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck     --env.scene.num-envs 4096 2>&1 | tee train.log

# 按 Ctrl+B 再按 D  detach 会话
# 稍后重连：tmux attach -t microduck-train
```

---

## 5. 训练可视化与验证

### 5.1 可视化方式总览

| 方式 | 命令 | 需要显示器 | 交互性 | 用途 |
|------|------|-----------|--------|------|
| **交互式 Viewer** | `MUJOCO_GL=glfw play ...` | ✅ 需要 | ✅ 可旋转/缩放 | 本地调试策略，观察细节 |
| **离屏视频录制** | `MUJOCO_GL=egl play --video ...` | ❌ 不需要 | ❌ 生成文件 | 无头环境记录训练结果 |
| **多鸭同框渲染** | `MUJOCO_GL=egl render-pack.py` | ❌ 不需要 | ❌ 生成文件 | 生成 RL 群像展示视频 |
| **浏览器 Sandbox** | 访问 HF Spaces | ❌ 不需要 | ✅ 手柄控制 | 在线测试 ONNX 策略 |
| **键盘 Sim 测试** | `infer_policy.py` | ⚠️ 可选 | ✅ WASD 控制 | 快速验证策略行为 |

### 5.2 方式一：交互式 3D Viewer（推荐，需接显示器）

这是最直观的可视化方式，可以实时旋转、缩放视角，观察机器人步态细节。

#### 前置条件

1. Jetson 连接了 HDMI/DP 显示器
2. 已登录桌面环境（或运行 X11/VNC）
3. 已安装 GLFW：`sudo apt install -y libglfw3 libglfw3-dev`

#### 查看训练好的策略

```bash
# 使用 GLFW 后端启动交互式窗口
MUJOCO_GL=glfw uv run --no-sync play Mjlab-Velocity-Flat-MicroDuck     --wandb-run-path <your-entity/project/run_id>

# 或从本地 checkpoint 查看
MUJOCO_GL=glfw uv run --no-sync play Mjlab-Velocity-Flat-MicroDuck     --checkpoint logs/rsl_rl/velocity/walk-main/model_50000.pt
```

**Viewer 操作说明：**

| 操作 | 功能 |
|------|------|
| **鼠标左键拖拽** | 旋转视角 |
| **鼠标右键拖拽** | 平移视角 |
| **滚轮** | 缩放 |
| **双击模型** | 跟踪该部位 |
| **空格键** | 暂停/继续仿真 |
| **Backspace** | 重置仿真 |
| **方向键 / WASD** | 控制机器人移动（如果策略支持） |

#### 多环境并排查看

```bash
# 同时显示 9 个环境（3×3 网格）
MUJOCO_GL=glfw uv run --no-sync play Mjlab-Velocity-Flat-MicroDuck     --checkpoint model_50000.pt     --num-envs 9
```

### 5.3 方式二：离屏视频录制（无头环境首选）

Jetson 通常作为无头服务器使用（纯 SSH），此时使用 EGL 离屏渲染生成视频。

#### 单环境视频录制

```bash
# 确保 EGL 后端
export MUJOCO_GL=egl

# 录制 10 秒视频（约 300 帧 @ 30fps）
uv run --no-sync play Mjlab-Velocity-Flat-MicroDuck     --checkpoint logs/rsl_rl/velocity/walk-main/model_50000.pt     --video     --video-length 10     --num-envs 1

# 视频默认保存在当前目录，文件名包含时间戳
```

#### 带速度指令的视频

```bash
# 录制机器人以特定速度行走的视频
uv run --no-sync play Mjlab-Velocity-Flat-MicroDuck     --checkpoint model_50000.pt     --video     --video-length 10     --num-envs 1     --cmd-res-lin-vel-x 1.0  # 向前 1.0 m/s
    --cmd-res-lin-vel-y 0.0
```

### 5.4 方式三：多鸭同框群像渲染（社区脚本）

`render-pack.py` 是社区贡献的渲染工具，可以生成经典的 RL 训练群像视频——一个主视角 + 周围多个同伴机器人。

```bash
export MUJOCO_GL=egl

uv run --no-sync python scripts/render-pack.py     Mjlab-Velocity-Flat-MicroDuck     --ckpt logs/rsl_rl/velocity/walk-main/model_50000.pt     --frames 400     --num-envs 64     --extra 12     --origin root     --distance 4.2     --elevation -25

# 输出原始 RGB 帧（默认 320×240）
# 可用 ffmpeg 转码为 MP4：
ffmpeg -i output_frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p duck_pack.mp4
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--frames 400` | 总帧数 |
| `--num-envs 64` | 总环境数 |
| `--extra 12` | 同框显示的邻居数量 |
| `--origin root` | 相机跟踪目标（root = 机器人质心） |
| `--distance 4.2` | 相机距离（米） |
| `--elevation -25` | 俯仰角（度） |

### 5.5 方式四：浏览器 Sandbox（零安装）

无需任何本地环境，直接在线可视化：

1. 访问 [Microduck Sandbox](https://huggingface.co/spaces/pollen-robotics/microduck-sandbox)
2. 上传你导出的 ONNX 策略文件
3. 使用键盘或游戏手柄实时控制
4. 支持轮滑模式切换

**适用场景：**
- 快速验证策略行为
- 没有 Jetson 或本地环境未配置好时
- 向他人展示训练成果

### 5.6 方式五：键盘控制的 Sim 验证

在本地 MuJoCo 中用键盘实时测试策略，支持多策略热切换：

```bash
# 单策略测试（需要显示器时用 GLFW，无头时用 EGL）
MUJOCO_GL=glfw uv run --no-sync scripts/infer_policy.py --walking output.onnx

# 多策略热切换测试（模拟实体机器人运行时行为）
MUJOCO_GL=glfw uv run --no-sync scripts/infer_policy.py     --walking walk.onnx     --standing stand.onnx     --sitstand sitstand.onnx     --roulade roulade.onnx     --new-cmd-obs
```

**键盘控制：**

| 按键 | 功能 |
|------|------|
| **WASD / 方向键** | 控制行走方向和速度 |
| **G** | 触发 GroundPick（拾取） |
| **Y** | 触发 Sit/Stand（坐/站切换） |
| **R** | 触发 Roulade（前滚翻） |
| **K / L** | 触发踢球 |
| **Esc** | 退出 |

> `--debug` 可打印观测值；`--save-csv` 可保存观测-动作对用于 sim2real 对比。

### 5.7 可视化工作流建议

**典型迭代流程：**

```
1. 训练（无头，tmux 后台）
   └── uv run --no-sync train ... --env.scene.num-envs 4096

2. 训练完成后，离屏录制视频检查
   └── MUJOCO_GL=egl uv run --no-sync play --video ...

3. 如果视频表现好，接显示器用交互式 Viewer 细调
   └── MUJOCO_GL=glfw uv run --no-sync play ...

4. 导出 ONNX → 浏览器 Sandbox 最终验证
   └── 访问 https://huggingface.co/spaces/pollen-robotics/microduck-sandbox

5. 部署到实体机器人
   └── scp output.onnx microduck@<ip>:~/policies/
```

---

## 6. 训练其他行为

Microduck 官方提供 13+ 个预定义任务，全部共享 61 维观测空间，支持运行时热切换。

| 任务 ID | 描述 | 源码默认最大迭代 |
|---------|------|------------------|
| `Mjlab-Velocity-Flat-MicroDuck` | 平地行走（主任务） | 50,000 |
| `Mjlab-Velocity-Rough-MicroDuck` | 崎岖地形行走 | 50,000 |
| `Mjlab-VelStand-Flat-MicroDuck` | 行走 + 跌倒自起 | 20,000 |
| `Mjlab-StandUp-Flat-MicroDuck` | 从趴/躺/坐姿态站起 | 15,000 |
| `Mjlab-SitStand-Flat-MicroDuck` | 指令式坐↔站切换 | 15,000 |
| `Mjlab-GroundPick-Flat-MicroDuck` | 低头触地拾取 | 20,000 |
| `Mjlab-BallKick-Flat-MicroDuck` | 左脚或右脚踢 70mm 小球 | 10,000 |
| `Mjlab-Roulade-Flat-MicroDuck` | 前滚翻 | 10,000 |
| `Mjlab-Velocity-Flat-MicroDuck-Rollers` | 轮滑速度控制 | 50,000 |
| `Mjlab-Velocity-Swizzle-MicroDuck` | 轮滑 Swizzle | 50,000 |
| `Mjlab-RollerCrouch-Flat-MicroDuck` | 轮滑下蹲滑行 | 8,000 |
| `Mjlab-RollerSlope-Flat-MicroDuck` | 轮滑斜坡 | 8,000 |
| `Mjlab-RollerStandUp-Flat-MicroDuck` | 轮滑起身 | 15,000 |
| `Mjlab-Spin-Flat-MicroDuck` | 轮滑原地旋转 | 8,000 |

训练命令格式相同：

```bash
uv run --no-sync train <TASK_ID> \
  --env.scene.num-envs 2048 \
  --agent.logger tensorboard
```

上表是源码中的默认上限，可通过 `--agent.max_iterations` 缩短。Jetson Orin NX 16GB 建议从 1024 或 2048 个环境开始。完整的多动作训练、左右脚踢球和 ONNX 导出说明见 `microduck_jetson_startup.md` 第 16 章。

---

## 7. 策略导出与验证

### 7.1 导出 ONNX

```bash
# 从 W&B 训练日志导出
uv run --no-sync python3 scripts/export.py \
  Mjlab-Velocity-Flat-MicroDuck \
  --wandb-run-path <your-entity/project/run_id>

# 或从本地 checkpoint 导出
uv run --no-sync python3 scripts/export.py \
  Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/velocity/训练目录/model_XXXX.pt \
  --onnx-file output.onnx
```

> **重要**：必须使用 `scripts/export.py` 导出 ONNX，它会将观测归一化器 bake 进 ONNX 图。手动转换的 checkpoint 在实体机器人上会看到未归一化的观测，导致行为异常。

导出后得到 `output.onnx`。

### 7.2 ONNX 契约校验

验证导出的模型是否符合 Microduck 的 61→14 契约：

```bash
uv run --no-sync python3 -c "
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('output.onnx')
input_name = sess.get_inputs()[0].name

# 61 维观测，float32
dummy_obs = np.zeros((1, 61), dtype=np.float32)
actions = sess.run(None, {input_name: dummy_obs})[0]

print('Input shape:', dummy_obs.shape)
print('Output shape:', actions.shape)
print('Output dtype:', actions.dtype)
print('Output range:', actions.min(), 'to', actions.max())
assert actions.shape == (1, 14), 'Output must be 14 actions!'
assert np.isfinite(actions).all(), 'Output contains NaN/Inf!'
print('ONNX validation PASSED')
"
```

---

## 8. Sim 验证（键盘测试）

在实体机器人上部署前，先在 CPU MuJoCo 中用键盘测试策略：

```bash
# 单策略测试
MUJOCO_GL=glfw uv run --no-sync scripts/infer_policy.py --walking output.onnx

# 多策略热切换测试（模拟实体机器人运行时行为）
MUJOCO_GL=glfw uv run --no-sync scripts/infer_policy.py     --walking walk.onnx     --standing stand.onnx     --sitstand sitstand.onnx     --roulade roulade.onnx     --new-cmd-obs
```

键盘控制：
- **WASD** / **方向键**：控制行走方向和速度
- **G**：触发 GroundPick（拾取）
- **Y**：触发 Sit/Stand（坐/站切换）
- **R**：触发 Roulade（前滚翻）
- **K / L**：触发踢球

> `--debug` 可打印观测值；`--save-csv` 可保存观测-动作对用于 sim2real 对比。

---

## 9. 部署到实体机器人

### 9.1 传输 ONNX 到 Microduck

```bash
# Microduck 的 onboard 计算机是 Rockchip RK3566 (aarch64)
# 通过 SSH 或 USB 传输策略文件
scp output.onnx microduck@<robot-ip>:~/policies/
```

### 9.2 实体机器人加载策略

Microduck 的 onboard 软件（`pollen-robotics/microduck`）支持运行时热切换策略：

```bash
# 在机器人上
robotctl policy load ~/policies/output.onnx

# 查看当前策略
robotctl policy status

# 使用游戏手柄触发不同行为（策略自动热切换）
```

### 9.3 无实体机器人的替代方案

如果你还没有拿到 Microduck 实体机（首批预计 2026 圣诞节前发货），可以：

1. **浏览器 Sandbox**：访问 [Microduck Sandbox](https://huggingface.co/spaces/pollen-robotics/microduck-sandbox)，直接运行 ONNX 策略，支持手柄和轮滑模式。
2. **持续在 Sim 中迭代**：完善策略后导出 ONNX，等实体机到手直接部署。

---

## 10. 常见问题排查

### Q1: `uv sync` 下载中断 / 超时

**现象**：`uv sync` 下载到一半报错 `HTTP timeout`。

**原因**：ARM 平台的 CUDA 轮子约 2GB，uv 默认 30 秒超时不够。

**解决**：
```bash
export UV_HTTP_TIMEOUT=600
uv sync
```

### Q2: `torch.cuda.is_available()` 为 True，但训练时 CUDA 报错

**现象**：`is_available()` 返回 True，但一跑训练就报 CUDA kernel 错误。

**原因**：装上了 PyPI 的 CPU 版 torch（aarch64 默认），或 CUDA 版本与 GPU Compute Capability 不匹配。

**解决**：
1. 确认从 NVIDIA 索引安装了 CUDA 版 torch
2. 用 512×512 matmul 验证真实 CUDA 算子
3. 对于 Jetson Thor (sm_110)，确保使用 `torch>=2.11.0+cu130`

### Q3: `uv run` 后 torch 被还原为 CPU 版

**现象**：手动装好 CUDA 版 torch 后，执行 `uv run train ...` 又变回 CPU 版。

**原因**：`uv run` 默认会重新同步 `uv.lock`。

**解决**：所有 `uv run` 命令都加 `--no-sync`：
```bash
uv run --no-sync train ...
uv run --no-sync play ...
```

### Q4: 录视频时 `mujoco.FatalError: mjr_makeContext` 崩溃

**现象**：`play --video` 或渲染脚本在无头机器上崩溃。

**原因**：没有可用的 GL context，或 `MUJOCO_GL` 环境变量未正确设置。

**解决**：
```bash
# 离屏渲染
export MUJOCO_GL=egl

# 如果 EGL 也不可用（某些 JetPack 版本）
export MUJOCO_GL=osmesa
```

### Q5: 交互式 Viewer 黑屏 / 闪退

**现象**：`MUJOCO_GL=glfw play ...` 窗口打开后黑屏或立即关闭。

**原因**：
1. 未安装 GLFW：`sudo apt install -y libglfw3 libglfw3-dev`
2. 未运行桌面环境（X11/Wayland 未启动）
3. 显示器未正确连接
4. Jetson 的 OpenGL 驱动问题

**解决**：
```bash
# 1. 确认 GLFW 已安装
sudo apt install -y libglfw3 libglfw3-dev

# 2. 确认在桌面环境中运行（不要纯 SSH）
echo $DISPLAY
# 应输出 :0 或类似值

# 3. 如果通过 SSH 连接，需要 X11 转发
ssh -X user@jetson-ip
# 然后在 SSH 会话中运行 viewer

# 4. 检查 OpenGL
python3 -c "import mujoco; print(mujoco.MjRenderContext)"
```

### Q6: Jetson Orin Nano 4GB 内存不足

**现象**：训练开始后不久被 OOM killer 终止。

**解决**：
```bash
# 1. 关闭桌面，纯命令行
sudo systemctl set-default multi-user.target
sudo reboot

# 2. 减少并行环境数
uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck     --env.scene.num-envs 512  # 或 1024

# 3. 关闭不必要的系统服务
sudo systemctl stop gdm3  # 或 sddm
```

### Q7: 训练速度比预期慢很多

**现象**：4096 环境训练超过 3 小时才出步态。

**排查**：
1. 确认处于 25W 高性能模式：`sudo nvpmodel -q`
2. 确认 CUDA 真正启用（matmul 测试）
3. 检查散热：温度超过 85°C 会降频，`tegrastats` 观察
4. Orin Nano 4GB 版本性能确实有限，考虑使用 Hugging Face Jobs 云端训练

### Q8: 没有 GPU，能在 Jetson 上纯 CPU 训练吗？

**回答**：**不能**。`microduck_rl` 基于 MuJoCo Warp，必须使用 CUDA GPU。Jetson 的 GPU 可以训练，只是较慢。如果没有 Jetson，使用 `--hf-jobs` 在 Hugging Face 云端训练。

```bash
# 云端训练（无需本地 GPU）
uv run train Mjlab-Velocity-Flat-MicroDuck     --env.scene.num-envs 4096     --hf-jobs
```

### Q9: 视频录制后找不到输出文件

**现象**：`play --video` 执行完但找不到 MP4 文件。

**原因**：
1. 视频可能以原始帧（PNG 序列）输出，需要手动转码
2. 输出路径未指定

**解决**：
```bash
# 查看当前目录下的视频文件
ls -la *.mp4 *.avi 2>/dev/null

# 如果只有 PNG 帧序列，用 ffmpeg 转码
ffmpeg -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4

# 指定输出路径
uv run --no-sync play ... --video --video-output ~/videos/duck_walk.mp4
```

---

## 11. 参考资源

### 官方仓库

| 仓库 | 用途 | 链接 |
|------|------|------|
| `pollen-robotics/microduck_rl` | RL 训练环境（本指南核心） | https://github.com/pollen-robotics/microduck_rl |
| `pollen-robotics/microduck` | 机器人 onboard 软件 | https://github.com/pollen-robotics/microduck |
| `pollen-robotics/microduck-gst-plugins` | GStreamer 插件 | https://github.com/pollen-robotics/microduck-gst-plugins |

### 社区资源

| 资源 | 说明 | 链接 |
|------|------|------|
| `microduck-rl-on-thor` | aarch64 CUDA 训练踩坑记录（中英双语） | https://github.com/metahubaifeel/microduck-rl-on-thor |
| `joeynyc/awesome-microduck` | 社区终极资源列表 | https://github.com/joeynyc/awesome-microduck |
| `microduck-lab` | DGX Spark 可复现训练工作区 | GitHub |
| `microduck-rl-genesis` | Genesis 引擎移植（AMD/ROCm 支持） | GitHub |
| `isaaclab-microduck` | Isaac Lab 3.0 移植 | GitHub |

### 官方文档

- **[AGENTS.md](https://github.com/pollen-robotics/microduck_rl/blob/main/AGENTS.md)** — 奖励设计的 distilled playbook
- **[Cheat Sheet](https://github.com/pollen-robotics/microduck/blob/main/docs/cheat-sheet.md)** — `robotctl` 日常命令
- **[Architecture](https://github.com/pollen-robotics/microduck/blob/main/docs/architecture.md)** — 守护进程与控制循环架构
- **[Microduck Sandbox](https://huggingface.co/spaces/pollen-robotics/microduck-sandbox)** — 浏览器内模拟器

### 关键论文/技术

- **BAM M6 执行器模型**：针对 Dynamixel XL330 的电压控制律 + 反电动势 + 摩擦建模
- **域随机化**：电池电压、电压跌落、指令延迟、摩擦系数、地形
- **齿轮间隙仿真**：±1° 以被动铰链建模
- **统一观测空间**：61 维（48 维本体感知 + 13 维指令/目标）

---

## 附录 A：一键设置脚本

以下脚本整合了 Jetson 上的所有初始化步骤，保存为 `setup_jetson.sh` 后执行：

```bash
#!/bin/bash
set -e

echo "=== Microduck Jetson Setup ==="

# 1. 环境变量
export UV_HTTP_TIMEOUT=600
export MUJOCO_GL=egl

# 2. 确保 CUDA 在 PATH
if ! command -v nvcc &> /dev/null; then
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
fi

# 3. 安装 uv
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# 4. 克隆仓库
if [ ! -d "~/microduck_rl" ]; then
    cd ~
    git clone https://github.com/pollen-robotics/microduck_rl
fi

cd ~/microduck_rl

# 5. 同步依赖
echo "Syncing dependencies (this may take 10-30 min on first run)..."
uv sync

# 6. 安装 NVIDIA CUDA 版 PyTorch
echo "Installing CUDA PyTorch for aarch64..."
uv pip install torch torchvision torchaudio     --index-url https://download.pytorch.org/whl/cu130

# 7. 验证 CUDA
echo "Verifying CUDA..."
uv run --no-sync python3 -c "
import torch
a = torch.randn(512, 512, device='cuda')
b = torch.randn(512, 512, device='cuda')
c = torch.matmul(a, b)
print('CUDA matmul OK!')
print('PyTorch:', torch.__version__)
print('Device:', torch.cuda.get_device_name(0))
"

# 8. 冒烟测试
echo "Running smoke test..."
uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck     --env.scene.num-envs 64     --agent.max_iterations 5

echo "=== Setup complete! Ready to train. ==="
echo "Run: uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096"
```

## 附录 B：可视化快速参考卡

```bash
# ========== 交互式 Viewer（需显示器）==========
MUJOCO_GL=glfw uv run --no-sync play Mjlab-Velocity-Flat-MicroDuck     --checkpoint model_50000.pt

# 多环境并排
MUJOCO_GL=glfw uv run --no-sync play ... --num-envs 9

# ========== 离屏视频录制（无头）==========
MUJOCO_GL=egl uv run --no-sync play Mjlab-Velocity-Flat-MicroDuck     --checkpoint model_50000.pt     --video --video-length 10

# ========== 多鸭同框群像 ==========
MUJOCO_GL=egl uv run --no-sync python scripts/render-pack.py     Mjlab-Velocity-Flat-MicroDuck     --ckpt model_50000.pt     --frames 400 --num-envs 64 --extra 12

# ========== 键盘 Sim 测试 ==========
MUJOCO_GL=glfw uv run --no-sync scripts/infer_policy.py --walking output.onnx

# ========== 浏览器 Sandbox（零安装）==========
# 访问 https://huggingface.co/spaces/pollen-robotics/microduck-sandbox
```

---

*本指南基于 pollen-robotics/microduck_rl 官方仓库及社区 microduck-rl-on-thor 的实测经验整理。如有更新，欢迎提交 PR 或 Issue。*
