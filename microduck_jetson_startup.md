# Microduck Jetson 启动与使用手册

> 更新日期：2026-09-04  
> Jetson 地址：`192.168.88.77`  
> 用户名：`seeed`  
> 项目目录：`/home/seeed/microduck-jetson/microduck_rl`

## 0. JetPack 7.2 训练环境配置

本章用于在 JetPack 7.2 Jetson 上首次安装或重新构建 Microduck 训练环境。后续日常训练可以直接从第 1 章开始。

### 0.1 已验证的平台

当前环境已在以下设备上验证：

| 项目 | 版本 |
|------|------|
| Jetson | NVIDIA Jetson Orin NX 16GB |
| 系统 | Ubuntu 24.04 LTS，aarch64 |
| JetPack | 7.2 |
| L4T | R39.2 |
| 系统 CUDA | 13.2 |
| Python | 3.12 |
| PyTorch | 2.9.1+cu130 |
| MuJoCo | 3.10.0 |
| Warp | 1.12.0 |

不要为了安装训练环境单独升级或降级 JetPack、CUDA 驱动和 L4T。系统 CUDA 应继续由 JetPack 管理，项目依赖安装在用户目录的 `.venv` 中。

### 0.2 硬件准备

建议满足以下条件：

- 使用 NVMe，至少保留 25GB 可用空间。
- 使用主动散热和可靠电源。
- Jetson 连接网络；首次安装需要下载数 GB Python/CUDA 包。
- 训练期间建议使用 40W 性能模式。

查看当前设备信息：

```bash
cat /proc/device-tree/model | tr '\0' '\n'
cat /etc/nv_tegra_release
dpkg-query -W -f='${Version}\n' nvidia-jetpack
/usr/local/cuda/bin/nvcc --version
python3 --version
free -h
df -h /
```

查看并设置性能模式：

```bash
sudo nvpmodel -q
```

当前 Orin NX 已使用 40W 模式。如果设备当前不是合适的高性能模式，应先用 `sudo nvpmodel` 查看该机型支持的模式编号，不要直接照抄其他 Jetson 型号的模式编号。

### 0.3 当前设备一键部署

当前 Jetson 已保存部署脚本：

```text
/home/seeed/microduck-jetson/deploy_microduck_jetson.sh
```

重新构建环境时执行：

```bash
ssh seeed@192.168.88.77

SUDO_PASSWORD=<JETSON_PASSWORD> \
TARGET_DIR=$HOME/microduck-jetson/microduck_rl \
bash $HOME/microduck-jetson/deploy_microduck_jetson.sh
```

脚本会完成：

1. 安装系统编译、MuJoCo、GLFW、GStreamer、FFmpeg 和 tmux 依赖。
2. 安装 `uv`。
3. 获取 Microduck RL 源码。
4. 创建 Python 3.12 虚拟环境。
5. 安装项目锁定依赖。
6. 覆盖安装 JetPack 7.2 可用的 CUDA 13 PyTorch。
7. 执行真实 CUDA 矩阵运算。
8. 运行 64 环境、5 轮训练冒烟测试。

### 0.4 从本机复制部署脚本

如果 Jetson 上没有部署脚本，在开发电脑的仓库目录执行：

```bash
cd /home/darklee/microduck-jetson

ssh seeed@192.168.88.77 'mkdir -p ~/microduck-jetson'

scp deploy_microduck_jetson.sh \
  seeed@192.168.88.77:~/microduck-jetson/
```

然后按照上一节运行脚本。

### 0.5 手动安装系统依赖

如果不使用一键脚本，先登录 Jetson：

```bash
ssh seeed@192.168.88.77
```

安装依赖：

```bash
sudo env DEBIAN_FRONTEND=noninteractive apt-get update

sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git curl wget build-essential tmux ffmpeg \
  libgl1 libgles2 libegl1 \
  libglfw3 libglfw3-dev libudev-dev \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  libgstreamer-plugins-bad1.0-dev
```

这些包用于源码构建、MuJoCo 本地窗口、EGL 离屏渲染、视频编码和后台训练。

### 0.6 配置 CUDA 环境变量

JetPack 7.2 的 CUDA 安装在 `/usr/local/cuda`。追加环境变量：

```bash
grep -Fqx 'export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH' ~/.bashrc || \
  echo 'export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH' >> ~/.bashrc

grep -Fqx 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' ~/.bashrc || \
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

grep -Fqx 'export UV_HTTP_TIMEOUT=600' ~/.bashrc || \
  echo 'export UV_HTTP_TIMEOUT=600' >> ~/.bashrc

grep -Fqx 'export MUJOCO_GL=egl' ~/.bashrc || \
  echo 'export MUJOCO_GL=egl' >> ~/.bashrc

source ~/.bashrc
```

默认使用 `MUJOCO_GL=egl`，适合 SSH 和训练。Jetson 本地打开 MuJoCo 窗口时临时改为 `MUJOCO_GL=glfw`。

### 0.7 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

如果安装后找不到 `uv`：

```bash
export PATH=$HOME/.local/bin:$PATH
```

### 0.8 配置可选网络代理

当前 Jetson 的本地代理端口为 `127.0.0.1:7897`。只有该端口正在监听时才设置代理：

```bash
if timeout 1 bash -c '</dev/tcp/127.0.0.1/7897' 2>/dev/null; then
  export HTTP_PROXY=http://127.0.0.1:7897
  export HTTPS_PROXY=http://127.0.0.1:7897
  export ALL_PROXY=socks5h://127.0.0.1:7897
fi
```

检查 PyTorch 下载源：

```bash
curl -IL --connect-timeout 10 \
  https://download.pytorch.org/whl/cu130/torch/
```

如果直连超时但代理可以访问，应在安装依赖的同一个终端中保留上述代理变量。

### 0.9 获取 Microduck RL 源码

创建工作目录：

```bash
mkdir -p ~/microduck-jetson
cd ~/microduck-jetson
```

国内网络可使用 GitHub 镜像下载源码快照：

```bash
curl -L --fail --retry 3 \
  -o /tmp/microduck_rl.tar.gz \
  https://ghfast.top/https://github.com/pollen-robotics/microduck_rl/archive/refs/heads/main.tar.gz

rm -rf ~/microduck-jetson/microduck_rl
mkdir -p ~/microduck-jetson/microduck_rl

tar -xzf /tmp/microduck_rl.tar.gz \
  --strip-components=1 \
  -C ~/microduck-jetson/microduck_rl
```

进入项目：

```bash
cd ~/microduck-jetson/microduck_rl
test -f pyproject.toml && echo 'Source OK'
```

### 0.10 安装项目锁定依赖

项目的 BAM 依赖来自 GitHub。网络无法直连 GitHub 时，临时配置 Git 镜像：

```bash
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0='url.https://ghfast.top/https://github.com/.insteadOf'
export GIT_CONFIG_VALUE_0='https://github.com/'
```

同步项目依赖：

```bash
cd ~/microduck-jetson/microduck_rl
export UV_HTTP_TIMEOUT=600
uv sync
```

这一步会创建 `.venv`，并安装 MuJoCo、Warp、MJLab、BAM、RSL-RL、ONNX Runtime 等依赖。

注意：锁文件安装的默认 PyTorch 可能是 CPU 版，因此下一步必须覆盖安装 CUDA 版。

### 0.11 安装 CUDA 13 PyTorch

先只替换 PyTorch、TorchVision 和 TorchAudio，避免升级 NumPy 等项目锁定包：

```bash
uv pip install --reinstall --no-deps \
  'torch==2.9.1' \
  'torchvision==0.24.1' \
  'torchaudio==2.9.1' \
  --index-url https://download.pytorch.org/whl/cu130
```

再安装 CUDA 13 运行库。继续使用 `--no-deps`，避免改变项目中的 NumPy、MediaPy 和其他版本：

```bash
uv pip install --no-deps \
  'nvidia-cuda-nvrtc==13.0.48' \
  'nvidia-cuda-runtime==13.0.48' \
  'nvidia-cuda-cupti==13.0.48' \
  'nvidia-cudnn-cu13==9.13.0.50' \
  'nvidia-cublas==13.0.0.19' \
  'nvidia-cufft==12.0.0.15' \
  'nvidia-curand==10.4.0.35' \
  'nvidia-cusolver==12.0.3.29' \
  'nvidia-cusparse==12.6.2.49' \
  'nvidia-cusparselt-cu13==0.8.0' \
  'nvidia-nccl-cu13==2.27.7' \
  'nvidia-nvshmem-cu13==3.3.24' \
  'nvidia-nvtx==13.0.39' \
  'nvidia-nvjitlink==13.0.39' \
  'nvidia-cufile==1.15.0.42' \
  'triton==3.5.1' \
  --index-url https://download.pytorch.org/whl/cu130
```

如果下载中断后出现 `Invalid Wheel-Version`，说明该包缓存不完整。只清理报错包并重试，例如：

```bash
uv cache clean nvidia-cuda-nvrtc
uv cache clean nvidia-nvjitlink
```

不要直接清空整个 `~/.cache/uv`，否则需要重新下载全部 CUDA 包。

### 0.12 验证 CUDA 和项目导入

不要只检查 `torch.cuda.is_available()`，必须执行真实 CUDA 算子：

```bash
cd ~/microduck-jetson/microduck_rl

uv run --no-sync python3 - <<'PY'
import mediapy
import mujoco
import numpy
import torch
import warp
import mjlab_microduck

print('PyTorch:', torch.__version__)
print('PyTorch CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('Capability:', torch.cuda.get_device_capability(0))
print('NumPy:', numpy.__version__)
print('MuJoCo:', mujoco.__version__)
print('Warp:', warp.__version__)

left = torch.randn(512, 512, device='cuda')
right = torch.randn(512, 512, device='cuda')
result = torch.matmul(left, right)
torch.cuda.synchronize()
print('CUDA matmul OK:', result.device, tuple(result.shape))
print('Project import: OK')
PY
```

JetPack 7.2 当前验证结果应包含：

```text
PyTorch: 2.9.1+cu130
PyTorch CUDA: 13.0
CUDA available: True
GPU: Orin
Capability: (8, 7)
CUDA matmul OK: cuda:0 (512, 512)
Project import: OK
```

### 0.13 运行训练冒烟测试

```bash
cd ~/microduck-jetson/microduck_rl
export MUJOCO_GL=egl

uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.logger tensorboard \
  --agent.max_iterations 5
```

正常情况下会显示：

- `device=cuda:0`
- Warp 识别到 `Orin` 和 `sm_87`
- 环境数为 64
- 训练从 iteration 0 运行到 iteration 4
- 最终正常退出并生成 `model_4.pt`

使用 `--agent.logger tensorboard` 是因为未登录 WandB 时，默认 WandB logger 会提示缺少 API Key。

### 0.14 JetPack 7.2 关键注意事项

1. 后续所有 `uv run` 命令都加 `--no-sync`。
2. CUDA PyTorch 安装完成后不要再次直接执行 `uv sync`，否则 CUDA Torch 和 CUDA 13 运行库可能被锁文件版本替换或删除。
3. 如确实需要重新同步依赖，应重新执行第 0.11 节的 CUDA PyTorch 覆盖安装。
4. 不要使用普通的 `pip install torch`，它可能安装 CPU 版。
5. 不要让 CUDA PyTorch 安装自动升级 NumPy；应使用文档中的 `--no-deps` 命令。
6. 无 WandB API Key 时统一使用 `--agent.logger tensorboard`。
7. SSH/训练使用 `MUJOCO_GL=egl`，Jetson 本地窗口使用 `MUJOCO_GL=glfw`。
8. `uv pip check` 可能把 `nvidia-cusparselt-cu13` 的 `manylinux2014_sbsa` 标签报告为平台不兼容；当前库文件已确认是 ARM aarch64，并已通过 CUDA训练验证。
9. 首次启动 MuJoCo/Warp 会编译 CUDA 内核，可能需要数分钟；后续会使用缓存，启动明显加快。
10. 训练时使用 `sudo tegrastats` 监控内存、温度和功耗。

## 1. 登录 Jetson

```bash
ssh seeed@192.168.88.77
```

登录时使用 Jetson 实际配置的用户密码。

## 2. 进入训练环境

```bash
cd ~/microduck-jetson/microduck_rl
export MUJOCO_GL=egl
```

训练和测试命令必须使用 `uv run --no-sync`。不要省略 `--no-sync`，否则 `uv` 可能根据锁文件把 CUDA 版 PyTorch 替换为 CPU 版。

## 3. 检查 GPU 环境

```bash
uv run --no-sync python3 - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

left = torch.randn(512, 512, device="cuda")
right = torch.randn(512, 512, device="cuda")
result = torch.matmul(left, right)
torch.cuda.synchronize()
print("CUDA matmul OK:", result.device)
PY
```

正常环境应显示：

- PyTorch：`2.9.1+cu130`
- CUDA Runtime：`13.0`
- GPU：`Orin`
- `CUDA available: True`
- `CUDA matmul OK: cuda:0`

## 4. 冒烟测试

正式训练前可运行小规模测试：

```bash
uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.logger tensorboard \
  --agent.max_iterations 5
```

测试完成后会在 `logs/rsl_rl/velocity/` 下生成日志和 checkpoint。

## 5. 正式训练

```bash
uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.logger tensorboard
```

如果显存不足，可依次降低并行环境数量：

```text
4096 -> 2048 -> 1024 -> 512
```

## 6. 使用 tmux 后台训练

创建训练会话：

```bash
tmux new -s microduck-train
```

在 tmux 中运行：

```bash
cd ~/microduck-jetson/microduck_rl
export MUJOCO_GL=egl

uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.logger tensorboard 2>&1 | tee ~/microduck-jetson/train.log
```

按 `Ctrl+B`，松开后再按 `D`，可退出 tmux 而不停止训练。

重新进入训练会话：

```bash
tmux attach -t microduck-train
```

查看现有会话：

```bash
tmux ls
```

## 7. 查看训练状态

查看训练进程：

```bash
ps -ef | grep '[t]rain Mjlab'
```

查看后台日志：

```bash
tail -f ~/microduck-jetson/train.log
```

查看 GPU、温度和功耗：

```bash
sudo tegrastats
```

查看当前功耗模式：

```bash
sudo nvpmodel -q
```

当前设备使用 `40W` 模式。

## 8. 查看训练输出

```bash
cd ~/microduck-jetson/microduck_rl
find logs/rsl_rl/velocity -name 'model_*.pt' -type f | sort
```

最近一次部署验证生成的 checkpoint：

```text
logs/rsl_rl/velocity/2026-09-03_18-50-37_velocity/model_4.pt
```

查看最新运行目录：

```bash
ls -dt logs/rsl_rl/velocity/* | head
```

## 9. 查看 TensorBoard

在 Jetson 上启动 TensorBoard：

```bash
cd ~/microduck-jetson/microduck_rl
uv run --no-sync tensorboard \
  --logdir logs/rsl_rl/velocity \
  --host 0.0.0.0 \
  --port 6006
```

然后在同一局域网电脑的浏览器中访问：

```text
http://192.168.88.77:6006
```

## 10. 断点续训

先查找需要恢复的 checkpoint：

```bash
find logs/rsl_rl/velocity -name 'model_*.pt' -type f | sort
```

恢复训练示例：

```bash
uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 4096 \
  --agent.logger tensorboard \
  --agent.run-name resume \
  --agent.load-checkpoint /完整路径/model_XXXX.pt \
  --agent.resume True
```

## 11. 停止训练

前台训练可按：

```text
Ctrl+C
```

如需停止后台进程，先查询 PID：

```bash
ps -ef | grep '[t]rain Mjlab'
```

再正常终止进程：

```bash
kill <PID>
```

不要直接使用 `kill -9`，除非进程无法正常退出。

## 12. 可视化

服务器或 SSH 环境默认使用 EGL：

```bash
export MUJOCO_GL=egl
```

通过 SSH 使用浏览器 Viewer：

```bash
uv run --no-sync play Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file /完整路径/model_XXXX.pt \
  --num-envs 1 \
  --viewer viser
```

浏览器访问 `http://192.168.88.77:8080`。

如果需要在 Jetson 本地打开 MuJoCo 窗口，Jetson 必须连接显示器并运行桌面环境。在 Jetson 桌面终端执行：

```bash
cd ~/microduck-jetson/microduck_rl
export MUJOCO_GL=glfw

uv run --no-sync play Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/velocity/2026-09-03_19-01-44_velocity/model_4.pt \
  --num-envs 1 \
  --viewer native
```

如果终端中的 `DISPLAY` 为空，可先执行 `export DISPLAY=:0`。关闭窗口或按 `Ctrl+C` 停止可视化。

## 13. ONNX 仿真推理与键盘遥控

### 13.1 PT 与 ONNX 的用途

训练产生的两类模型文件用途不同：

| 文件 | 主要用途 | 能否继续训练 |
|------|----------|--------------|
| `model_XXXX.pt` | 断点续训、使用 `play` 查看、重新导出模型 | 可以 |
| `*.onnx` | 键盘仿真推理、浏览器 Sandbox、实体机器人部署 | 不可以 |

键盘遥控脚本 `scripts/infer_policy.py` 使用 ONNX Runtime，只需要 `.onnx`，不需要同时加载 `.pt`。

### 13.2 查找训练模型

查看已有 checkpoint：

```bash
cd ~/microduck-jetson/microduck_rl
find logs/rsl_rl/velocity -type f -name 'model_*.pt' | sort
```

查看已有 ONNX：

```bash
find logs/rsl_rl/velocity -type f -name '*.onnx' | sort
```

当前已训练模型示例：

```text
logs/rsl_rl/velocity/2026-09-03_19-02-46_velocity/model_3000.pt
```

对应 ONNX：

```text
logs/rsl_rl/velocity/2026-09-03_19-02-46_velocity/2026-09-03_19-02-46_velocity.onnx
```

### 13.3 从 checkpoint 导出 ONNX

如果训练目录内还没有 ONNX，可执行：

```bash
cd ~/microduck-jetson/microduck_rl

uv run --no-sync python3 scripts/export.py \
  Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/velocity/训练目录/model_XXXX.pt \
  --onnx-file logs/rsl_rl/velocity/训练目录/walking.onnx
```

必须使用项目的 `scripts/export.py`，它会把观测归一化器一起写入 ONNX。

### 13.4 Jetson 本地启动键盘推理

Jetson 必须连接显示器并运行桌面环境。在 Jetson 本地桌面终端执行：

```bash
cd ~/microduck-jetson/microduck_rl

export DISPLAY=:0
export MUJOCO_GL=glfw

uv run --no-sync python3 scripts/infer_policy.py \
  --walking logs/rsl_rl/velocity/2026-09-03_19-02-46_velocity/2026-09-03_19-02-46_velocity.onnx \
  --new-cmd-obs
```

`--new-cmd-obs` 用于当前 61 维观测模型，不应省略。

MuJoCo 窗口用于观看仿真，但控制按键必须在**启动命令的终端窗口**中输入。终端需要保持焦点。

### 13.5 速度控制

默认启动后处于速度控制模式：

| 按键 | 功能 |
|------|------|
| `↑` | 增加前进速度 |
| `↓` | 降低前进速度；继续降低可后退 |
| `←` / `→` | 左右横移 |
| `A` / `E` | 左转 / 右转 |
| `Space` | 速度、横移和转向全部归零 |
| `T` | 暂停或恢复策略推理 |
| `P` | 随机推鸭子，测试抗扰和恢复能力 |
| `Q` | 退出仿真 |

按键是增量控制。例如多按几次 `↑` 会逐步增加前进速度，按 `Space` 可立即清零指令。

### 13.6 头部控制

按 `H` 进入或退出头部控制模式：

| 按键 | 功能 |
|------|------|
| `Z` / `S` | 颈部俯仰 |
| `↑` / `↓` | 头部俯仰 |
| `←` / `→` | 头部左右转动 |
| `A` / `E` | 头部左右侧倾 |
| `Space` | 头部指令恢复为零 |

### 13.7 身体姿态控制

按 `B` 进入或退出身体姿态控制模式：

| 按键 | 功能 |
|------|------|
| `↑` / `↓` | 身体升高 / 降低，每次约 10 mm |
| `←` / `→` | 身体前倾 / 后倾，每次约 10° |
| `A` / `E` | 身体左右侧倾，每次约 10° |
| `Z` / `S` | 身体左右偏航，每次约 10° |
| `Space` | 身体姿态指令恢复为零 |

### 13.8 多策略行为按键

只有在启动时提供了对应的 ONNX 参数，下面的行为键才有效：

| 按键 | 行为 | 所需参数 |
|------|------|----------|
| `G` | 拾取 | `--ground-pick ground_pick.onnx` |
| `Y` | 坐下/站起 | `--sitstand sitstand.onnx` |
| `K` | 左脚踢球 | `--kick-left kick_left.onnx` |
| `L` | 右脚踢球 | `--kick-right kick_right.onnx` |
| `R` | 前滚翻 | `--roulade roulade.onnx` |

多策略启动示例：

```bash
uv run --no-sync python3 scripts/infer_policy.py \
  --walking walking.onnx \
  --standing standing.onnx \
  --sitstand sitstand.onnx \
  --ground-pick ground_pick.onnx \
  --roulade roulade.onnx \
  --kick-left kick_left.onnx \
  --kick-right kick_right.onnx \
  --new-cmd-obs
```

### 13.9 调试和数据记录

打印观测与动作：

```bash
uv run --no-sync python3 scripts/infer_policy.py \
  --walking walking.onnx \
  --new-cmd-obs \
  --debug
```

保存观测和动作到 CSV：

```bash
uv run --no-sync python3 scripts/infer_policy.py \
  --walking walking.onnx \
  --new-cmd-obs \
  --save-csv inference.csv
```

### 13.10 常见问题

如果没有弹出 MuJoCo 窗口：

```bash
echo $DISPLAY
```

在 Jetson 本地桌面中一般为 `:0`，为空时执行：

```bash
export DISPLAY=:0
export MUJOCO_GL=glfw
```

如果按键没有反应：

1. 点击启动命令所在的终端窗口，使终端获得焦点。
2. 不要在 MuJoCo Viewer 窗口中输入控制键。
3. 不要使用后台方式启动，标准输入必须连接终端 TTY。
4. 按 `T` 检查策略推理是否被暂停。
5. 按 `Space` 清零指令后重新输入。

如果提示 ONNX 输入维度不匹配，应确认当前模型是 61 维观测模型，并保留 `--new-cmd-obs`。

## 14. 环境损坏时重新部署

部署脚本位于：

```text
~/microduck-jetson/deploy_microduck_jetson.sh
```

重新部署：

```bash
SUDO_PASSWORD=<JETSON_PASSWORD> \
TARGET_DIR=$HOME/microduck-jetson/microduck_rl \
bash ~/microduck-jetson/deploy_microduck_jetson.sh
```

脚本会自动执行系统依赖安装、项目同步、CUDA PyTorch 安装、GPU 验证和训练冒烟测试。

## 15. 重要注意事项

1. 所有训练、播放和导出命令都使用 `uv run --no-sync`。
2. 未配置 WandB API Key 时使用 `--agent.logger tensorboard`。
3. 无显示器或通过 SSH 使用时设置 `MUJOCO_GL=egl`。
4. 正式训练建议在 tmux 中运行，避免 SSH 断线导致训练停止。
5. 训练期间建议使用 `sudo tegrastats` 监控温度、内存和功耗。
6. 当前系统磁盘空间有限，定期清理不需要的 checkpoint 和缓存。

## 16. 官方多动作模型与训练自己的 PT

### 16.1 已下载的官方模型

Pollen Robotics 官方 `microduck-policies` 仓库发布了 9 个推理模型，已下载到：

```text
~/microduck-jetson/microduck_rl/pretrained/pollen-robotics/
```

文件列表：

| ONNX | 功能 | 本地推理参数 |
|------|------|--------------|
| `alpha_walking.onnx` | 双足行走 | `--walking` |
| `alpha_stand.onnx` | 站立策略 | `--standing` |
| `alpha_sitstand.onnx` | 坐下/站起 | `--sitstand` |
| `alpha_ground_pick.onnx` | 低头触地拾取 | `--ground-pick` |
| `ball_kick_left.onnx` | 左脚踢球 | `--kick-left` |
| `ball_kick_right.onnx` | 右脚踢球 | `--kick-right` |
| `roller.onnx` | 轮滑速度控制 | 轮滑模式的 `--walking` |
| `roller_crouch.onnx` | 轮滑下蹲滑行 | 轮滑模式的 `--ground-pick` |
| `roulade.onnx` | 前滚翻 | `--roulade` |

检查文件：

```bash
cd ~/microduck-jetson/microduck_rl
cd pretrained/pollen-robotics
sha256sum -c SHA256SUMS
```

### 16.2 为什么没有对应的官方 PT

截至 2026-09-04，官方公开模型仓库及其完整 Git 提交历史只包含上述 ONNX、`manifest.json` 和校验文件，没有发布对应的 `.pt` 训练 checkpoint。

这些官方 ONNX 的元数据中 `run_path` 为 `None`，因此也不能通过 ONNX 追溯到公开 W&B run 下载原始 checkpoint。

因此：

- 官方 ONNX 可以直接用于仿真推理和机器人部署。
- 官方 ONNX 不能用于断点续训。
- 不能把 ONNX 无损还原为包含 Actor、Critic、优化器和训练状态的原始 `.pt`。
- 若需要可续训 `.pt`，必须使用相应任务自行训练。

当前自己训练的 walking checkpoint 位于：

```text
logs/rsl_rl/velocity/2026-09-03_19-02-46_velocity/model_3000.pt
```

### 16.3 一次加载全部官方双足动作

在 Jetson 本地桌面终端执行：

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

- 方向键：前后和横移。
- `A` / `E`：转向。
- `G`：拾取。
- `Y`：坐下/站起。
- `R`：前滚翻。
- `K` / `L`：左脚/右脚踢球。
- `Space`：速度指令清零。
- `Q`：退出。

### 16.4 加载官方轮滑动作

```bash
cd ~/microduck-jetson/microduck_rl
export DISPLAY=:0
export MUJOCO_GL=glfw

uv run --no-sync python3 scripts/infer_policy.py \
  --roller \
  --walking pretrained/pollen-robotics/roller.onnx \
  --ground-pick pretrained/pollen-robotics/roller_crouch.onnx \
  --new-cmd-obs
```

轮滑模式下方向键控制速度，`A` / `E` 控制转向，`G` 触发下蹲滑行。

### 16.5 通用训练参数

主要动作任务共用以下 PPO 参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| Actor 隐藏层 | `512, 256, 128` | 输出 14 个关节动作 |
| Critic 隐藏层 | `512, 256, 128` | 只在训练期间使用 |
| 激活函数 | `ELU` | Actor 和 Critic 相同 |
| 每环境每轮步数 | `24` | `num_steps_per_env` |
| PPO epochs | `5` | 每批数据更新次数 |
| mini-batches | `4` | 每轮 PPO 分批数量 |
| 学习率 | `1e-3` | adaptive schedule |
| gamma | `0.99` | 折扣因子 |
| lambda | `0.95` | GAE 参数 |
| entropy | `0.01` | 轮滑速度/Swizzle 为 `0.03` |
| checkpoint 间隔 | `250` iterations | 产生 `model_250.pt` 等 |

每轮采样数量为：

```text
并行环境数 × 24
```

例如 2048 个环境每轮采集 49,152 条 transition。增加环境数通常能提升 GPU 吞吐，但会增加显存和内存占用。

Jetson Orin NX 16GB 建议从 1024 或 2048 环境开始；稳定后再测试 4096。

### 16.6 各动作任务和默认训练长度

下表中的迭代数是当前源码配置的 `max_iterations`，不是必须一次跑完。可以先训练 1000～3000 轮查看效果，再决定是否继续。

| 动作 | Task ID | 日志目录 | 默认最大迭代 |
|------|---------|----------|--------------|
| 平地行走 | `Mjlab-Velocity-Flat-MicroDuck` | `logs/rsl_rl/velocity/` | 50,000 |
| 崎岖地形行走 | `Mjlab-Velocity-Rough-MicroDuck` | `logs/rsl_rl/velocity/` | 50,000 |
| 行走+跌倒恢复 | `Mjlab-VelStand-Flat-MicroDuck` | `logs/rsl_rl/velstand/` | 20,000 |
| 从地面站起 | `Mjlab-StandUp-Flat-MicroDuck` | `logs/rsl_rl/microduck_stand/` | 15,000 |
| 坐下/站起 | `Mjlab-SitStand-Flat-MicroDuck` | `logs/rsl_rl/microduck_sitstand/` | 15,000 |
| 低头触地拾取 | `Mjlab-GroundPick-Flat-MicroDuck` | `logs/rsl_rl/ground_pick/` | 20,000 |
| 左/右脚踢球 | `Mjlab-BallKick-Flat-MicroDuck` | `logs/rsl_rl/ball_kick_<foot>/` | 10,000 |
| 前滚翻 | `Mjlab-Roulade-Flat-MicroDuck` | `logs/rsl_rl/microduck_roulade/` | 10,000 |
| 轮滑速度控制 | `Mjlab-Velocity-Flat-MicroDuck-Rollers` | `logs/rsl_rl/velocity_rollers/` | 50,000 |
| 轮滑 Swizzle | `Mjlab-Velocity-Swizzle-MicroDuck` | `logs/rsl_rl/velocity_swizzle/` | 50,000 |
| 轮滑下蹲 | `Mjlab-RollerCrouch-Flat-MicroDuck` | `logs/rsl_rl/roller_crouch/` | 8,000 |
| 轮滑斜坡 | `Mjlab-RollerSlope-Flat-MicroDuck` | `logs/rsl_rl/roller_slope/` | 8,000 |
| 轮滑起身 | `Mjlab-RollerStandUp-Flat-MicroDuck` | `logs/rsl_rl/roller_standup/` | 15,000 |
| 轮滑原地旋转 | `Mjlab-Spin-Flat-MicroDuck` | `logs/rsl_rl/spin/` | 8,000 |

`alpha_stand.onnx` 没有公开训练来源。自行训练替代模型时：

- 如果需要站立、行走和跌倒恢复一体化，优先训练 `Mjlab-VelStand-Flat-MicroDuck`。
- 如果重点是从趴下、仰躺或坐姿站起，训练 `Mjlab-StandUp-Flat-MicroDuck`。

### 16.7 通用训练命令

```bash
cd ~/microduck-jetson/microduck_rl
export MUJOCO_GL=egl

uv run --no-sync train <TASK_ID> \
  --env.scene.num-envs 2048 \
  --agent.logger tensorboard
```

限制本次训练迭代数：

```bash
uv run --no-sync train <TASK_ID> \
  --env.scene.num-envs 2048 \
  --agent.logger tensorboard \
  --agent.max_iterations 2000
```

正式长时间训练建议放入 tmux：

```bash
tmux new -s microduck-action
```

### 16.8 各动作训练命令

行走：

```bash
uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 2048 --agent.logger tensorboard
```

行走和跌倒恢复：

```bash
uv run --no-sync train Mjlab-VelStand-Flat-MicroDuck \
  --env.scene.num-envs 2048 --agent.logger tensorboard
```

从地面站起：

```bash
uv run --no-sync train Mjlab-StandUp-Flat-MicroDuck \
  --env.scene.num-envs 2048 --agent.logger tensorboard
```

坐下/站起：

```bash
uv run --no-sync train Mjlab-SitStand-Flat-MicroDuck \
  --env.scene.num-envs 2048 --agent.logger tensorboard
```

低头触地拾取：

```bash
uv run --no-sync train Mjlab-GroundPick-Flat-MicroDuck \
  --env.scene.num-envs 2048 --agent.logger tensorboard
```

前滚翻：

```bash
uv run --no-sync train Mjlab-Roulade-Flat-MicroDuck \
  --env.scene.num-envs 2048 --agent.logger tensorboard
```

轮滑速度控制：

```bash
uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck-Rollers \
  --env.scene.num-envs 2048 --agent.logger tensorboard
```

轮滑下蹲：

```bash
uv run --no-sync train Mjlab-RollerCrouch-Flat-MicroDuck \
  --env.scene.num-envs 2048 --agent.logger tensorboard
```

### 16.9 分别训练左右脚踢球

当前源码默认训练右脚：

```text
src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py
KICK_FOOT = "right"
```

训练右脚：

```bash
uv run --no-sync train Mjlab-BallKick-Flat-MicroDuck \
  --env.scene.num-envs 2048 --agent.logger tensorboard
```

训练左脚前，把 `KICK_FOOT` 改为 `left`：

```bash
sed -i 's/KICK_FOOT = "right"/KICK_FOOT = "left"/' \
  src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py

uv run --no-sync train Mjlab-BallKick-Flat-MicroDuck \
  --env.scene.num-envs 2048 --agent.logger tensorboard
```

左脚训练和导出结束后，可改回右脚：

```bash
sed -i 's/KICK_FOOT = "left"/KICK_FOOT = "right"/' \
  src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py
```

左右脚模型必须分别保存，不能用一个 checkpoint 同时代表两只脚。

### 16.10 从自己的 PT 导出 ONNX

通用格式：

```bash
uv run --no-sync python3 scripts/export.py <TASK_ID> \
  --checkpoint-file /完整路径/model_XXXX.pt \
  --onnx-file /输出路径/action.onnx
```

例如导出 SitStand：

```bash
uv run --no-sync python3 scripts/export.py \
  Mjlab-SitStand-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/microduck_sitstand/训练目录/model_XXXX.pt \
  --onnx-file sitstand.onnx
```

例如导出 GroundPick：

```bash
uv run --no-sync python3 scripts/export.py \
  Mjlab-GroundPick-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/ground_pick/训练目录/model_XXXX.pt \
  --onnx-file ground_pick.onnx
```

### 16.11 训练顺序建议

建议按以下顺序逐个训练和验证，不要同时启动多个大规模训练：

1. `Velocity`：先建立可靠行走基线。
2. `VelStand` 或 `StandUp`：补充跌倒恢复能力。
3. `SitStand`：训练可控坐站。
4. `GroundPick`：训练相位动作。
5. `Roulade` 和左右脚 `BallKick`：训练一次性技巧动作。
6. `Velocity-Rollers` 和 `RollerCrouch`：最后训练轮滑模型。

每个动作先以 64 个环境、5 轮做冒烟测试，再用 1024 或 2048 环境长时间训练。训练完成后先用 `play` 查看 `.pt`，再导出 ONNX 并通过 `infer_policy.py` 做键盘热切换测试。

## 17. 自定义动作训练进阶篇

如何新增自己的动作任务、奖励函数、相位命令、课程学习、Task ID、测试和键盘推理接入，请阅读：

```text
~/microduck-jetson/microduck_custom_action_training.md
```

该文档以自定义“鞠躬 Bow”动作为完整示例，并包含 61→14 策略契约和奖励设计检查清单。
