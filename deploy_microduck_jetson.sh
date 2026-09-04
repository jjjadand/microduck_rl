#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${TARGET_DIR:-$HOME/microduck-jetson/microduck_rl}"
GITHUB_MIRROR="${GITHUB_MIRROR:-https://ghfast.top/https://github.com/}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0="url.${GITHUB_MIRROR}.insteadOf"
export GIT_CONFIG_VALUE_0="https://github.com/"

if timeout 1 bash -c '</dev/tcp/127.0.0.1/7897' 2>/dev/null; then
    export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7897}"
    export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7897}"
    export ALL_PROXY="${ALL_PROXY:-socks5h://127.0.0.1:7897}"
fi

run_sudo() {
    if sudo -n true 2>/dev/null; then
        sudo "$@"
    else
        printf '%s\n' "${SUDO_PASSWORD:?SUDO_PASSWORD is required}" | sudo -S -p '' "$@"
    fi
}

ensure_bashrc_line() {
    local line="$1"
    grep -Fqx "$line" "$HOME/.bashrc" || printf '%s\n' "$line" >> "$HOME/.bashrc"
}

run_sudo env DEBIAN_FRONTEND=noninteractive apt-get update
run_sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git curl wget build-essential tmux ffmpeg \
    libgl1 libgles2 libegl1 libglfw3 libglfw3-dev libudev-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$(dirname "$TARGET_DIR")"
if [[ ! -f "$TARGET_DIR/pyproject.toml" ]]; then
    rm -rf "$TARGET_DIR"
    archive_path="$(mktemp --suffix=.tar.gz)"
    curl -L --fail --retry 3 \
        -o "$archive_path" \
        "${GITHUB_MIRROR}pollen-robotics/microduck_rl/archive/refs/heads/main.tar.gz"
    mkdir -p "$TARGET_DIR"
    tar -xzf "$archive_path" --strip-components=1 -C "$TARGET_DIR"
    rm -f "$archive_path"
fi

ensure_bashrc_line 'export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH'
ensure_bashrc_line 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH'
ensure_bashrc_line 'export UV_HTTP_TIMEOUT=600'
ensure_bashrc_line 'export MUJOCO_GL=egl'

cd "$TARGET_DIR"
uv sync
uv pip install --reinstall --no-deps \
    'torch==2.9.1' 'torchvision==0.24.1' 'torchaudio==2.9.1' \
    --index-url "$PYTORCH_INDEX"
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
    --index-url "$PYTORCH_INDEX"

uv run --no-sync python3 - <<'PY'
import torch

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
left = torch.randn(512, 512, device="cuda")
right = torch.randn(512, 512, device="cuda")
result = torch.matmul(left, right)
print("Matmul OK, result device:", result.device)
print("PyTorch version:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
PY

uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 64 \
    --agent.logger tensorboard \
    --agent.max_iterations 5

printf 'Microduck training environment is ready at %s\n' "$TARGET_DIR"
