<div align="center">

# Microduck on Jetson

**Reinforcement-learning training, MuJoCo visualization, and keyboard-controlled inference**

<p>
  <a href="README.zh-CN.md"><strong>中文文档 / Chinese README →</strong></a>
</p>

</div>

This repository contains a reproducible Microduck reinforcement-learning demo for
Seeed Jetson devices. It supports official motion tasks, MuJoCo visualization,
keyboard-controlled ONNX inference, and custom motion development with MJLab.

The workflow was validated on JetPack 7.2, Ubuntu 24.04, CUDA 13.2, Python 3.12,
MuJoCo 3.10, and Warp 1.12. Jetson Orin Nano and Orin NX are supported; reduce
the number of parallel environments on devices with less memory.

> **中文说明：** The complete Chinese version, including the detailed deployment
> walkthrough, is available in [`README.zh-CN.md`](README.zh-CN.md).

## What This Demo Provides

- GPU-parallel training for walking, stand-up, sit/stand, ground-pick, roulade,
  ball-kick, roller, and custom motions.
- MuJoCo Native Viewer or browser-based Viser visualization.
- Keyboard-controlled inference using the official multi-policy ONNX demo.
- Checkpoint playback and ONNX export with the observation normalizer embedded.
- A complete one-leg balance example that can be used as a custom-task template.

## Repository Layout

```text
.
├── src/mjlab_microduck/tasks/       # Task environments, rewards, and registry
├── scripts/                         # Inference, export, and pose-editor scripts
├── pretrained/pollen-robotics/      # Official ONNX policies
├── models/checkpoints/              # Included PT training checkpoints
├── microduck_jetson_startup.md     # JetPack 7.2 setup and startup guide
├── microduck_jetson_training_guide.md
├── microduck_custom_action_training.md
├── README.md                        # English entry point
└── README.zh-CN.md                  # 中文完整说明
```

## 1. Environment Setup

Set the actual Jetson host and user before connecting. Do not put a private IP,
password, or machine-specific absolute path in scripts or documentation.

```bash
export JETSON_USER="<JETSON_USER>"
export JETSON_HOST="<JETSON_IP_OR_HOSTNAME>"
ssh "${JETSON_USER}@${JETSON_HOST}"
```

On the Jetson, deploy into a user-owned directory:

```bash
export TARGET_DIR="$HOME/microduck-jetson/microduck_rl"
export SUDO_PASSWORD="<JETSON_PASSWORD>"
bash "$HOME/microduck-jetson/deploy_microduck_jetson.sh"
cd "$TARGET_DIR"
```

For a complete JetPack 7.2 setup, dependency, display, TensorBoard, and
troubleshooting procedure, see [`microduck_jetson_startup.md`](microduck_jetson_startup.md).

## 2. Quick Start

List registered tasks:

```bash
cd ~/microduck-jetson/microduck_rl
uv run --no-sync list-envs | grep MicroDuck
```

Run a training smoke test before a long run:

```bash
export MUJOCO_GL=egl
uv run --no-sync train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.logger tensorboard \
  --agent.max_iterations 5
```

Typical official task IDs include:

| Motion | Task ID |
|---|---|
| Walking | `Mjlab-Velocity-Flat-MicroDuck` |
| Stand-up | `Mjlab-StandUp-Flat-MicroDuck` |
| Sit/stand | `Mjlab-SitStand-Flat-MicroDuck` |
| Ground pick | `Mjlab-GroundPick-Flat-MicroDuck` |
| Forward roulade | `Mjlab-Roulade-Flat-MicroDuck` |
| Ball kick | `Mjlab-BallKick-Flat-MicroDuck` |

### MuJoCo Playback

Use a trained PT checkpoint with the Native Viewer on a Jetson desktop:

```bash
export DISPLAY=:0
export MUJOCO_GL=glfw
uv run --no-sync play Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file /path/to/model_XXXX.pt \
  --num-envs 1 \
  --viewer native
```

For a browser viewer, use `--viewer viser` and open port `8080` on the Jetson
host. When the browser runs on the Jetson itself, `http://127.0.0.1:8080` can be
used.

### Official ONNX Keyboard Demo

Run this command from a Jetson desktop terminal:

```bash
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

Arrow keys control translation, `A`/`E` control turning, `G` triggers ground
pick, `Y` switches sit/stand, `R` triggers roulade, `K`/`L` trigger left/right
kick, `Space` zeros the velocity command, and `Q` exits.

Export a checkpoint with the project wrapper so the observation normalizer is
embedded in the ONNX file:

```bash
uv run --no-sync python3 scripts/export.py \
  Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file /path/to/model_XXXX.pt \
  --onnx-file walking_custom.onnx
```

## 3. Custom Motion Training

Custom motions are implemented as registered MJLab tasks rather than by editing
an ONNX file. The usual workflow is:

1. Select the closest existing task template.
2. Define the pose, phase timeline, commands, rewards, and termination rules.
3. Keep the shared 61-dimensional actor observation and 14-dimensional action
   contract.
4. Register the task in `src/mjlab_microduck/tasks/__init__.py`.
5. Run a one-environment viewer check and a 64-environment smoke test.
6. Increase training duration and parallelism only after the smoke test passes.
7. Play the resulting PT checkpoint and export ONNX for inference.

### One-Leg Balance Example

The repository includes a left-foot-support, right-foot-lift task:

```text
Task ID:        Mjlab-OneLegBalance-Flat-MicroDuck
Environment:    src/mjlab_microduck/tasks/microduck_one_leg_balance_env_cfg.py
Registration:   src/mjlab_microduck/tasks/__init__.py
Pose editor:    scripts/one_leg_pose_editor.py
```

The task ID is the key passed to the MJLab registry. It is not a file name and
is not an argument to the environment factory:

```bash
uv run --no-sync list-envs | grep OneLegBalance

export MUJOCO_GL=glfw
uv run --no-sync python scripts/one_leg_pose_editor.py

export MUJOCO_GL=egl
uv run --no-sync train Mjlab-OneLegBalance-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.logger tensorboard \
  --agent.max_iterations 5
```

The pose, phase timing, reward terms, and PPO configuration are defined in
`microduck_one_leg_balance_env_cfg.py`. The pose editor prints the final named
pose when the MuJoCo window closes. The five-iteration run only validates the
training chain; it is not a trained policy.

Read [`microduck_custom_action_training.md`](microduck_custom_action_training.md)
for the full custom-task example, reward design, curriculum, Backlash variants,
testing, and deployment guidance.

## Models

- Official ONNX policies are under `pretrained/pollen-robotics/`.
- Included PT files under `models/checkpoints/` are training checkpoints, not
  official Pollen Robotics releases.
- PT files contain training state; ONNX files are inference artifacts and do not
  contain the PPO optimizer or critic state.
- Always use `scripts/export.py` to export a checkpoint for deployment.

See [`models/README.md`](models/README.md) for the artifact policy and checksum
information.

## Documentation

- **Chinese complete guide:** [`README.zh-CN.md`](README.zh-CN.md)
- **Jetson setup and startup:** [`microduck_jetson_startup.md`](microduck_jetson_startup.md)
- **Training and visualization:** [`microduck_jetson_training_guide.md`](microduck_jetson_training_guide.md)
- **Custom actions:** [`microduck_custom_action_training.md`](microduck_custom_action_training.md)

## License

See [`LICENSE`](LICENSE).
