# Microduck 自定义动作训练进阶篇

> 适用环境：Jetson Orin NX 16GB、JetPack 7.2、L4T R39.2、CUDA 13.2  
> 项目目录：`/home/seeed/microduck-jetson/microduck_rl`  
> 更新日期：2026-09-04

本文说明如何在 `microduck_rl` 中新增自己的强化学习动作，包括任务模板选择、目标姿态、命令编码、奖励函数、课程学习、任务注册、测试、训练、ONNX 导出和键盘推理接入。

## 1. 自定义动作由哪些部分组成

一个完整动作通常涉及以下文件：

| 文件 | 作用 |
|------|------|
| `src/mjlab_microduck/tasks/microduck_<action>_env_cfg.py` | 环境、传感器、奖励、事件、课程和 PPO 配置 |
| `src/mjlab_microduck/tasks/mdp.py` | 自定义奖励、重置、命令和课程函数 |
| `src/mjlab_microduck/tasks/__init__.py` | 把新任务注册为可训练的 Task ID |
| `tests/test_<action>_cfg.py` | 检查配置、奖励符号、命令和维度 |
| `tests/test_<action>.py` | 测试纯奖励函数和相位逻辑 |
| `scripts/infer_policy.py` | 可选：增加键盘按键和多策略热切换 |

查看当前任务：

```bash
cd ~/microduck-jetson/microduck_rl
uv run --no-sync list-envs | grep MicroDuck
```

## 2. 开发前必须保持的契约

### 2.1 观测必须保持 61 维

所有可热切换策略共享：

```text
48 维本体观测 + 13 维命令 = 61 维
```

13 维命令顺序固定：

```text
twist(3) + head_pose(4) + body_pose(6)
```

动作不使用某个命令槽时，应将它保持为零填充，不能删除该观测项。否则导出的策略不能和 walking、sitstand 等模型热切换。

### 2.2 动作输出必须保持 14 维

14 个输出对应 Microduck 的 14 个主动舵机。轮子和齿隙铰链属于 `passive_*`，不能被加入策略动作。

### 2.3 不要硬编码关节索引

普通模型、轮滑模型和 backlash 模型的 MuJoCo joint 排列可能不同。自定义 MDP 函数应：

- 使用关节名称查找。
- 或使用 `mdp.py` 中的 `_servo_joint_ids`、`_servo_joint_pos` 等辅助函数。
- 不要假定 MuJoCo `qpos` 的前 14 项永远就是主动舵机。

### 2.4 保留 BAM 和域随机化

Microduck 使用 BAM M6 执行器模型，不是理想 PD 电机。新任务应保留：

- 电池电压和压降模型。
- 执行器延迟。
- 摩擦、质量、惯量、质心和编码器偏置随机化。
- `expand_bam_friction_fields` startup event。
- IMU 安装误差和观测噪声。
- `nan_state` termination。

最安全的方法是从已有 Microduck 任务复制，而不是从空白的 MJLab 环境开始。

### 2.5 ONNX 必须使用项目脚本导出

训练启用了 observation normalization。必须使用：

```text
scripts/export.py
```

它会把归一化器写入 ONNX。不要直接调用 `torch.onnx.export` 转换 checkpoint。

## 3. 根据动作类型选择模板

| 自定义动作类型 | 推荐模板 | 示例 |
|----------------|----------|------|
| 连续速度控制 | `microduck_velocity_env_cfg.py` | 跑步、侧走、慢走、原地转向 |
| 行走并自动恢复 | `microduck_velstand_env_cfg.py` | 跌倒恢复、推搡恢复 |
| 从特定状态到目标姿态 | `microduck_standup_env_cfg.py` | 从趴下起身、翻身后站立 |
| 两状态指令切换 | `microduck_sitstand_env_cfg.py` | 坐/站、蹲/站、趴/站 |
| 按相位完成慢动作 | `microduck_ground_pick_env_cfg.py` | 鞠躬、点头、下蹲、伸展 |
| 快速动态技巧 | `microduck_roulade_env_cfg.py` | 翻滚、跳跃、快速转身 |
| 轮滑连续动作 | `microduck_velocity_rollers_env_cfg.py` | 轮滑前进、转向、滑步 |
| 轮滑相位动作 | `microduck_roller_crouch_env_cfg.py` | 轮滑下蹲、轮滑伸展 |

首次开发推荐选择**按相位完成的慢动作**，因为目标和调试过程更直观。

## 4. 推荐开发流程

不要一开始就运行几千轮训练。推荐顺序：

1. 定义动作成功标准。
2. 在 MuJoCo 中确认目标姿态物理可行。
3. 选择最接近的已有任务。
4. 复制环境配置并注册新 Task ID。
5. 先复用已有通用奖励函数。
6. 为新增数学逻辑编写 CPU 单元测试。
7. 运行 1 个环境的 viewer 检查模型和重置状态。
8. 运行 64 环境、5 轮冒烟训练。
9. 运行 1024 或 2048 环境的短训练。
10. 查看奖励分项，再调整奖励和课程。
11. 长时间训练并导出 ONNX。
12. 接入 `infer_policy.py` 做键盘热切换验证。

## 5. 示例：新增鞠躬 Bow 动作

下面以“从站立缓慢鞠躬、保持、再恢复站立”为例。该动作与 GroundPick 都是相位动作，因此使用 GroundPick 作为模板。

### 5.1 备份源码

当前 Jetson 项目可能来自源码快照而不是 Git clone，编辑前先做备份：

```bash
cd ~/microduck-jetson/microduck_rl

cp src/mjlab_microduck/tasks/__init__.py \
  src/mjlab_microduck/tasks/__init__.py.bak

cp src/mjlab_microduck/tasks/mdp.py \
  src/mjlab_microduck/tasks/mdp.py.bak
```

### 5.2 复制最接近的环境

```bash
cp \
  src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py \
  src/mjlab_microduck/tasks/microduck_bow_env_cfg.py
```

在新文件中至少完成以下重命名：

```text
make_microduck_ground_pick_env_cfg -> make_microduck_bow_env_cfg
MicroduckGroundPickRlCfg          -> MicroduckBowRlCfg
experiment_name="ground_pick"     -> experiment_name="bow"
run_name="ground_pick"            -> run_name="bow"
```

不要直接覆盖原来的 GroundPick 任务，否则已有模型、日志路径和导出命令都会混淆。

### 5.3 设计动作时间线

复用 `GroundPickPhaseCommand`。该命令使用：

```text
[cos(2π phase), sin(2π phase), 0]
```

定义一个 4 秒动作周期：

```python
BOW_PERIOD = 4.0
DESCENT_END = 0.30
HOLD_END = 0.50
RISE_END = 0.80
```

含义：

| Phase | 动作 |
|-------|------|
| `0.00 → 0.30` | 从站立逐渐进入鞠躬姿态 |
| `0.30 → 0.50` | 保持鞠躬 |
| `0.50 → 0.80` | 恢复站立 |
| `0.80 → 1.00` | 稳定站立 |

配置命令：

```python
command = cfg.commands["twist"]
command.rel_standing_envs = 0.0
command.rel_heading_envs = 0.0

cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
    **{
        **vars(command),
        "class_type": microduck_mdp.GroundPickPhaseCommand,
        "period": BOW_PERIOD,
        "randomize_phase": True,
    }
)
```

训练时随机 phase 可以让所有阶段同时获得样本。部署按键触发时通常从 phase 0 开始。

### 5.4 定义目标姿态

使用关节名称定义目标，不要用 MuJoCo joint index：

```python
BOW_POSE = {
    "left_hip_pitch": math.radians(-38.0),
    "left_knee": math.radians(-8.0),
    "left_ankle": math.radians(30.0),
    "neck_pitch": math.radians(28.0),
    "head_pitch": math.radians(38.0),
    "right_hip_pitch": math.radians(38.0),
    "right_knee": math.radians(8.0),
    "right_ankle": math.radians(-30.0),
}
```

这些角度只是开发起点，不是已验证的最终姿态。必须先在 MuJoCo 中检查：

- 双脚是否稳定接触地面。
- 质心是否仍在支撑区域内。
- 头部是否撞地或撞身体。
- 关节是否接近限位。
- 保持目标控制 3 秒后是否仍能站稳。

目标姿态不稳定时，奖励调得再好也无法训练出稳定动作。

### 5.5 首先复用通用相位姿态奖励

`mdp.py` 已提供通用函数：

- `phase_pose_track`：高斯型姿态奖励，接近目标时信号强。
- `phase_pose_track_l1`：负 L1 误差，远离目标时仍有方向性梯度。

在 `make_microduck_bow_env_cfg()` 中配置：

```python
cfg.rewards["bow_pose"] = RewardTermCfg(
    func=microduck_mdp.phase_pose_track,
    weight=3.0,
    params={
        "command_name": "twist",
        "target_pose": BOW_POSE,
        "std": 0.25,
        "descent_end": DESCENT_END,
        "hold_end": HOLD_END,
        "rise_end": RISE_END,
        "asset_cfg": SceneEntityCfg("robot"),
    },
)

cfg.rewards["bow_pose_l1"] = RewardTermCfg(
    func=microduck_mdp.phase_pose_track_l1,
    weight=0.5,
    params={
        "command_name": "twist",
        "target_pose": BOW_POSE,
        "descent_end": DESCENT_END,
        "hold_end": HOLD_END,
        "rise_end": RISE_END,
        "asset_cfg": SceneEntityCfg("robot"),
    },
)
```

注意 `phase_pose_track_l1()` 自身返回负数，因此它的 `weight` 应为正数。不要把负 penalty 再乘负权重，否则会奖励偏离目标。

### 5.6 增加动作结果奖励

只奖励关节姿态容易学成僵硬的“摆造型”。还应奖励动作结果，例如：

- 鞠躬阶段头部高度降低。
- 双脚保持接地。
- 横向倾斜保持较小。
- 恢复阶段重新直立。
- 动作结束后速度接近零。

尽量先复用模板已经创建好的奖励项，例如：

```python
cfg.rewards["upright"].weight = 1.0
cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)

cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=mdp.action_rate_l2,
    weight=-0.1,
)
```

`upright` 来自基础速度环境，复制官方动作模板后通常已经存在；如果你的模板删除过它，应从该模板的基础环境恢复，而不要猜测函数名。其他实际函数名和参数应以当前 `mdp.py` 及模板任务为准。复制模板已有的稳定性、碰撞、关节限位和动作平滑项通常比重新设计更安全。

### 5.7 删除模板中不属于 Bow 的项目

从 GroundPick 复制后，应删除或替换以下 GroundPick 专用内容：

- `mouth_ground_proximity*` 奖励。
- `mouth_perpendicular*` 奖励。
- `head_impact_contact`，如果 Bow 不可能接近地面。
- `mouth_payload_force`。
- `sample_mouth_payload` 和 payload force event。
- GroundPick 专用下降、拾取和回程奖励。

保留：

- 61 维 actor observation。
- critic 的安全观测。
- BAM startup event。
- action history reset。
- `nan_state` termination。
- 质量、质心、摩擦、延迟、IMU 和编码器随机化。
- 关节限位、自碰撞和 action-rate regularizer。

每删除一个 reward/event，都应搜索其名称，确保 curriculum 没有继续引用它：

```bash
grep -n 'mouth\|ground_pick\|payload' \
  src/mjlab_microduck/tasks/microduck_bow_env_cfg.py
```

### 5.8 设置课程学习

动态动作一开始就使用很强的平滑惩罚，可能导致策略完全不动。建议逐步增加：

```python
cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
    func=microduck_mdp.reward_weight,
    params={
        "reward_name": "action_rate_l2",
        "weight_stages": [
            {"step": 0, "weight": -0.05},
            {"step": 500 * 24, "weight": -0.10},
            {"step": 1500 * 24, "weight": -0.30},
        ],
    },
)
```

课程中的 `step` 是环境步数尺度。当前 PPO 每环境每 iteration 收集 24 步，因此常用：

```text
目标 iteration × 24
```

课程建议：

1. 初期使用小姿态幅度、小随机化和轻 regularizer。
2. 策略能完成基本动作后扩大目标幅度。
3. 再逐步增加 CoM、push、摩擦和延迟范围。
4. 最后增加平滑、落地和冲击惩罚。

### 5.9 配置 PPO 和日志目录

新动作应使用独立名称：

```python
MicroduckBowRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="bow",
    run_name="bow",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
```

第一版自定义动作建议关闭 symmetry。只有确认动作左右对称且 symmetry permutation 正确时才启用。

## 6. 注册新任务

编辑：

```text
src/mjlab_microduck/tasks/__init__.py
```

添加 import：

```python
from .microduck_bow_env_cfg import (
    make_microduck_bow_env_cfg,
    MicroduckBowRlCfg,
)
```

添加注册：

```python
register_mjlab_task(
    task_id="Mjlab-Bow-Flat-MicroDuck",
    env_cfg=make_microduck_bow_env_cfg(),
    play_env_cfg=make_microduck_bow_env_cfg(play=True),
    rl_cfg=MicroduckBowRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
```

检查注册：

```bash
uv run --no-sync list-envs | grep Mjlab-Bow
```

如果以后需要齿隙鲁棒版本，再将它加入 `_BACKLASH_TASKS`。第一版先确保普通任务能训练，不要同时引入 backlash 复杂度。

## 7. 何时需要在 mdp.py 新增函数

以下情况才需要新增 MDP 函数：

- 需要新的几何量，例如嘴、脚或头与目标点的距离。
- 需要新的相位 gate。
- 需要新的 reset 初始状态。
- 需要新的 command encoding。
- 需要自定义 curriculum 修改参数。

如果只是跟踪目标关节姿态，应先复用 `phase_pose_track` 和 `phase_pose_track_l1`。

### 7.1 推荐的纯函数结构

把数学逻辑拆成不依赖环境的纯函数：

```python
def bow_height_reward_from_values(
    head_height: torch.Tensor,
    target_height: torch.Tensor,
    std: float,
) -> torch.Tensor:
    error = (head_height - target_height) / std
    return torch.exp(-(error**2))
```

再写环境 wrapper：

```python
def bow_height_reward(
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", site_names=["head_camera"]
    ),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    current_height = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2]
    target = torch.full_like(current_height, target_height)
    return bow_height_reward_from_values(current_height, target, std)
```

这样可以不启动 MuJoCo 就测试数学正确性。

### 7.2 奖励符号约定

常见两种函数：

```text
reward 函数返回正数       → weight 通常为正
penalty/cost 函数返回正数 → weight 通常为负
函数自身返回负误差        → weight 通常为正
```

训练日志中所有 penalty 分项原则上应 `<= 0`。如果某个 penalty 长期为正，优先检查是否发生双重取负。

## 8. 编写配置测试

如果环境中还没有 pytest：

```bash
uv pip install pytest
```

创建：

```text
tests/test_bow_cfg.py
```

示例：

```python
from mjlab_microduck.tasks.mdp import GroundPickPhaseCommand
from mjlab_microduck.tasks.microduck_bow_env_cfg import (
    make_microduck_bow_env_cfg,
)


def test_bow_cfg_builds():
    cfg = make_microduck_bow_env_cfg()
    assert "bow_pose" in cfg.rewards
    assert "bow_pose_l1" in cfg.rewards
    assert "nan_state" in cfg.terminations
    assert "expand_bam_friction_fields" in cfg.events


def test_bow_command_is_phase_encoded():
    cfg = make_microduck_bow_env_cfg()
    assert cfg.commands["twist"].class_type is GroundPickPhaseCommand


def test_bow_reward_signs():
    cfg = make_microduck_bow_env_cfg()
    assert cfg.rewards["bow_pose"].weight > 0
    assert cfg.rewards["bow_pose_l1"].weight > 0
    assert cfg.rewards["action_rate_l2"].weight < 0
```

运行：

```bash
uv run --no-sync python3 -m pytest -q tests/test_bow_cfg.py
```

还应为 phase blend、reward gate 和纯数学 reward 编写边界测试。

## 9. 训练前静态检查

检查 Python 语法：

```bash
uv run --no-sync python3 -m py_compile \
  src/mjlab_microduck/tasks/microduck_bow_env_cfg.py
```

检查模块导入和任务注册：

```bash
uv run --no-sync python3 - <<'PY'
import mjlab_microduck
from mjlab_microduck.tasks.microduck_bow_env_cfg import (
    make_microduck_bow_env_cfg,
)

cfg = make_microduck_bow_env_cfg()
print("actor terms:", tuple(cfg.observations["actor"].terms))
print("rewards:", tuple(cfg.rewards))
print("commands:", tuple(cfg.commands))
print("config build: OK")
PY

uv run --no-sync list-envs | grep Mjlab-Bow
```

## 10. Viewer 检查

先检查环境能否创建，不加载训练策略：

```bash
export DISPLAY=:0
export MUJOCO_GL=glfw

uv run --no-sync play Mjlab-Bow-Flat-MicroDuck \
  --agent random \
  --num-envs 1 \
  --viewer native
```

这一阶段关注：

- 模型是否正常出现。
- 初始姿态是否合理。
- 是否发生穿模或爆炸。
- command phase 是否推进。
- reset 后是否恢复有效状态。

随机 agent 不会完成动作，它只用于验证环境能否运行。

## 11. 训练冒烟测试

```bash
export MUJOCO_GL=egl

uv run --no-sync train Mjlab-Bow-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.logger tensorboard \
  --agent.max_iterations 5
```

必须确认：

- 使用 `device=cuda:0`。
- Actor input 为 61。
- Actor output 为 14。
- 所有 reward term 都能计算。
- `Episode_Termination/nan_state` 为 0。
- 训练正常生成 `model_4.pt` 和 ONNX。

## 12. 分阶段训练

### 12.1 第一阶段：动作发现

```bash
uv run --no-sync train Mjlab-Bow-Flat-MicroDuck \
  --env.scene.num-envs 1024 \
  --agent.logger tensorboard \
  --agent.max_iterations 1000
```

第一阶段使用：

- 较强主任务 reward。
- 较弱 action-rate 和 torque-rate penalty。
- 较小的姿态幅度。
- 较小的 CoM 和 push 随机化。

### 12.2 第二阶段：提高动作完整性

恢复训练至 3000～5000 iteration，逐渐增加：

- 目标姿态幅度。
- 恢复站立奖励。
- 保持时间。
- 动作平滑约束。

### 12.3 第三阶段：提高 sim2real 鲁棒性

再扩大：

- CoM 和质量范围。
- 电机延迟与编码器偏置。
- 地面摩擦范围。
- 外力 push。
- backlash 版本训练或微调。

## 13. 如何判断奖励设计失败

### 13.1 策略完全不动

可能原因：

- 平滑或扭矩 penalty 太强。
- 目标 reward 在初始状态附近没有梯度。
- 高斯 std 太小，远离目标时 reward 接近零。
- command slot 在训练中一直为零。

处理：

- 降低 regularizer。
- 增加负 L1 bootstrap 项。
- 增大高斯 std。
- 从较小动作幅度开始做 curriculum。

### 13.2 策略猛烈撞击目标

可能原因：

- “到达目标”奖励可被高速重复刷取。
- 没有速度、冲击或 overshoot 约束。
- 相位目标瞬间跳变。

处理：

- 使用平滑插值目标。
- 增加冲击和到达速度 penalty。
- 将 action-rate/torque-rate 在技能形成后逐步增强。

### 13.3 到达目标但无法恢复站立

可能原因：

- 只奖励动作前半段。
- return phase 没有同等明确的目标。
- 动作结束后的 standing reward 太弱。

处理：

- 使用完整的 `phase_pose_track` 插值回 HOME。
- 增加 return upright、height 和 stillness reward。
- 延长恢复阶段和站立保持阶段。

### 13.4 总 reward 上升但动作没学会

可能是策略只优化了容易获得的 regularizer 或存活奖励。应单独观察：

- 主任务 reward 是否增长。
- episode length 是否符合任务要求。
- 每个 penalty 是否保持非正。
- 失败 termination 的比例。
- phase 各阶段是否都出现成功样本。

## 14. TensorBoard 分析

```bash
cd ~/microduck-jetson/microduck_rl

uv run --no-sync tensorboard \
  --logdir logs/rsl_rl/bow \
  --host 0.0.0.0 \
  --port 6006
```

浏览器访问：

```text
http://192.168.88.77:6006
```

重点观察：

- `Episode_Reward/bow_pose`
- `Episode_Reward/bow_pose_l1`
- `Episode_Reward/action_rate_l2`
- `Episode_Termination/fell_over`
- `Episode_Termination/nan_state`
- Mean reward
- Mean episode length
- Mean action std

## 15. 查看训练结果

```bash
find logs/rsl_rl/bow -type f -name 'model_*.pt' | sort
```

使用 checkpoint 查看：

```bash
export DISPLAY=:0
export MUJOCO_GL=glfw

uv run --no-sync play Mjlab-Bow-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/bow/训练目录/model_XXXX.pt \
  --num-envs 1 \
  --viewer native
```

## 16. 导出 ONNX

```bash
uv run --no-sync python3 scripts/export.py \
  Mjlab-Bow-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/bow/训练目录/model_XXXX.pt \
  --onnx-file bow.onnx \
  --num-envs 1
```

验证 61→14 契约：

```bash
uv run --no-sync python3 - <<'PY'
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("bow.onnx")
model_input = session.get_inputs()[0]
actions = session.run(
    None,
    {model_input.name: np.zeros((1, 61), dtype=np.float32)},
)[0]

print("input:", model_input.shape)
print("output:", actions.shape)
print("finite:", np.isfinite(actions).all())
assert actions.shape == (1, 14)
assert np.isfinite(actions).all()
PY
```

## 17. 接入键盘推理脚本

训练和 ONNX 验证成功后，再修改：

```text
scripts/infer_policy.py
```

典型改动包括：

1. 增加参数：

```python
parser.add_argument("--bow", help="Path to bow policy ONNX")
```

2. 加载 `bow.onnx` session。
3. 选择一个未占用按键，例如 `V`。
4. 按下时将 phase 从 0 开始写入 twist 的前两个槽：

```text
twist.vx = cos(2π phase)
twist.vy = sin(2π phase)
twist.wz = 0
```

5. phase 完成后切回 standing 或 walking policy。
6. 在启动提示中加入 `V: bow`。

这部分必须与训练时的 `BOW_PERIOD`、结束 phase 和命令槽完全一致。训练使用 4 秒周期，而 runtime 使用 2 秒周期，会导致目标姿态和策略时序错位。

## 18. Backlash 版本

普通任务稳定后，可以注册齿隙版本：

```text
Mjlab-Bow-Flat-Backlash-MicroDuck
```

在 `tasks/__init__.py` 的 `_BACKLASH_TASKS` 中加入：

```python
(
    "Mjlab-Bow-Flat-Backlash-MicroDuck",
    make_microduck_bow_env_cfg,
    {},
    MicroduckBowRlCfg,
    _BL_ALLCOL,
),
```

Bow 使用普通双足全碰撞模型，因此选择 `_BL_ALLCOL`。轮滑动作应使用 `_BL_ROLLERS`，行走模型按现有 velocity 任务选择 `_BL_WALK`。

## 19. 自定义动作检查清单

### 配置

- [ ] 新任务使用独立文件、函数名、Cfg 名和 experiment name。
- [ ] actor observation 仍为 61 维。
- [ ] action 仍为 14 维。
- [ ] 未使用的 command slot 保持零填充。
- [ ] 没有硬编码被动关节混入后的 qpos index。
- [ ] 保留 BAM、NaN guard、观测噪声和关键域随机化。

### 奖励

- [ ] 主任务 reward 明确表示动作成功。
- [ ] 远离目标时仍存在方向性梯度。
- [ ] 正奖励不能通过重复撞击或抖动无限刷取。
- [ ] 所有 penalty 权重符号正确。
- [ ] 前半动作和恢复阶段都有明确目标。
- [ ] 目标姿态经过 MuJoCo 物理验证。

### 验证

- [ ] CPU 配置测试通过。
- [ ] 1 环境 viewer 可启动。
- [ ] 64 环境、5 轮冒烟测试无 NaN。
- [ ] TensorBoard 主任务 reward 确实增长。
- [ ] checkpoint 可以通过 `play` 加载。
- [ ] ONNX 是 61 输入、14 输出。
- [ ] `infer_policy.py` 的 phase、周期和命令槽与训练一致。

## 20. 常用命令汇总

```bash
# 查看任务注册
uv run --no-sync list-envs | grep MicroDuck

# 测试配置
uv run --no-sync python3 -m pytest -q tests/test_bow_cfg.py

# 64 环境冒烟训练
uv run --no-sync train Mjlab-Bow-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.logger tensorboard \
  --agent.max_iterations 5

# 正式训练
uv run --no-sync train Mjlab-Bow-Flat-MicroDuck \
  --env.scene.num-envs 2048 \
  --agent.logger tensorboard

# 查看 checkpoint
uv run --no-sync play Mjlab-Bow-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/bow/训练目录/model_XXXX.pt \
  --num-envs 1 \
  --viewer native

# 导出 ONNX
uv run --no-sync python3 scripts/export.py \
  Mjlab-Bow-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/bow/训练目录/model_XXXX.pt \
  --onnx-file bow.onnx
```
