"""Microduck front-back split stance task."""

import math
from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

SPLIT_PERIOD = 6.0
SPLIT_END = 0.30
HOLD_END = 0.58
RETURN_END = 0.78
EPISODE_LENGTH_S = 12.0
TARGET_SAGITTAL_SEPARATION = 0.095

FRONT_BACK_SPLIT_POSE = {
    "left_hip_pitch": -1.1865,
    "left_knee": -0.1386,
    "left_ankle": 1.0452,
    "right_hip_pitch": 0.0603,
    "right_knee": 0.4927,
    "right_ankle": 0.4293,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
}


def make_microduck_front_back_split_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg = make_microduck_velocity_env_cfg(play=play)
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground, self_collision)
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S

    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = 1.0

    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
        "head_pose_tracking",
        "body_pose_tracking",
        "head_pose_bias",
    ):
        cfg.rewards.pop(name, None)

    pose_params = {
        "command_name": "twist",
        "target_pose": FRONT_BACK_SPLIT_POSE,
        "descent_end": SPLIT_END,
        "hold_end": HOLD_END,
        "rise_end": RETURN_END,
        "asset_cfg": SceneEntityCfg("robot"),
    }
    cfg.rewards["split_pose"] = RewardTermCfg(
        func=microduck_mdp.phase_pose_track,
        weight=5.0,
        params={**pose_params, "std": 0.28},
    )
    cfg.rewards["split_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.phase_pose_track_l1,
        weight=0.8,
        params=pose_params,
    )
    cfg.rewards["feet_grounded"] = RewardTermCfg(
        func=microduck_mdp.feet_grounded_reward,
        weight=3.0,
        params={"sensor_name": feet_ground.name},
    )
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
        },
    )
    cfg.rewards["sagittal_separation"] = RewardTermCfg(
        func=microduck_mdp.phase_sagittal_foot_separation_track,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", site_names=("left_foot", "right_foot")
            ),
            "command_name": "twist",
            "target_separation": TARGET_SAGITTAL_SEPARATION,
            "std": 0.025,
            "split_end": SPLIT_END,
            "hold_end": HOLD_END,
            "return_end": RETURN_END,
        },
    )
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 0.7
    cfg.rewards["upright"].params["std"] = math.sqrt(0.12)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.1
    cfg.rewards["action_rate_l2"].weight = -0.15
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision.name},
    )

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4}
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6}
        )
    cfg.observations["critic"].terms.pop("foot_height", None)

    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    phase_fields = {
        field.name for field in fields(microduck_mdp.GroundPickPhaseCommandCfg)
    }
    phase_kwargs = {
        name: value for name, value in vars(command).items() if name in phase_fields
    }
    phase_kwargs.update(
        class_type=microduck_mdp.GroundPickPhaseCommand,
        period=SPLIT_PERIOD,
        randomize_phase=not play,
    )
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(**phase_kwargs)

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )
    cfg.events["foot_friction"].params["ranges"] = (0.8, 1.3)
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.03, 0.03)
    cfg.events.pop("push_robot", None)

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("command_vel", None)
    cfg.curriculum.pop("standing_envs", None)
    cfg.curriculum.pop("head_pose_range", None)
    cfg.curriculum.pop("body_pose_range", None)
    cfg.curriculum.pop("head_pose_bias_weight", None)
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.15},
                {"step": 500 * 24, "weight": -0.4},
                {"step": 1000 * 24, "weight": -0.8},
            ],
        },
    )
    return cfg


MicroduckFrontBackSplitRlCfg = RslRlOnPolicyRunnerCfg(
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
        hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True
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
    experiment_name="front_back_split",
    run_name="front_back_split_left_forward",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=1_000,
)
