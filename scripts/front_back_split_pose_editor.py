"""Interactive editor for the Microduck front-back split stance."""

import re
import time

import mujoco
import mujoco.viewer

from mjlab_microduck.robot.microduck_constants import HOME_FRAME, get_standup_spec
from mjlab_microduck.tasks.microduck_front_back_split_env_cfg import (
    FRONT_BACK_SPLIT_POSE,
)


def home_value(joint_name: str) -> float:
    for pattern, value in HOME_FRAME.joint_pos.items():
        if re.search(pattern, joint_name):
            return float(value)
    return 0.0


model = get_standup_spec().compile()
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
model.opt.gravity[:] = (0.0, 0.0, 0.0)

joints = []
for joint_id in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if not name or "freejoint" in name or "passive_" in name:
        continue
    joints.append((name, model.jnt_qposadr[joint_id]))

for actuator_id in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    data.ctrl[actuator_id] = FRONT_BACK_SPLIT_POSE.get(name, home_value(name))

data.qpos[0:3] = (0.0, 0.0, 0.16)
data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
for name, qpos_address in joints:
    data.qpos[qpos_address] = FRONT_BACK_SPLIT_POSE.get(name, home_value(name))

base_xy = data.qpos[0:2].copy()
base_quat = data.qpos[3:7].copy()
robot_geoms = list(range(model.ngeom))
mujoco.mj_forward(model, data)

print("=== Microduck Front-Back Split Pose Editor ===")
print("Left foot is forward; right foot is backward; both feet remain grounded.")
print("Open the MuJoCo Control panel to adjust joint sliders.")
print("Close the window to print the final named pose.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -8
    viewer.cam.distance = 0.55
    viewer.cam.lookat[:] = (0.0, 0.0, 0.11)
    while viewer.is_running():
        data.qpos[0:2] = base_xy
        data.qpos[3:7] = base_quat
        data.qvel[0:6] = 0.0
        mujoco.mj_step(model, data)
        data.qpos[0:2] = base_xy
        data.qpos[3:7] = base_quat
        data.qvel[0:6] = 0.0
        mujoco.mj_forward(model, data)
        z_min = min(
            float(data.geom_xpos[geom_id, 2] - model.geom_rbound[geom_id])
            for geom_id in robot_geoms
        )
        data.qpos[2] -= z_min
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(1.0 / 60.0)

print("\nFRONT_BACK_SPLIT_POSE = {")
for name, qpos_address in joints:
    print(f'    "{name}": {float(data.qpos[qpos_address]):.4f},')
print("}")
