"""Interactive editor for the Microduck left-foot one-leg balance pose."""

import math
import re
import time

import mujoco
import mujoco.viewer

from mjlab_microduck.robot.microduck_constants import HOME_FRAME, get_standup_spec
from mjlab_microduck.tasks.microduck_one_leg_balance_env_cfg import ONE_LEG_POSE


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
    data.ctrl[actuator_id] = ONE_LEG_POSE.get(name, home_value(name))

roll = math.radians(-10.0)
data.qpos[0:3] = (0.0, 0.0, 0.16)
data.qpos[3:7] = (math.cos(roll / 2.0), math.sin(roll / 2.0), 0.0, 0.0)
for name, qpos_address in joints:
    data.qpos[qpos_address] = ONE_LEG_POSE.get(name, home_value(name))

base_xy = data.qpos[0:2].copy()
base_quat = data.qpos[3:7].copy()
robot_geoms = list(range(model.ngeom))
mujoco.mj_forward(model, data)

print("=== Microduck One-Leg Balance Pose Editor ===")
print("Left foot supports the robot; right foot is lifted.")
print("Open the MuJoCo Control panel to adjust joint sliders.")
print("Close the window to print the final named pose.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 145
    viewer.cam.elevation = -12
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
            float(data.geom_xpos[g, 2] - model.geom_rbound[g]) for g in robot_geoms
        )
        data.qpos[2] -= z_min
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(1.0 / 60.0)

print("\nONE_LEG_POSE = {")
for name, qpos_address in joints:
    print(f'    "{name}": {float(data.qpos[qpos_address]):.4f},')
print("}")
