# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
from mujoco_mpc import agent as agent_lib
import numpy as np
import time

import pathlib

model_path = (
    pathlib.Path(__file__).parent.parent.parent
    / "../../build/mjpc/tasks/trifinger_simplified/task.xml"
)
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)
agent = agent_lib.Agent(task_id="Simplified Trifinger", model=model)

max_traj_length = 300
frame_skip = 20  # only plan after 20 * 0.001 = 0.02s

# set initial state
data.qpos = np.array(
    [0, 0, 0.0325, 1, 0, 0, 0, 0, 0.06, 0.0325, 0, -0.06, 0.0325, 0.06, 0, 0.0325]
)
data.qvel = np.zeros(model.nv)
data.ctrl = np.zeros(model.nu)
data.time = 0.0

# frame buffer
frames = []
FPS = 6 / 0.02

for t in range(max_traj_length):
    agent.set_state(
        time=data.time,
        qpos=data.qpos,
        qvel=data.qvel,
        act=data.act,
        mocap_pos=data.mocap_pos,
        mocap_quat=data.mocap_quat,
        userdata=data.userdata,
    )

    # run planner for num_steps
    num_steps = 10
    for _ in range(num_steps):
        agent.planner_step()

    # set ctrl from agent policy
    data.ctrl = agent.get_action()

    # step
    for _ in range(frame_skip):
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        pixels = renderer.render()
        frames.append(pixels)

# reset
agent.reset()
media.write_video("simplified_trifinger.mp4", frames, fps=FPS)
