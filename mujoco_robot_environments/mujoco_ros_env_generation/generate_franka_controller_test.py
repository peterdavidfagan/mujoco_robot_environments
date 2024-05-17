"""A script to generate the XML file for the model."""

import os
import numpy as np
import mujoco
from dm_control import mjcf

from mujoco_robot_environments.tasks.rearrangement import RearrangementEnv

import hydra
from hydra import compose, initialize

# clear hydra global state to avoid conflicts with other hydra instances
hydra.core.global_hydra.GlobalHydra.instance().clear()
initialize(version_base=None, config_path="../config", job_name="default_config")

# define known task configurations
COLOR_SEPARATING_CONFIG = compose(
            config_name="rearrangement",
            overrides=[
                "physics_dt=5e-4",
                "control_dt=1e-2",
                "arena/props=colour_splitter",
                "robots/arm/actuator_config=motor",
                ]
            )

if __name__ == "__main__":
    # generate the XML file for the model
    env = RearrangementEnv(COLOR_SEPARATING_CONFIG)
    env.reset()
    env._physics.step()

    # save as mjb
    model_size = mujoco.mj_sizeModel(physics.model.ptr)
    buffer = np.empty(model_size, dtype=np.uint8)
    mujoco.mj_saveModel(physics.model.ptr, os.path.join(os.getcwd(), "franka_controller_test.mjb"), None)
