"""A script to generate the XML file for the model."""

import os
import time
import numpy as np
import mujoco
from dm_control import mjcf

from mujoco_robot_environments.tasks.rearrangement import RearrangementEnv, DEFAULT_CONFIG, APPLE_CONFIG

if __name__ == "__main__":
    # generate the XML file for the model
    env = RearrangementEnv(cfg=APPLE_CONFIG, viewer=True)
    env.reset()
    env._physics.step()
    
    # ensure cubes are in the correct position
    mjcf_model = env._arena.mjcf_model
    geoms = [geom for geom in mjcf_model.find_all("geom") if geom.name is not None and "apple" in geom.name]
    for geom in geoms:
        geom.pos = env._physics.named.data.geom_xpos[f"{geom.name}/{geom.name}"]
    physics = mjcf.Physics.from_mjcf_model(mjcf_model)

    # save as mjb
    model_size = mujoco.mj_sizeModel(physics.model.ptr)
    buffer = np.empty(model_size, dtype=np.uint8)
    mujoco.mj_saveModel(physics.model.ptr, os.path.join(os.getcwd(), "rearrangement.mjb"), None)

    # sleep to allow for inspection of saved environment
    time.sleep(10.0)