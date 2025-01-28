"""Collecting robot demonstrations of LASA drawing dataset."""

import jax 
import jax.numpy as jnp
import jax.export as export
import numpy as np
import mujoco

import pyLasaDataset as lasa
import h5py

from mujoco_robot_environments.tasks.lasa_draw import LasaDrawEnv

import hydra
from hydra import compose, initialize


if __name__=="__main__":
    # clear hydra global state to avoid conflicts with other hydra instances
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="./config", job_name="rearrangement")
    
    # add task configs
    COLOR_SEPARATING_CONFIG = compose(
            config_name="rearrangement",
            overrides=[
                "arena/props=colour_splitter",
                "simulation_tuning_mode=False",
                "robots/arm/actuator_config=position",
                ]
                )

    # load demo data to evaluate against
    def h5_to_dict(h5_group):
        """
        Recursively convert an HDF5 group or dataset into a nested dictionary.
        """
        result = {}
        for key, item in h5_group.items():
            if isinstance(item, h5py.Group):  # If it's a group, recurse
                result[key] = h5_to_dict(item)
            elif isinstance(item, h5py.Dataset):  # If it's a dataset, load it
                result[key] = item[:]
        return result
    
    with h5py.File("./robot_trajectories.h5", "r") as f:
        data = h5_to_dict(f)

    starting_joint_position = data["trajectory_0"]["position"][0]
    current_joint_position = starting_joint_position

    # load the trained model
    with open("flax_apply_method.bin", "rb") as f:
        serialized_from_file = f.read()
    model = export.deserialize(serialized_from_file)
    dynamics_state = jnp.zeros((1, 5000))
    
    # instantiate the task
    env = LasaDrawEnv(viewer=True, cfg=COLOR_SEPARATING_CONFIG) 
    _, _, _, obs = env.reset(current_joint_position)

    # draw the demonstration trajectory
    # Leverage demonstrations from LASA dataset
    import pyLasaDataset as lasa
    s_data = lasa.DataSet.Sshape
    demos = s_data.demos 

    def preprocess_demo(demo_data):
        pos = demo_data.pos
        vel = demo_data.vel 

        # scale position data
        pos_scaled =  (pos / 200) + 0.2
        pos_scaled = pos_scaled[:,::4]
        positions = np.vstack([pos_scaled[1,:] + 0.2, -pos_scaled[0,:] + 0.2, np.repeat(0.55, pos_scaled.shape[1])]).T

        # scale velocity data
        vel_scaled =  (vel / 800)
        vel_scaled = vel_scaled[:,::4]
        velocities = np.vstack([vel_scaled[1,:], vel_scaled[0,:], np.repeat(0.0, pos_scaled.shape[1])]).T

        return positions, velocities
    
    positions, velocities = preprocess_demo(demos[0])

    for target_position in positions:
        with env.passive_view.lock():
            env.passive_view.user_scn.ngeom += 1
            mujoco.mjv_initGeom(
                env.passive_view.user_scn.geoms[env.passive_view.user_scn.ngeom-1],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.001, 0, 0],
                pos=target_position,
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 1]
            )
            env.passive_view.sync()

    while True:
        # make a prediction using the model
        position_target, dynamics_state = model.call(jnp.expand_dims(current_joint_position, axis=0), dynamics_state)

        # pass the target to position actuators
        current_joint_position = env.move_to_joint_position_target(position_target[0, :7])

        # get eef xpos 
        eef_pos = env._robot.arm_controller.current_eef_position - np.array([0.0, 0.0, 0.1])

        # draw the current state
        with env.passive_view.lock():
            env.passive_view.user_scn.ngeom += 1
            mujoco.mjv_initGeom(
                env.passive_view.user_scn.geoms[env.passive_view.user_scn.ngeom-1],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.001, 0, 0],
                pos=eef_pos,
                mat=np.eye(3).flatten(),
                rgba=[0, 1, 0, 1]
            )
            env.passive_view.sync()
                    
    env.close()
    