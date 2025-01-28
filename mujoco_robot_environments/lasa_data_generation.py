"""Collecting robot demonstrations of LASA drawing dataset."""

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
                "simulation_tuning_mode=True"
                ]
                )

    # instantiate color separation task
    env = LasaDrawEnv(viewer=True, cfg=COLOR_SEPARATING_CONFIG) 

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
   
    # interactive control of robot with mocap body
    _, _, _, obs = env.reset()
    data = {}
    for demo_idx, demo in enumerate(demos):
        positions, velocities = preprocess_demo(demo)
        joint_positions, joint_velocities, joint_torques = [], [], []
        data[f"trajectory_{demo_idx}"] = {}

        for idx, (target_pos, target_vel) in enumerate(zip(positions, velocities)):
            while True:
                pos, vel, torque = env.move_to_draw_target(target_pos, target_vel)

                if idx != 0: 
                    joint_positions.append(pos)
                    joint_velocities.append(vel)
                    joint_torques.append(torque)

                # check if target is reached
                if env._robot.arm_controller.current_position_error() < 5e-3:
                    break
                
                with env.passive_view.lock():
                    env.passive_view.user_scn.ngeom += 1
                    mujoco.mjv_initGeom(
                        env.passive_view.user_scn.geoms[env.passive_view.user_scn.ngeom-1],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.001, 0, 0],
                        pos=target_pos,
                        mat=np.eye(3).flatten(),
                        rgba=[1, 0, 0, 1]
                    )
                    env.passive_view.sync()
                
        data[f"trajectory_{demo_idx}"]["joint_positions"] = np.vstack(joint_positions)
        data[f"trajectory_{demo_idx}"]["joint_velocities"] = np.vstack(joint_velocities)
        data[f"trajectory_{demo_idx}"]["joint_torques"] = np.vstack(joint_torques)
    
    # Save to HDF5
    with h5py.File("robot_trajectories.h5", "w") as f:
        for traj_name, data in data.items():
            group = f.create_group(traj_name)
            group.create_dataset("position", data=data["joint_positions"], compression="gzip")
            group.create_dataset("velocity", data=data["joint_velocities"], compression="gzip")
            group.create_dataset("torque", data=data["joint_torques"], compression="gzip")
            
    env.close()
    