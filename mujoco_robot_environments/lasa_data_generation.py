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
    CONFIG = compose(
            config_name="rearrangement",
            overrides=[
                "arena/props=colour_splitter",
                "simulation_tuning_mode=True"
                ]
                )

    # Leverage demonstrations from LASA dataset
    import pyLasaDataset as lasa
    shapes = ["CShape", "GShape", "JShape", "LShape", "NShape", "PShape", "RShape", "Sshape", "WShape", "Zshape"]
    
    def transform_to_workspace(trajectories, workspace_bounds):
        """
        Transform trajectories to fit within a workspace while maintaining aspect ratio.
        """
        # Extract workspace
        workspace_x_min, workspace_x_max, workspace_y_min, workspace_y_max = workspace_bounds

        # Extract range of demonstrations
        stack = np.hstack([d.pos for d in demos])
        original_x_min, original_y_min = np.min(stack, axis=1)
        original_x_max, original_y_max = np.max(stack, axis=1)
        
        # Compute original width and height
        original_width = original_x_max - original_x_min
        original_height = original_y_max - original_y_min
        
        # Compute workspace width and height
        workspace_width = workspace_x_max - workspace_x_min
        workspace_height = workspace_y_max - workspace_y_min
        
        # Compute scaling factors
        scale_x = workspace_width / original_width
        scale_y = workspace_height / original_height
        
        # Use the smaller scaling factor to maintain aspect ratio
        scale = min(scale_x, scale_y)
        
        # Compute offsets to center trajectories in the workspace
        offset_x = workspace_x_min + (workspace_width - (original_width * scale)) / 2
        offset_y = workspace_y_min + (workspace_height - (original_height * scale)) / 2
        
        # Transform each trajectory
        transformed_position_trajectories = []
        transformed_velocity_trajectories = []
        for traj in trajectories:
            pos = traj.pos
            vel = traj.vel            
            
            # Scale and shift
            transformed_x = (pos[0, :] - original_x_min) * scale + offset_x
            transformed_y = (pos[1, :] - original_y_min) * scale + offset_y
            z = np.repeat(0.55, pos.shape[1]) 
            transformed_position_trajectories.append(np.vstack((transformed_x, transformed_y, z)))

            # scale
            vel_z = np.repeat(0.0, pos.shape[1]) 
            transformed_velocity_trajectories.append(np.vstack([vel * scale / 800, vel_z])) # TODO: formalize velocity scaling
        
        return np.array(transformed_position_trajectories), np.array(transformed_velocity_trajectories)

    env = LasaDrawEnv(cfg=CONFIG)
    _, _, _, obs = env.reset()

    # interactive control of robot with mocap body
    data = {}
    for char in shapes:
        data[f"{char}"] = {}
        demos = lasa.DataSet.__getattr__(char).demos
        pos, vel = transform_to_workspace(demos, workspace_bounds=[0.3, 0.6, -0.3, 0.3])
        for demo_idx, (positions, velocities) in enumerate(zip(pos, vel)):
            data[f"{char}"][f"trajectory_{demo_idx}"] = {}
            print(f"Processing {char} demo {demo_idx}...")
            joint_positions, joint_velocities, joint_torques = [], [], []
            for idx, (target_pos, target_vel) in enumerate(zip(positions.T, velocities.T)):
                while True:
                    pos, vel, torque = env.move_to_draw_target(target_pos, target_vel)

                    if idx != 0: 
                        joint_positions.append(pos)
                        joint_velocities.append(vel)
                        joint_torques.append(torque)

                    # check if target is reached
                    if env._robot.arm_controller.current_position_error() < 5e-3:
                        break
                    
                    if CONFIG.viewer:
                        with env.passive_view.lock():
                            env.passive_view.user_scn.ngeom += 1
                            mujoco.mjv_initGeom(
                                env.passive_view.user_scn.geoms[env.passive_view.user_scn.ngeom - 1],
                                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                size=[0.001, 0, 0],
                                pos=target_pos,
                                mat=np.eye(3).flatten(),
                                rgba=[1, 0, 0, 1]
                            )
                            env.passive_view.sync()

            # env.close()
            data[f"{char}"][f"trajectory_{demo_idx}"]["joint_positions"] = np.vstack(joint_positions)
            data[f"{char}"][f"trajectory_{demo_idx}"]["joint_velocities"] = np.vstack(joint_velocities)
            data[f"{char}"][f"trajectory_{demo_idx}"]["joint_torques"] = np.vstack(joint_torques)
            
            
    with h5py.File("robot_trajectories.h5", "w") as f:
        for character, trajectories in data.items():
            char_group = f.create_group(character)  # Create a group for each character
            for traj_id, traj_data in trajectories.items():
                traj_group = char_group.create_group(traj_id)  # Create a group for each trajectory ID
                traj_group.create_dataset("position", data=traj_data["joint_positions"], compression="gzip")
                traj_group.create_dataset("velocity", data=traj_data["joint_velocities"], compression="gzip")
                traj_group.create_dataset("torque", data=traj_data["joint_torques"], compression="gzip")

    env.close()
    