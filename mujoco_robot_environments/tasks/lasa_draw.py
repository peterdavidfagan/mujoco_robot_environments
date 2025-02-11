"""Mujoco environment for prototyping LASA dataset drawing motions."""
from abc import abstractmethod
from typing import Optional, Dict
from copy import deepcopy
from pathlib import Path
import random

import mujoco
from mujoco import viewer
from mujoco import mj_name2id, mj_id2name
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation as R
import dm_env
from dm_control import composer, mjcf
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations
import hydra
from hydra.utils import instantiate
from hydra import compose, initialize
from omegaconf import DictConfig

from mujoco_robot_environments.models.arenas import empty
from mujoco_robot_environments.models.robot_arm import standard_compose
from mujoco_robot_environments.environment.props import add_objects, Rectangle
from mujoco_robot_environments.environment.cameras import add_camera
from mujoco_robot_environments.environment.prop_initializer import PropPlacer
from mujoco_robot_environments.models.robot_arm import RobotArm

import h5py

def generate_default_config():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(config_path="../config", job_name="lasa"):
        cfg = compose(
            config_name="lasa",
            overrides=[
                "simulation_tuning_mode=True",
                ]
                )
    return cfg



def generate_deployment_config():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(config_path="../config", job_name="lasa"):
        cfg = compose(
            config_name="lasa",
            overrides=[
                "simulation_tuning_mode=False",
                "robots/arm/actuator_config=position",
                ]
                )
    return cfg


class LasaDrawEnv(dm_env.Environment):
    """MuJoCo powered robotics environment with dm_env interface."""

    def __init__(
        self,
        viewer: Optional = None,
        cfg: DictConfig = generate_default_config(),
    ):
        """Initializes the simulation environment from config."""
        # ensure mjcf paths are relative to this file
        file_path = Path(__file__).parent.absolute()
        self._cfg = cfg
       
        # check if viewer is requested in input args, otherwise use config
        if viewer is not None:
            self.has_viewer = viewer
        elif self._cfg.viewer is None:
            self.has_viewer = False
            print("Viewer not requested, running headless.")
        else:
            self.has_viewer = self._cfg.viewer

        # create arena
        self._arena = empty.Arena()

        # set general physics parameters
        self._arena.mjcf_model.option.timestep = cfg.physics_dt
        self._arena.mjcf_model.option.gravity = cfg.gravity
        self._arena.mjcf_model.size.nconmax = cfg.nconmax
        self._arena.mjcf_model.size.njmax = cfg.njmax
        self._arena.mjcf_model.visual.__getattr__("global").offheight = cfg.offheight
        self._arena.mjcf_model.visual.__getattr__("global").offwidth = cfg.offwidth
        self._arena.mjcf_model.visual.map.znear = cfg.znear


        # add table for manipulation task 
        table = Rectangle(
            name="table",
            x_len=1.0,
            y_len=1.0,
            z_len=0.2,
            rgba=(1.0, 1.0, 1.0, 1.0),
            margin=0.0,
            gap=0.0,
            mass=10,
            )

        table_attach_site = self._arena.mjcf_model.worldbody.add(
            "site",
            name=f"table_center",
            pos=(0.4, 0.0, 0.2),
        )
        self._arena.attach(table, table_attach_site)
    
        # add robot model with actuators and sensors
        self.arm = instantiate(cfg.robots.arm.arm)

        # add cylinder to end effector for non-prehensile manipulation
        cylinder_geom = self.arm._attachment_site.parent.add(
            'geom',
            type='cylinder',
            size=[0.015, 0.05],  # radius and half-height
            pos=[0.0, 0.0, 0.05],
            rgba=[0.02, 0.302, 0.4, 1.0]  # red color
        )

        robot_base_site = self._arena.mjcf_model.worldbody.add(
            "site",
            name="robot_base",
            pos=(0.0, 0.0, 0.4),
        )
        self._arena.attach(self.arm, robot_base_site)


        # if debugging the task environment add mocap for controller eef
        if cfg.simulation_tuning_mode:
            self.eef_target_mocap=self._arena.mjcf_model.worldbody.add(
                    'body',
                    name="eef_target_mocap",
                    mocap="true",
                    pos=[0.6, -0.25, 0.55],
                    quat=R.from_euler('xyz', [180, 180, 0], degrees=True).as_quat(),
                )
            
            self.eef_target_mocap.add(
                'geom',
                name='mocap_target_viz',
                type='box',
                size=[0.025, 0.025, 0.025],  
                rgba=[1.0, 0.0, 0.0, 0.25],  
                pos=[0.0, 0.0, 0.0],
                contype=0,  # no collision with any object
                conaffinity=0  # no influence on collision detection
                )
            
        # add a plane for drawing task
        self.draw_plane_center=self._arena.mjcf_model.worldbody.add(
                'body',
                name="draw_plane",
                mocap="false",
                pos=[0.4, 0.0, 0.55],
                quat=R.from_euler('xyz', [180, 180, 0], degrees=True).as_quat(),
            )
        
        self.draw_plane_center.add(
            'geom',
            name='draw_plane',
            type='box',
            size=[0.3, 0.4, 0.001],  
            rgba=[0.0, 0.0, 1.0, 0.05],  
            pos=[0.0, 0.0, 0.0],
            contype=0,  # no collision with any object
            conaffinity=0  # no influence on collision detection
            )


        # add cameras 
        for camera in cfg.arena.cameras:
            add_camera(
                self._arena,
                name=camera.name,
                pos=camera.pos,
                quat=camera.quat,
                height=camera.height,
                width=camera.width,
                fovy=camera.fovy,
            )

            # this environment uses this camera for observation specification
            if camera.name == "main_camera":
                self.main_camera_height = camera.height
                self.main_camera_width = camera.width
        
        # compile environment
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)
        if self._cfg.renderer:
            self.renderer = mujoco.Renderer(self._physics.model.ptr, height=self.main_camera_height, width=self.main_camera_width)
            self.seg_renderer = mujoco.Renderer(self._physics.model.ptr, height=self.main_camera_height, width=self.main_camera_width)
            self.seg_renderer.enable_segmentation_rendering()
            self.depth_renderer = mujoco.Renderer(self._physics.model.ptr, height=self.main_camera_height, width=self.main_camera_width)
            self.depth_renderer.enable_depth_rendering()
        else:
            print("Renderer not requested, not rendering.")
        self.passive_view = None
                        
    def close(self) -> None:
        if self.passive_view is not None:
            self.passive_view.close()

    @property
    def model(self) -> mujoco.MjModel:
        return self.physics.model

    @property
    def data(self) -> mujoco.MjData:
        return self.physics.data
        
    def reset(self, arm_configuration = None) -> dm_env.TimeStep:
        """Resets the environment to an initial state and returns the first
        `TimeStep` of the new episode.
        """
        if arm_configuration is None:
            arm_configuration = self.arm.named_configurations["home"]

        # reset the simulation instance
        self._physics.reset()
        
        # reset arm to home position
        # Note: for other envs we may want random sampling of initial arm positions
        self.arm.set_joint_angles(self._physics, arm_configuration)

        # configure viewer
        if self.has_viewer:
            if self.passive_view is not None:
                self.passive_view.close()
            self.passive_view = viewer.launch_passive(self._physics.model.ptr, self._physics.data.ptr)

        # create an instance of a robot interface for robot and controllers
        self._robot = RobotArm(
                arm=self.arm, 
                physics=self._physics,
                passive_viewer=self.passive_view,
                )
        
        # set the initial eef pose to home
        self.eef_home_pose = self._robot.eef_pose.copy()
        
        return dm_env.TimeStep(
                step_type=dm_env.StepType.FIRST,
                reward=0.0,
                discount=0.0,
                observation=self._compute_observation(),
                )

    def step(self, action_dict) -> dm_env.TimeStep:
        """
        Updates the environment according to the action and returns a `TimeStep`.
        """
        observation = self._compute_observation()

        return dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=0.0,
                discount=0.0,
                observation=observation,
            )

    def observation_spec(self) -> dm_env.specs.Array:
        """Returns the observation spec."""
        # get shape of main camera
        camera = self._arena.mjcf_model.find("camera", "main_camera/main_camera")
        camera_shape = self.main_camera_height, self.main_camera_width, 3
        return {
                "main_camera/depth": dm_env.specs.Array(shape=camera_shape[:-1], dtype=np.float32),
                "main_camera/rgb": dm_env.specs.Array(shape=camera_shape, dtype=np.float32),
                }

    def action_spec(self) -> Dict[str, dm_env.specs.Array]:
        """Returns the action spec."""
        return {
                "pose": dm_env.specs.Array(shape=(7,), dtype=np.float64), # [x, y, z, qx, qy, qz, qw]
                "pixel_coords": dm_env.specs.Array(shape=(2,), dtype=np.int64), # [u, v]
                "gripper_rot": dm_env.specs.Array(shape=(1,), dtype=np.float64),
                }

    def _compute_observation(self) -> np.ndarray:
        """Returns the observation."""
        # get main camera
        camera_id = mj_name2id(self._physics.model.ptr, mujoco.mjtObj.mjOBJ_CAMERA, "main_camera/main_camera")
        self.renderer.update_scene(self._physics.data.ptr, camera_id)
        self.depth_renderer.update_scene(self._physics.data.ptr, camera_id)
        
        # get rgb data
        rgb = self.renderer.render()

        # get depth data
        depth = self.depth_renderer.render()
        
        # add to observation
        obs = {}
        obs["main_camera/rgb"] = rgb
        obs["main_camera/depth"] = depth

        return obs

    def interactive_tuning(self):
        """
        Interactively control arm to tune simulation parameters. 
        """

        # get difference between eef site and mocap body
        mocap_pos = self._physics.data.mocap_pos[0]
        mocap_quat = self._physics.data.mocap_quat[0]

        # update control target
        self._robot.arm_controller.set_target(
            position=mocap_pos + [0.0, 0.0, 0.1],
            quat=mocap_quat, 
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            )

        control_command = self._robot.arm_controller.compute_control_output()

        # step the simulation
        for _ in range(5):
            self._physics.set_control(control_command)
            self._physics.step()
            if self.passive_view is not None:
                self.passive_view.sync()
    
    def move_to_draw_target(self, target_position, target_velocity):
        """
        Move to position and velocity target for drawing task. 
        """
        mocap_quat = self._physics.data.mocap_quat[0]

        # update control target
        self._robot.arm_controller.set_target(
            position=target_position + [0.0, 0.0, 0.1], # account for height of toolpiece
            quat=mocap_quat, 
            velocity=target_velocity,
            angular_velocity=np.zeros(3),
            )

        control_command = self._robot.arm_controller.compute_control_output()

        # step the simulation
        for _ in range(5):
            self._physics.set_control(control_command)
            self._physics.step()
            if self.passive_view is not None:
                self.passive_view.sync()

        # return joint data for recording
        return self._physics.bind(self._robot.arm_joints).qpos.copy(), self._physics.bind(self._robot.arm_joints).qvel.copy(), self._physics.bind(self._robot.arm_actuators).ctrl.copy()

    def move_to_joint_position_target(self, target_position):
        """
        Move to position and velocity target for drawing task. 
        """
        scale = 0
        # sample random force and torque to apply to arm
        force = np.random.uniform(-scale, scale, size=3) 
        # force = np.array([scale, 0, 0])
        torque = np.random.uniform(-scale, scale, size=3)

        # step the simulation
        for _ in range(5):
            self._physics.set_control(target_position)
            self._physics.data.xfrc_applied[12] = np.concatenate([force, torque])
            self._physics.step()
            if self.passive_view is not None:
                self.passive_view.sync()

        return self._physics.bind(self._robot.arm_joints).qpos.copy()


if __name__=="__main__":
    # instantiate color separation task
    env = LasaDrawEnv(viewer=True) 

    # interactive control of robot with mocap body
    _, _, _, obs = env.reset()
    while True:
        env.interactive_tuning()
    env.close()
    
