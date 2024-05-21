"""Mujoco environment for interactive task learning."""
from abc import abstractmethod
from functools import partial
from typing import Optional, Dict
from copy import deepcopy
from pathlib import Path
import random

import mujoco
from mujoco import mj_name2id, mj_id2name
from mujoco import mjx
from mujoco.mjx._src.support import jac, full_m
import dm_env
from dm_control import composer, mjcf
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from mujoco_robot_environments.models.arenas import empty
from mujoco_robot_environments.models.robot_arm import standard_compose
from mujoco_robot_environments.environment.props import add_objects, Rectangle
from mujoco_robot_environments.environment.cameras import add_camera
from mujoco_robot_environments.environment.prop_initializer import PropPlacer
from mujoco_robot_environments.models.robot_arm import RobotArm

import hydra
from hydra.utils import instantiate
from hydra import compose, initialize
from omegaconf import DictConfig

def generate_default_config():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="../config", job_name="rearrangement")
    return compose(
            config_name="rearrangement",
            overrides=[
                "arena/props=colour_splitter",
                "arena/cameras=rearrangement",
                "simulation_tuning_mode=False",
                ]
                )

def mul_quat(q1, q2):
    """
    A utility function to multiply quaternions. 
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jnp.array([w, x, y, z])

@partial(jax.jit)
def compute_osc_control(
    target_position,
    target_quat, 
    target_velocity, 
    target_angular_velocity,
    data,
    model, 
    nullspace_configuration,  
    eef_site_id, 
    eef_body_id,
    arm_joint_ids
    ):
    """
    A mjx method for computing operational space control signal. 
    """        
    # compute eef jacobian (seems to be different convention than regular mujoco API)
    jacp, jacr = jac(
        model, 
        data, 
        data.site_xpos[eef_site_id, :], # point
        eef_body_id, # bodyid
    )
    jacp = jacp[arm_joint_ids, :].T # filter jacobian for joints we care about
    jacr = jacr[arm_joint_ids, :].T # filter jacobian for joints we care about
    eef_jacobian = jnp.vstack([jacp, jacr])

    # compute eef mass matrix
    mass_matrix = full_m(model, data)
    arm_mass_matrix = mass_matrix[arm_joint_ids, :][:, arm_joint_ids] # filter for links we care about

    mass_matrix_inv = jnp.linalg.inv(arm_mass_matrix)
    mass_matrix_inv = jnp.dot(eef_jacobian, jnp.dot(mass_matrix_inv, eef_jacobian.T))
    eef_mass_matrix = jnp.linalg.pinv(mass_matrix_inv, rcond=1e-2) # TODO: consider switching to inv
    
    # get current end-effector state variables
    eef_position = data.xpos[eef_site_id] 
    eef_quat = data.xquat[eef_site_id]
    eef_velocity = eef_jacobian[:3, :] @ jnp.take_along_axis(data.qvel, arm_joint_ids, axis=0)
    eef_angular_velocity = eef_jacobian[3:, :] @ jnp.take_along_axis(data.qvel, arm_joint_ids, axis=0)

    # compute position error 
    position_error = target_position - eef_position

    # compute orientation error
    eef_quat_conjugate = jnp.array([-eef_quat[0], -eef_quat[1], -eef_quat[2], eef_quat[3]])
    orientation_error = mul_quat(target_quat, eef_quat_conjugate)
    orientation_error = jnp.sign(orientation_error[-1]) * orientation_error[:-1]

    # compute velocity error
    velocity_error = target_velocity - eef_velocity

    # compute angular velocity error
    angular_velocity_error = target_angular_velocity - eef_angular_velocity

    # pd term for position and orientation
    position_pd = (200.0 * position_error) + (30.0 * velocity_error)
    orientation_pd = (500.0 * orientation_error) + (100.0 * angular_velocity_error)
    pd_error = jnp.hstack([position_pd, orientation_pd])

    # compute control signal
    nullspace_position_error = nullspace_configuration - jnp.take_along_axis(data.qpos, arm_joint_ids, axis=0)
    nullspace_velocity_error = jnp.zeros((7,)) - jnp.take_along_axis(data.qvel, arm_joint_ids, axis=0)
    nullspace_pd = (200.0 * nullspace_position_error) + (30.0 * nullspace_velocity_error)
    null_jacobian = jnp.linalg.inv(arm_mass_matrix) @ eef_jacobian.T @ eef_mass_matrix
        

    tau = eef_jacobian.T @ eef_mass_matrix @ pd_error # pd control against eef target
    tau += (jnp.eye(7) - eef_jacobian.T @ null_jacobian.T) @ nullspace_pd # nullspace projection
    tau += jnp.take_along_axis(data.qfrc_bias, arm_joint_ids, axis=0) # compensate for external forces

    # compute effective torque through compensating for actuator moment
    actuator_moment_inv = jnp.linalg.pinv(data.actuator_moment)
    actuator_moment_inv = actuator_moment_inv[arm_joint_ids, :][:, arm_joint_ids]
    tau = tau @ actuator_moment_inv 

    return tau

class RearrangementEnv(dm_env.Environment):
    """MuJoCo powered robotics environment with dm_env interface."""

    def __init__(
        self,
        cfg: DictConfig = generate_default_config(),
    ):
        """Initializes the simulation environment from config."""
        # ensure mjcf paths are relative to this file
        file_path = Path(__file__).parent.absolute()
        cfg.robots.arm.arm.mjcf_path = str(file_path / cfg.robots.arm.arm.mjcf_path)
        cfg.robots.end_effector.end_effector.mjcf_path = str(file_path / cfg.robots.end_effector.end_effector.mjcf_path)
        self._cfg = cfg

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
        x_len=0.9,
        y_len=1.0,
        z_len=0.2,
        rgba=(0.5, 0.5, 0.5, 1.0),
        margin=0.0,
        gap=0.0,
        mass=10,
        )
        table_attach_site = self._arena.mjcf_model.worldbody.add(
            "site",
            name="table_center",
            pos=(0.4, 0.0, 0.2),
        )
        self._arena.attach(table, table_attach_site)

        # add robot model with actuators and sensors
        self.arm = instantiate(cfg.robots.arm.arm)

        # while testing turn off mesh collision geoms so I can iterate quickly 
        # TODO: refine franka emika panda meshes as currently not coarse enough.
        geoms = self.arm.mjcf_model.find_all('geom')
        for geom in geoms:
            geom.contype = 0
            geom.conaffinity=0
        
        # Comment: robotiq gripper results in an error as tendons are not supported.
        # self.end_effector = instantiate(cfg.robots.end_effector.end_effector)
        # standard_compose(arm=self.arm, gripper=self.end_effector)

        # Add the capsule geometry to eef for non-prehensile manipulation (until robotiq is supported)
        cylinder_geom = self.arm._attachment_site.parent.add(
            'geom',
            type='capsule',
            size=[0.015, 0.05],  # radius and half-height
            pos=[0.0, 0.0, 0.05],
            rgba=[1, 0, 0, 0.7]  # red color
        )
        
        robot_base_site = self._arena.mjcf_model.worldbody.add(
            "site",
            name="robot_base",
            pos=(0.0, 0.0, 0.4),
        )
        self._arena.attach(self.arm, robot_base_site)   
        
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
            if camera.name == "front_camera":
                self.camera_height = camera.height
                self.camera_width = camera.width
        
        # compile environment
        self._arena.mjcf_model.option.integrator = 'EULER'
        self._arena.mjcf_model.option.cone = 'PYRAMIDAL'
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # get arm joint ids and eef site id
        self.arm_joint_ids = []
        for joint in self.arm.joints:
            self.arm_joint_ids.append(mj_name2id(self._physics.model.ptr, 3, 'panda nohand/' + joint.name))
        self.arm_joint_ids = jnp.array(self.arm_joint_ids)
        self.eef_site_id = mj_name2id(self._physics.model.ptr, 6, 'panda nohand/attachment_site')
        self.eef_body_id = self._physics.model.ptr.site_bodyid[self.eef_site_id]

        # nullspace joint configuration for osc controller
        self.nullspace_config = jnp.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

        # define renderer
        self.renderer = mujoco.Renderer(self._physics.model.ptr)

        # put model on device
        self.mjx_model = mjx.put_model(self._physics.model.ptr) 

    
    @partial(jax.jit, static_argnums=(0,))
    @partial(jax.vmap, in_axes=(None, 0))
    def reset(self, qpos) -> dm_env.TimeStep:
        """
        Resets the environment to an initial state and returns the first `TimeStep` of the new episode.
        """
        # init sim data and put on device
        mjx_data = mjx.make_data(self.mjx_model)
        
        # TODO: replace with randomised starting positions
        mjx_data = mjx_data.replace(qpos=jnp.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785]))

        # step environment dynamics
        mjx_data = mjx.step(self.mjx_model, mjx_data)

        return mjx_data

    @partial(jax.jit, static_argnums=(0,))
    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0))
    def step(
        self, 
        mjx_data, 
        target_position, 
        target_quat, 
        target_velocity, 
        target_angular_velocity, 
        ):
        """
        Updates the environment according to the action and returns a `TimeStep`.
        """
        
        # set controls using operation space controller
        ctrl = compute_osc_control(
            target_position,
            target_quat, 
            target_velocity, 
            target_angular_velocity,
            mjx_data,
            self.mjx_model,
            self.nullspace_config,
            self.eef_site_id,
            self.eef_body_id,
            self.arm_joint_ids,
        )
        mjx_data = mjx_data.replace(ctrl=ctrl)

        # step environment dynamics
        mjx_data = mjx.step(self.mjx_model, mjx_data)

        return mjx_data

    def render_observation(self, data):
        camera_id = mj_name2id(self._physics.model.ptr, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera/front_camera")
        self.renderer.update_scene(data, camera_id)
        pixels = self.renderer.render()

        import matplotlib.pyplot as plt 
        plt.imshow(pixels)
        plt.show(block=True)

    def observation_spec(self) -> dm_env.specs.Array:
        """Returns the observation spec."""
        pass

    def action_spec(self) -> Dict[str, dm_env.specs.Array]:
        """Returns the action spec."""
        pass

    def _compute_observation(self) -> np.ndarray:
        """Returns the observation."""
        pass

    
if __name__=="__main__":
    # clear hydra global state to avoid conflicts with other hydra instances
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    # read hydra config
    initialize(version_base=None, config_path="../config", job_name="rearrangement")
    
    # add task configs
    COLOR_SEPARATING_CONFIG = compose(
            config_name="rearrangement",
            overrides=[
                "arena/props=colour_splitter",
                "arena/cameras=rearrangement",
                "simulation_tuning_mode=False",
                "robots=franka_robotiq_2f85_mjx",
                ]
                )

    # instantiate prng for jax
    key = jax.random.PRNGKey(0)

    # instantiate task environment
    env = RearrangementEnv(cfg=COLOR_SEPARATING_CONFIG) 
    qpos = jnp.zeros((1000, 7))
    mjx_data = env.reset(qpos)

    # while testing controller fix the target to reset position 
    target_position = mjx_data.xpos[:, env.eef_site_id, :] 
    target_quat = mjx_data.xquat[:, env.eef_site_id, :]
    target_velocity =  jnp.zeros((1000, 3))
    target_angular_velocity = jnp.zeros((1000, 3))

    iter = 0
    while True:
        
        mjx_data = env.step(
            mjx_data, 
            target_position, 
            target_quat, 
            target_velocity,
            target_angular_velocity,
            )

        iter+=1
        data = mjx.get_data(env._physics.model.ptr, mjx_data)
        if iter % 100 == 0:
            env.render_observation(data[0])
