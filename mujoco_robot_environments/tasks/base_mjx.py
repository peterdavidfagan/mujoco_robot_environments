"""Mujoco environment for interactive task learning."""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from abc import abstractmethod
from functools import partial
from typing import Optional, Dict
from copy import deepcopy
from pathlib import Path
import threading
import random
import matplotlib.pyplot as plt

import mujoco
from mujoco import viewer
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
    with initialize(config_path="../config", job_name="lasa"):
        cfg = compose(
            config_name="pick_mjx",
            overrides=[
                "simulation_tuning_mode=True",
                ]
                )
    return cfg

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

class BaseEnv(dm_env.Environment):
    """MuJoCo powered robotics environment with dm_env interface."""

    def __init__(
        self,
        cfg: DictConfig = generate_default_config(),
    ):
        """Initializes the simulation environment from config."""
        # ensure mjcf paths are relative to this file
        file_path = Path(__file__).parent.absolute()
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

        # add cube for pick and place task
        cube = Rectangle(
            name="cube",
            x_len=0.025,
            y_len=0.025,
            z_len=0.025,
            rgba=(1.0, 0.0, 0.0, 1.0),
            mass=0.1,
            friction=(1, 1, 1),
            solimp=(0.95, 0.995, 0.001, 0.5, 3),
            solref=(0.01, 1.1),
            margin = 0.15,
            gap = 0.15,
        )
        frame = self._arena.add_free_entity(cube)
        cube.set_freejoint(frame.freejoint)

        # add robot model with actuators and sensors
        self.arm = instantiate(cfg.robots.arm.arm)
        self.end_effector = instantiate(cfg.robots.end_effector.end_effector)
        standard_compose(arm=self.arm, gripper=self.end_effector)
        
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
        # self._arena.mjcf_model.option.integrator = 'IMPLICITFAST'
        # self._arena.mjcf_model.option.cone = 'PYRAMIDAL'
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)
        print(self._arena.mjcf_model.to_xml_string())

        cube.set_pose(self._physics, np.array([0.4, 0.0, 0.6]), np.array([1, 0, 0, 0]))

        # get arm joint ids and eef site id
        self.arm_joint_ids = []
        for joint in self.arm.joints:
            self.arm_joint_ids.append(mj_name2id(self._physics.model.ptr, 3, 'panda nohand/' + joint.name))
        self.arm_joint_ids = jnp.array(self.arm_joint_ids)
        self.eef_site_id = mj_name2id(self._physics.model.ptr, 6, 'panda nohand/attachment_site')
        self.eef_body_id = self._physics.model.ptr.site_bodyid[self.eef_site_id]

        # nullspace joint configuration for osc controller
        self.nullspace_config = jnp.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

        # put model on device
        self.mjx_model = mjx.put_model(self._physics.model.ptr) 

        if self._cfg.madrona.use:
            from madrona_mjx.renderer import BatchRenderer 
            self.renderer = BatchRenderer(
                m=self.mjx_model,
                gpu_id=0,
                num_worlds=3,
                batch_render_view_width=64,
                batch_render_view_height=64,
                enabled_geom_groups=np.asarray(
                    [0,1,2]
                ),
                enabled_cameras=np.asarray([
                    0,
                ]),
                add_cam_debug_geo=False,
                use_rasterizer=False,
                viz_gpu_hdls=None,
            )

    
    @partial(jax.jit, static_argnums=(0,))
    @partial(jax.vmap, in_axes=(None, 0))
    def reset(self, qpos) -> dm_env.TimeStep:
        """
        Resets the environment to an initial state and returns the first `TimeStep` of the new episode.
        """
        # init sim data and put on device
        mjx_data = mjx.make_data(self.mjx_model)
        
        # TODO: replace with randomised starting positions
        mjx_data = mjx_data.replace(qpos=jnp.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0, 0, 0, 0, 0, 0]))
        # mjx_data = mjx_data.replace(qvel=jnp.zeros((7,)))

        # step environment dynamics
        mjx_data = mjx.forward(self.mjx_model, mjx_data)

        if self._cfg.madrona.use:
            # initialise render token
            self.render_token, rgb, _ = self.renderer.init(mjx_data, self.mjx_model)
        
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
        # ctrl = compute_osc_control(
        #     target_position,
        #     target_quat, 
        #     target_velocity, 
        #     target_angular_velocity,
        #     mjx_data,
        #     self.mjx_model,
        #     self.nullspace_config,
        #     self.eef_site_id,
        #     self.eef_body_id,
        #     self.arm_joint_ids,
        # )
        # mjx_data = mjx_data.replace(ctrl=ctrl)

        # step environment dynamics
        mjx_data = mjx.step(self.mjx_model, mjx_data)

        if self._cfg.madrona.use:
            # try get data from renderer
            _, rgb, _ = self.renderer.render(self.render_token, mjx_data)

        return mjx_data

    # def render_observation(self, data):
    #     camera_id = mj_name2id(self._physics.model.ptr, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera/front_camera")
    #     self.renderer.update_scene(data, camera_id)
    #     pixels = self.renderer.render()

    #     import matplotlib.pyplot as plt 
    #     plt.imshow(pixels)
    #     plt.show(block=True)

    def observation_spec(self) -> dm_env.specs.Array:
        """Returns the observation spec."""
        pass

    def action_spec(self) -> Dict[str, dm_env.specs.Array]:
        """Returns the action spec."""
        pass

    def _compute_observation(self) -> np.ndarray:
        """Returns the observation."""
        pass

    def interactive_debug(self):
        """
        Interactively debug the environment.
        """
        passive_view = viewer.launch_passive(self._physics.model.ptr, self._physics.data.ptr)
        while True:
            self._physics.step()
            passive_view.sync()
        passive_view.close()

    
if __name__=="__main__":
    # clear hydra global state to avoid conflicts with other hydra instances
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    # read hydra config
    initialize(version_base=None, config_path="../config", job_name="lasa")

    # instantiate prng for jax
    key = jax.random.PRNGKey(0)

    # instantiate task environment
    env = BaseEnv() 
    env.interactive_debug()

    qpos = np.zeros((3, 7))
    mjx_data = env.reset(qpos)
    # print(mjx_data.qpos.shape)
    # print(rgb.shape)

    # plt.plt(rgb[..., :3])
    # plt.show()

    # while testing controller fix the target to reset position 
    # print(mjx_data.xpos[:, env.eef_site_id])
    # target_position = mjx_data.xpos[:, env.eef_site_id, :].copy()
    # target_quat = mjx_data.xquat[:, env.eef_site_id, :].copy()
    target_position = np.zeros((3, 3))
    target_quat = np.array([[0, 0, 0, 1]] * 3)
    target_velocity =  np.zeros((3, 3))
    target_angular_velocity = np.zeros((3, 3))

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
        mj_data = mjx.get_data(env._physics.model.ptr, mjx_data)
        print(mj_data)
        # if iter % 100 == 0:
        #     env.render_observation(data[0])
