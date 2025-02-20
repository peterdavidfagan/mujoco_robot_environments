"""Mujoco environment for interactive task learning."""
import os
import time
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
from scipy.spatial.transform import Rotation as R

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
        data.xpos[eef_body_id, :], # point
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
    eef_mass_matrix = jnp.linalg.pinv(mass_matrix_inv, rtol=1e-2) # TODO: consider switching to inv

    # get current end-effector state variables
    eef_position = data.xpos[eef_body_id] 
    eef_quat = data.xquat[eef_body_id]
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
    position_pd = (300.0 * position_error) + (30.0 * velocity_error)
    orientation_pd = (100.0 * orientation_error) + (100.0 * angular_velocity_error)
    pd_error = jnp.hstack([position_pd, orientation_pd])

    # compute control signal
    nullspace_position_error = nullspace_configuration - jnp.take_along_axis(data.qpos, arm_joint_ids + 1, axis=0)
    nullspace_velocity_error = jnp.zeros((7,)) - jnp.take_along_axis(data.qvel, arm_joint_ids, axis=0)
    nullspace_pd = (20.0 * nullspace_position_error) + (10.0 * nullspace_velocity_error)
    null_jacobian = jnp.linalg.inv(arm_mass_matrix) @ eef_jacobian.T @ eef_mass_matrix

    tau = eef_jacobian.T @ eef_mass_matrix @ pd_error # pd control against eef target
    tau += (jnp.eye(7) - eef_jacobian.T @ null_jacobian.T) @ nullspace_pd # nullspace projection
    tau += jnp.take_along_axis(data.qfrc_bias, arm_joint_ids, axis=0) # compensate for external forces

    # TODO: need to update acuator moment calculation in mjx 
    # compute effective torque through compensating for actuator moment 
    # actuator_moment_inv = jnp.linalg.pinv(data.actuator_moment)
    # actuator_moment_inv = actuator_moment_inv[arm_joint_ids, :][:, arm_joint_ids]
    # tau = tau @ actuator_moment_inv 
    # jax.debug.print("tau: {}", tau)

    tau = jnp.clip(tau, -87, 87)

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
        self._arena.mjcf_model.option.integrator="implicitfast"
        self._arena.mjcf_model.option.iterations = 10
        self._arena.mjcf_model.option.ls_iterations = 10 
        self._arena.mjcf_model.option.timestep = 0.005
        self._arena.mjcf_model.option.gravity = cfg.gravity
        self._arena.mjcf_model.size.nconmax = 10
        self._arena.mjcf_model.size.njmax = cfg.njmax
        self._arena.mjcf_model.visual.__getattr__("global").offheight = cfg.offheight
        self._arena.mjcf_model.visual.__getattr__("global").offwidth = cfg.offwidth
        self._arena.mjcf_model.visual.map.znear = cfg.znear

        # add table for manipulation task
        table = Rectangle(
        name="table",
        x_len=1.5,
        y_len=1.0,
        z_len=0.2,
        rgba=(0.5, 0.5, 0.5, 1.0),
        margin=0.0,
        gap=0.0,
        mass=10,
        friction=(1, 0.005, 0.0001),
        solimp=(0.95, 0.995, 0.001, 0.5, 2),
        solref=(0.02, 1.0),
        )
        table_attach_site = self._arena.mjcf_model.worldbody.add(
            "site",
            name="table_center",
            pos=(0.0, 0.0, 0.2),
        )
        self._arena.attach(table, table_attach_site)


        # add barrier for transport task (in future make sure this can be sampled)
        barrier_1 = Rectangle(
            name="barrier_1",
            x_len=0.3,
            y_len=0.05,
            z_len=0.25,
            rgba=(0.5, 0.5, 0.5, 1.0),
            mass=10.0,
            friction=(1, 0.005, 0.0001),
            solimp=(0.95, 0.995, 0.001, 0.5, 2),
            solref=(0.02, 1.0),
            margin = 0.0,
            gap = 0.0,
        )
        barrier_1_attach_site = self._arena.mjcf_model.worldbody.add(
            "site",
            name="barrier_1_center",
            pos=(0.15, 0.0, 0.45),
        )
        self._arena.attach(barrier_1, barrier_1_attach_site)

        barrier_2 = Rectangle(
            name="barrier_2",
            x_len=0.3,
            y_len=0.05,
            z_len=0.25,
            rgba=(0.5, 0.5, 0.5, 1.0),
            mass=10.0,
            friction=(1, 0.005, 0.0001),
            solimp=(0.95, 0.995, 0.001, 0.5, 2),
            solref=(0.02, 1.0),
            margin = 0.0,
            gap = 0.0,
        )
        barrier_2_attach_site = self._arena.mjcf_model.worldbody.add(
            "site",
            name="barrier_2_center",
            pos=(0.15, 0.0, 1.05),
        )
        self._arena.attach(barrier_2, barrier_2_attach_site)

        # add beam for shared transport task (in future make inertia parameters possibly sampled)
        beam = Rectangle(
            name="cube",
            x_len=0.5,
            y_len=0.025,
            z_len=0.025,
            rgba=(1.0, 0.0, 0.0, 1.0),
            mass=2.0,
            friction=(1, 0.005, 0.0001),
            solimp=(0.95, 0.995, 0.001, 0.5, 2),
            solref=(0.02, 1.0),
            margin = 0.0,
            gap = 0.0,
        )
        frame = self._arena.add_free_entity(beam)
        beam.set_freejoint(frame.freejoint)
        beam_body = beam.mjcf_model.find("body", "prop_root")
        beam_body.add(
            'site',
            name='beam_left_site',
            pos=[-0.5, 0, 0],
        )


        # define a mocap for beam control target
        self.beam_target_mocap=self._arena.mjcf_model.worldbody.add(
                    'body',
                    name="beam_target_mocap",
                    mocap="true",
                    pos=[0.75, -0.35, 0.5],
                    quat=R.from_euler('xyz', [0, 0, 0], degrees=True).as_quat(),
                )
            
        self.beam_target_mocap.add(
            'geom',
            name='beam_target_viz',
            type='box',
            size=[0.025, 0.025, 0.025],  
            rgba=[1.0, 0.0, 0.0, 0.25],  
            pos=[0.0, 0.0, 0.0],
            contype=0,  # no collision with any object
            conaffinity=0  # no influence on collision detection
        )

        # define a mocap for eef control target
        self.eef_target_mocap=self._arena.mjcf_model.worldbody.add(
                'body',
                name="eef_target_mocap",
                mocap="true",
                pos=[-0.25, -0.35, 0.5],
                quat=R.from_euler('xyz', [180, 180, 0], degrees=True).as_quat(),
            )
            
        self.eef_target_mocap.add(
            'geom',
            name='eef_target_viz',
            type='box',
            size=[0.025, 0.025, 0.025],  
            rgba=[1.0, 0.0, 0.0, 0.25],  
            pos=[0.0, 0.0, 0.0],
            contype=0,  # no collision with any object
            conaffinity=0  # no influence on collision detection
            )

        # define a mocap for beam goal position
        self.beam_goal_mocap=self._arena.mjcf_model.worldbody.add(
                    'body',
                    name="beam_goal_mocap",
                    mocap="true",
                    pos=[0.25, 0.35, 0.5],
                    quat=R.from_euler('xyz', [0, 0, 0], degrees=True).as_quat(),
                )
        
        self.beam_goal_mocap.add(
            'geom',
            name='beam_goal_viz',
            type='box',
            size=[0.5, 0.025, 0.025],
            rgba=[0.0, 1.0, 0.0, 0.25],
            pos=[0.0, 0.0, 0.0],
            contype=0,  # no collision with any object
            conaffinity=0  # no influence on collision detection
            )

        # add robot model with actuators and sensors
        self.arm = instantiate(cfg.robots.arm.arm)

        robot_base_site = self._arena.mjcf_model.worldbody.add(
            "site",
            name="robot_base",
            pos=(-0.7, 0.0, 0.4),
        )
        self._arena.attach(self.arm, robot_base_site) 

        # now try to add a weld constraint between the beam and the robot
        # equality = self._arena.mjcf_model.find_all('equality')
        # equality = self._arena.mjcf_model.equality
        # equality.add(
        #     "weld",
        #     name="robot_beam_weld",
        #     body1="cube/prop_root",
        #     body2="panda/attachment",
        #     relpose=(-0.5, 0.0, -0.05, 0.0, 0.0, 0.0, 1.0),
        # )

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
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # get details of the joint for shared transport
        self.beam_joint_id = mj_name2id(self._physics.model.ptr, mujoco.mjtObj.mjOBJ_JOINT, 'cube/')
        self.beam_body_id = mj_name2id(self._physics.model.ptr, mujoco.mjtObj.mjOBJ_BODY, 'cube/')
        self.beam_geom_id = mj_name2id(self._physics.model.ptr, mujoco.mjtObj.mjOBJ_GEOM, 'cube/cube')
        self.beam_site_id = mj_name2id(self._physics.model.ptr, mujoco.mjtObj.mjOBJ_SITE, 'cube/box_centre')
        beam.set_pose(self._physics, np.array([0.25, -0.35, 0.5]), np.array([1, 0, 0, 0]))

        # get arm joint ids and eef site id
        self.arm_joint_ids = []
        for joint in self.arm.joints:
            self.arm_joint_ids.append(mj_name2id(self._physics.model.ptr, 3, 'panda/' + joint.name))
        self.arm_joint_ids = jnp.array(self.arm_joint_ids) + 5 # add 5 for now to account for free joint
        self.eef_site_id = mj_name2id(self._physics.model.ptr, 6, 'panda/attachment_site')
        self.eef_body_id = self._physics.model.ptr.site_bodyid[self.eef_site_id]

        # print all actuators parameters
        # print(self.arm.mjcf_model.to_xml_string())

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
    def reset(self, qpos=None) -> dm_env.TimeStep:
        """
        Resets the environment to an initial state and returns the first `TimeStep` of the new episode.
        """
        # details for resetting the environment state (for parallisation probably don't want sampling here)
        # joint_angles = [-2.9, -0.8, 2.27, -2.09, 0.911, 2.42, -0.449]
        # self.arm.set_joint_angles(self._physics, joint_angles)
        
        # init sim data and put on device
        # mjx_data = mjx.make_data(self.mjx_model)
        mjx_data = mjx.put_data(self._physics.model.ptr, self._physics.data.ptr)

        # TODO: replace with randomised starting positions
        if qpos is not None:
            mjx_data = mjx_data.replace(qpos=qpos)
        else:
            mjx_data = mjx_data.replace(qpos=jnp.asarray([0.25, -0.35, 0.5, 0, 0, 0, -1, -2.9, -0.8, 2.27, -2.09, 0.911, 2.42, -0.449]))
        
        # mjx_data = mjx_data.replace(qvel=jnp.zeros((7,)))

        # step environment dynamics
        mjx_data = mjx.forward(self.mjx_model, mjx_data)
        # mjx_data = mjx.put_data(self.mjx_model, mjx_data)

        if self._cfg.madrona.use:
            # initialise render token
            self.render_token, rgb, _ = self.renderer.init(mjx_data, self.mjx_model)
        
        return mjx_data

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, 
        mjx_data, 
        ctrl=jnp.asarray([87, 87, 87, 87, 87, 87, 87]),
        ):
        """
        Updates the environment according to the action and returns a `TimeStep`.
        """

        # apply user inputed control                
        mjx_data = mjx_data.replace(ctrl=ctrl)

        # apply external force to the beam 

        # step environment dynamics
        mjx_data = mjx.step(self.mjx_model, mjx_data)

        if self._cfg.madrona.use:
            # try get data from renderer
            _, rgb, _ = self.renderer.render(self.render_token, mjx_data)

        return mjx_data

    def observation_spec(self) -> dm_env.specs.Array:
        """Returns the observation spec."""
        pass

    def action_spec(self) -> Dict[str, dm_env.specs.Array]:
        """Returns the action spec."""
        pass

    def _compute_observation(self) -> np.ndarray:
        """Returns the observation."""
        pass

    def debug_mjx(self):
        """
        Visualize MJX simulated environment.
        """
        view = viewer.launch_passive(self._physics.model.ptr, self._physics.data.ptr)
        target_position = jnp.asarray([0.25, 0.0, 0.8])
        target_quat = R.from_euler('xyz', [180, 180, 0], degrees=True).as_quat()
        target_velocity = jnp.zeros(3)
        target_angular_velocity = jnp.zeros(3)

        # test vmapped reset
        input_qpos = jnp.repeat(jnp.array([0.25, -0.35, 0.5, 0, 0, 0, -1, -2.9, -0.8, 2.27, -2.09, 0.911, 2.42, -0.449])[None, :], 50, axis=0)
        input_qpos += jax.random.uniform(key, (50, 14), minval=-0.1, maxval=0.1)
        mjx_data = jax.vmap(self.reset)(input_qpos)
        # mjx_data = self.reset()
        print(mjx_data.qpos)

        while True:
            # mjx_data = env.step(
            #     mjx_data, 
            # )
            mjx_data = jax.vmap(self.step)(mjx_data)
            print(mjx_data.qpos)

            # mjx.get_data_into(self._physics.data.ptr, self._physics.model.ptr, mjx_data)
            # view.sync()


    def interactive_debug(self):
        """
        Interactively debug the environment with a passive viewer.
        """
        passive_view = viewer.launch_passive(self._physics.model.ptr, self._physics.data.ptr)
        joint_angles = [-2.9, -0.8, 2.27, -2.09, 0.911, 2.42, -0.449]
        # self.arm.set_joint_angles(self._physics, self.arm.named_configurations["home"])
        self.arm.set_joint_angles(self._physics, joint_angles)
        # create an instance of a robot interface for robot and controllers
        self._robot = RobotArm(
                arm=self.arm, 
                gripper=None, 
                physics=self._physics,
                passive_viewer=passive_view,
                )

        while True:
            
            ### apply osc control to the robot arm ###
            target_position = np.array([0.6, 0.0, 0.7])
            target_quat = np.array([0, 0, 0, 1])
            
            self._robot.arm_controller.set_target(
                position=target_position,
                quat=target_quat, 
                velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            )

            # get difference between eef site and mocap body
            mocap_pos = self._physics.data.mocap_pos[1]
            mocap_quat = self._physics.data.mocap_quat[1]

            # update control target
            self._robot.arm_controller.set_target(
                position=mocap_pos + [0.0, 0.0, 0.05],
                quat=mocap_quat, 
                velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
                )

            arm_command = self._robot.arm_controller.compute_control_output()
            self._physics.data.ctrl[:] = arm_command

            ### prototype of the agent carrying the beam ###

            # get target position from mocap 
            target_position = self._physics.data.mocap_pos[0]

            # Get COM position of the beam in world coordinates
            com_position = self._physics.data.xpos[self.beam_body_id]  # COM position in world frame

            # Get the rotation matrix of the beam
            rotation_matrix = self._physics.data.xmat[self.beam_body_id].reshape(3, 3)  # (3x3 rotation matrix)

            # Define local offset to the endpoint (in body frame)
            beam_half_length = self._physics.model.geom_size[self.beam_geom_id][0]  # X-size is half-length

            # Correct local offset (instead of assuming 0.25)
            r_local = np.array([beam_half_length, 0.0, 0.0])  
            r_global = rotation_matrix @ r_local

            # Compute the endpoint position in world coordinates
            endpoint_position = com_position + r_global

            # Compute position error for PD control
            position_error = target_position - endpoint_position

            # Get velocity of the beam's COM and angular velocity
            v_com = self._physics.data.qvel[self.beam_joint_id * 6 : self.beam_joint_id * 6 + 3]  # (vx, vy, vz)
            omega = self._physics.data.qvel[self.beam_joint_id * 6 + 3 : self.beam_joint_id * 6 + 6]  # (wx, wy, wz)

            # Compute velocity of the endpoint using rigid body motion
            v_endpoint = v_com + np.cross(omega, rotation_matrix @ r_local)  
            velocity_error = v_endpoint  # We want to slow movement

            # **PD Controller Gains**
            kp_pos = np.array([50.0, 50.0, 500.0])  # Position gains (x, y, z)
            kd_pos = np.array([0.0, 0.0, 0.0])  # Velocity damping (x, y, z)

            # Compute control force at the endpoint
            force = (kp_pos * position_error) - (kd_pos * velocity_error)

            # **Compute Correct Torque Compensation (\(\tau = r \times F\))**
            r_global = rotation_matrix @ r_local  # Offset in world frame
            torque = np.cross(r_global, force)

            # **Clamp force and torque values to prevent instability**
            max_force = 20
            max_torque = 10
            force = np.clip(force, -max_force, max_force)
            torque = np.clip(torque, -max_torque, max_torque)

            # **Apply the corrected force and torque**
            self._physics.data.xfrc_applied[self.beam_body_id, :3] = force  # Apply force at COM
            self._physics.data.xfrc_applied[self.beam_body_id, 3:] = torque  # Apply torque to correct for offset

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

    env.debug_mjx()
    # env.interactive_debug()

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
            )
        
        iter+=1
        mj_data = mjx.get_data(env._physics.model.ptr, mjx_data)
        # print(mj_data)
        # if iter % 100 == 0:
        #     env.render_observation(data[0])
