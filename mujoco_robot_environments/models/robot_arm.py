"""A module for encapsulating the robot arm and gripper"""

import abc
from typing import List, Optional, Sequence, Generic, TypeVar

import mujoco
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.initializers import utils
from dm_control.mjcf import traversal_utils
import numpy as np

from mujoco_controllers.osc import OSC

import hydra
from hydra.utils import instantiate
from hydra import compose, initialize
from omegaconf import DictConfig

class RobotArm(abc.ABC):
  """Abstract base class for MOMA robotic arms and their attachments."""
   
  def __init__(
        self,
        arm,
        physics: mjcf.Physics,
        gripper = None,
        passive_viewer: Optional = None,
        ):
    """Initializes the robot arm and gripper."""
    # core simulation instance
    self.physics = physics
    self.passive_view = passive_viewer
    
    # set arm and controller
    self.arm = arm
    self.arm_controller = self.arm.controller_config.controller(self.physics, self.arm)
    self.eef_site = arm.attachment_site
    self.arm_joints = arm.joints
    self.arm_joint_ids = np.array(physics.bind(self.arm_joints).dofadr)
    
    # set gripper and controller
    if gripper is not None:
      self.end_effector = gripper
      self.end_effector_controller = self.end_effector.controller_config.controller # very simple controller no need for models or physics
      self.end_effector_joints = self.end_effector.joints
      self.end_effector_joint_ids = np.array(physics.bind(self.end_effector_joints).dofadr)
    else:
      self.end_effector = None  
    
    # set control timestep
    self.control_steps = int(self.arm.controller_config.controller_params.control_dt // self.physics.model.opt.timestep)

  @property
  def eef_pose(self):
    site_id = mujoco.mj_name2id(self.physics.model.ptr, mujoco.mjtObj.mjOBJ_SITE, 'panda nohand/robotiq_2f85/pinch')
    return self.physics.data.site_xpos[site_id]


  def run_controller(self, duration):
    """Runs the controller for a specified duration."""
    # set controller convergence status
    arm_converged = False
    gripper_converged = False if self.end_effector is not None else True
    converged = False

    start_time = self.physics.data.time
    while (self.physics.data.time - start_time < duration) and (not converged):
        # compute control command
        control_command = self.arm_controller.compute_control_output()
        if self.end_effector is not None:
          gripper_command = np.array([self.end_effector_controller.compute_control_output()])
          control_command = np.concatenate((control_command, gripper_command))

        # step the simulation
        for _ in range(self.control_steps):
            self.physics.set_control(control_command)
            self.physics.step()
            if self.passive_view is not None:
                self.passive_view.sync()

        if self.arm_controller.is_converged():
            arm_converged = True

        if arm_converged and gripper_converged:
            converged = True
            break
        
    # for now just run the gripper controller for the full duration
    if arm_converged:
        converged = True

    return converged


def standard_compose(
    arm,
    gripper,
    wrist_ft = None,
    wrist_cameras = ()
) -> None:
  """Creates arm and attaches gripper."""

  if wrist_ft:
    wrist_ft.attach(gripper)
    arm.attach(wrist_ft)
  else:
    arm.attach(gripper)

  for cam in wrist_cameras:
    arm.attach(cam, arm.wrist_site)
