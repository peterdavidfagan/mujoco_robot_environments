"""Simplified Robotiq Gripper Class."""

from typing import List, Tuple, Optional

from dm_control import mjcf
from mujoco_controllers.models.end_effectors.robot_hand import RobotHand
import numpy as np

class Robotiq2F85(RobotHand):
  """Robotiq 2-finger 85 adaptive gripper."""

  def __init__(self,
          mjcf_path: str,
          actuator_config: dict = None, # for now we don't alter default actuator config
          sensor_config: dict = None,
          controller_config: dict = None,
          ):
    """Initializes the Robotiq 2-finger 85 gripper."""
    self.mjcf_path = mjcf_path
    self.actuator_config = actuator_config
    self.sensor_config = sensor_config
    self.controller_config = controller_config
    super().__init__()


  def _build(self):
    """
    Initializes the Robotiq 2-finger 85 gripper, here we use defaults from mujoco menagerie without any changes.
    """
    self.robotiq_root = mjcf.from_path(self.mjcf_path)
    # overwrite actuator params
    overwrite = {
    #"tendon": "split",
    "forcerange": "-1.5 1.5",
    #"ctrlrange": "0 1",
    #"gainprm": "80 0 0",
    #"biasprm": "0 -100 -10"
    }
    self.robotiq_root.find("actuator", "fingers_actuator").set_attributes(**overwrite)
    self._joints = self.robotiq_root.find_all('joint') 
    self._actuators = self.robotiq_root.find_all('actuator')
    self._tool_center_point = self.robotiq_root.find('site', 'pinch')
    
    # consider adding tcp site

  @property
  def joints(self):
    """List of joint elements belonging to the hand."""
    return self._joints

  @property
  def actuators(self):
    """List of actuator elements belonging to the hand."""
    return self._actuators

  @property
  def mjcf_model(self) -> mjcf.RootElement:
    """Returns the `mjcf.RootElement` object corresponding to the robot hand."""
    return self.robotiq_root

  @property
  def name(self) -> str:
    """Name of the robot hand."""
    return "robotiq_2f85"

  @property
  def tool_center_point(self):
    """Tool center point site of the hand."""
    return self._tool_center_point

