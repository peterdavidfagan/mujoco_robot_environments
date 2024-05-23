"""Franka Emika Panda Robot Arm."""

import os
import numpy as np

from dm_control import mjcf
from robot_descriptions import panda_mj_description
from mujoco_controllers.models.arms.robot_arm import RobotArm

PANDA_MJCF_PATH = os.path.join(panda_mj_description.PACKAGE_PATH, 'panda_nohand.xml')

class FER(RobotArm):
    """Franka Emika Panda Robot Arm."""

    def __init__(
        self, 
        mjcf_path: str = PANDA_MJCF_PATH,
        actuator_config: dict = None, 
        sensor_config: dict = None, 
        controller_config: dict = None, 
        configuration_config: dict = None
        ):
        """Initialize the robot arm."""
        self.mjcf_path = mjcf_path
        self.actuator_config = actuator_config
        self.sensor_config = sensor_config
        self.controller_config = controller_config
        self._named_configurations = configuration_config
        self.controller = None
        super().__init__()

    def _build(self):
        self._fer_root = mjcf.from_path(self.mjcf_path, escape_separators=True)
        for keyframe in self._fer_root.find_all('key'):
            keyframe.remove()

        # assign joints
        self._joints = self._fer_root.find_all("joint")
        
        # add sensors and actuators
        self._add_actuators()
        self._actuators = self._fer_root.find_all("actuator")
        self._add_sensors()
        
        # define attachment and wrist sites
        self._attachment_site = self._fer_root.find("site", "attachment_site")
        self._wrist_site = self._attachment_site

    # TODO: Refactor this
    def _add_actuators(self):
        """Override the actuator model by config."""
        # remove default actuator models from mjcf
        for actuator in self._fer_root.find_all('actuator'):
            actuator.remove()

        if self.actuator_config["type"] == "motor":
            for idx, (joint, joint_type) in enumerate(self.actuator_config["joint_actuator_mapping"].items()):
                print("Adding actuator for joint: {}".format(joint))
                actuator = self._fer_root.actuator.add(
                    "motor",
                    name="actuator{}".format(idx + 1),
                    **self.actuator_config[joint_type],
                )
                actuator.joint = self._joints[idx]

        elif self.actuator_config["type"] == "general":
            for idx, (joint, joint_type) in enumerate(self.actuator_config["joint_actuator_mapping"].items()):
                print("Adding actuator for joint: {}".format(joint))
                actuator = self._fer_root.actuator.add(
                    "general",
                    name="actuator{}".format(idx + 1),
                    **self.actuator_config[joint_type],
                )
                actuator.joint = self._joints[idx]
        
        elif self.actuator_config["type"] == "velocity":
            for idx, (joint, joint_type) in enumerate(self.actuator_config["joint_actuator_mapping"].items()):
                print("Adding actuator for joint: {}".format(joint))
                actuator = self._fer_root.actuator.add(
                    "velocity",
                    name="actuator{}".format(idx + 1),
                    **self.actuator_config[joint_type],
                )
                actuator.joint = self._joints[idx]

        else:
            raise ValueError("Unsupported actuator model: {}".format(self.actuator_model))

    # TODO: Refactor this
    def _add_sensors(self):
        """Override the sensor model by config."""
        for sensor_suite in self.sensor_config:
            if sensor_suite["type"] == "jointpos":
                for idx, (joint, joint_type) in enumerate(sensor_suite["joint_sensor_mapping"].items()):
                    print("Adding sensor for joint: {}".format(joint))
                    sensor = self._fer_root.sensor.add(
                        "jointpos",
                        **sensor_suite[joint],
                    )
                    sensor.joint = self._joints[idx]
            elif sensor_suite["type"] == "jointtorque":
                for idx, (joint, joint_type) in enumerate(sensor_suite["joint_sensor_mapping"].items()):
                    print("Adding sensor for joint: {}".format(joint))
                    sensor = self._fer_root.sensor.add(
                        "jointpos",
                        **sensor_suite[joint],
                    )
                    sensor.joint = self._joints[idx]
            else:
                raise ValueError

    @property
    def joints(self):
        """Returns a list of joints in the robot."""
        return self._joints

    @property
    def actuators(self):
        """Returns a list of actuators in the robot."""
        return self._actuators

    @property
    def mjcf_model(self):
        """Returns the MJCF model for the robot."""
        return self._fer_root

    @property
    def name(self):
        """Returns the name of the robot."""
        return "franka_emika_panda"

    @property
    def wrist_site(self):
        """Returns the wrist site."""
        return self._wrist_site

    @property
    def attachment_site(self):
        """Returns the attachment site."""
        return self._attachment_site
    
    @property
    def named_configurations(self):
        """Returns the named configurations for the robot."""
        return self._named_configurations

    def set_joint_angles(self, physics: mjcf.Physics, qpos: np.ndarray) -> None:
        """Set the joint angles of the robot."""
        physics.bind(self._joints).qpos = qpos
