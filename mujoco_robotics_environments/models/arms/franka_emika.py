"""Franka Emika Panda Robot Arm."""

import numpy as np

from dm_control import mjcf
from mujoco_controllers.models.arms.robot_arm import RobotArm


class FER(RobotArm):
    """Franka Emika Panda Robot Arm."""

    def __init__(self, mjcf_path: str, actuator_config: dict, sensor_config: dict=None, controller_config: dict=None, configuration_config: dict=None):
        """Initialize the robot arm."""
        self.mjcf_path = mjcf_path
        self.actuator_config = actuator_config
        self.sensor_config = sensor_config
        self.controller_config = controller_config
        self._named_configurations = configuration_config
        self.controller = None
        super().__init__()

    def _build(self):
        self._fer_root = mjcf.from_path(self.mjcf_path)
        self._joints = self._fer_root.find_all("joint")
        # add sensors and actuators
        self._add_actuators()
        self._add_sensors()
        self._actuators = self._fer_root.find_all("actuator")
        # define attachment and wrist sites
        self._attachment_site = self._fer_root.find("site", "attachment_site")
        self._wrist_site = self._attachment_site

    # TODO: Refactor this
    def _add_actuators(self):
        """Override the actuator model by config."""
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
        if self.sensor_config["type"] == "jointpos":
            for idx, (joint, joint_type) in enumerate(self.sensor_config["joint_sensor_mapping"].items()):
                print("Adding sensor for joint: {}".format(joint))
                sensor = self._fer_root.sensor.add(
                    "jointpos",
                    **self.sensor_config[joint],
                )
                sensor.joint = self._joints[idx]

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
