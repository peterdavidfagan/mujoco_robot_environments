"""DEPRECATED: Build a MuJoCo scene for robot manipulation tasks."""

from typing import Tuple

# physics
from dm_control import composer, mjcf

# custom props
from mujoco_robot_environments.models.arenas import empty
from mujoco_robot_environments.environment.props import add_objects, Rectangle
from mujoco_robot_environments.environment.cameras import add_camera

# config
import hydra
from hydra.utils import instantiate
from hydra import compose, initialize
from omegaconf import DictConfig


def build_arena(name: str) -> composer.Arena:
    """Build a MuJoCo arena."""
    arena = empty.Arena(name=name)
    arena.mjcf_model.option.timestep = 0.001
    arena.mjcf_model.option.gravity = (0.0, 0.0, -9.8)
    arena.mjcf_model.size.nconmax = 1000
    arena.mjcf_model.size.njmax = 2000
    arena.mjcf_model.visual.__getattr__("global").offheight = 640
    arena.mjcf_model.visual.__getattr__("global").offwidth = 640
    arena.mjcf_model.visual.map.znear = 0.0005
    return arena


def add_basic_table(arena: composer.Arena) -> Rectangle:
    """Add a basic table to the arena."""
    table = Rectangle(
        name="table",
        x_len=0.8,
        y_len=1.0,
        z_len=0.2,
        rgba=(0.5, 0.5, 0.5, 1.0),
        margin=0.0,
        gap=0.0,
    )

    # define attachment site
    attach_site = arena.mjcf_model.worldbody.add(
        "site",
        name="table_center",
        pos=(0.4, 0.0, 0.2),
    )

    arena.attach(table, attach_site)

    return table


def add_robot_and_gripper(arena: composer.Arena, arm, gripper) -> Tuple[composer.Entity, composer.Entity]:
    """Add a robot and gripper to the arena."""
    # attach the gripper to the robot
    robot.standard_compose(arm=arm, gripper=gripper)

    # define robot base site
    robot_base_site = arena.mjcf_model.worldbody.add(
        "site",
        name="robot_base",
        pos=(0.0, 0.0, 0.4),
    )

    # add the robot and gripper to the arena
    arena.attach(arm, robot_base_site)

    return arm, gripper


@hydra.main(version_base=None, config_path="../config", config_name="scene")
def construct_base_scene(cfg: DictConfig) -> None:
    """Build a base scene for robot manipulation tasks."""
    # build the base arena
    arena = build_arena("base_scene")

    # add a basic table to the arena
    add_basic_table(arena)

    # add robot arm and gripper to the arena
    arm = instantiate(cfg.robots.arm)
    gripper = instantiate(cfg.robots.gripper)
    arm, gripper = add_robot_and_gripper(arena, arm, gripper)

    # add props to the arena
    props = add_objects(
        arena,
        shapes=cfg.props.shapes,
        colours=cfg.props.colours,
        min_object_size=cfg.props.min_object_size,
        max_object_size=cfg.props.max_object_size,
        min_objects=cfg.props.min_objects,
        max_objects=cfg.props.max_objects,
        sample_size=cfg.props.sample_size,
        sample_colour=cfg.props.sample_colour,
    )

    # add cameras to the arena
    for camera in cfg.cameras:
        camera = add_camera(
            arena,
            name=camera.name,
            pos=camera.pos,
            quat=camera.quat,
            height=camera.height,
            width=camera.width,
            fovy=camera.fovy,
        )

    # build the physics
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    
    return {
        "arena": arena,
        "physics": physics,
        "arm": arm,
        "gripper": gripper,
        "props": props,
    }


if __name__ == "__main__":
    # clear hydra global state to avoid conflicts with other hydra instances
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    # generate config
    initialize(version_base=None, config_path="../config", job_name="default_config")
    DEFAULT_CONFIG = compose(config_name="transporter_data_collection")
    
    scene_components = construct_base_scene(DEFAULT_CONFIG)
