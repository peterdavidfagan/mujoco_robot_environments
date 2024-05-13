"""Mujoco environment for interactive task learning."""
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


def generate_default_config():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="../config", job_name="rearrangement")
    return compose(
            config_name="rearrangement",
            overrides=[
                "arena/props=colour_splitter",
                "simulation_tuning_mode=False"
                ]
                          )


class RearrangementEnv(dm_env.Environment):
    """MuJoCo powered robotics environment with dm_env interface."""

    def __init__(
        self,
        viewer: Optional = None,
        cfg: DictConfig = generate_default_config(),
    ):
        """Initializes the simulation environment from config."""
        # ensure mjcf paths are relative to this file
        file_path = Path(__file__).parent.absolute()
        cfg.robots.arm.arm.mjcf_path = str(file_path / cfg.robots.arm.arm.mjcf_path)
        cfg.robots.end_effector.end_effector.mjcf_path = str(file_path / cfg.robots.end_effector.end_effector.mjcf_path)
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

        # add visuals for target locations
        for key, value in cfg.task.target_locations.items():
            self._arena.mjcf_model.worldbody.add(
                'geom',
                name=key,
                type='box',
                size=value["size"],  
                rgba=value["rgba"],  
                pos=value["location"],
                contype=0,  # no collision with any object
                conaffinity=0  # no influence on collision detection
            )

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


        # if debugging the task environment add mocap for controller eef
        if cfg.simulation_tuning_mode:
            self.eef_target_mocap=self._arena.mjcf_model.worldbody.add(
                    'body',
                    name="eef_target_mocap",
                    mocap="true",
                    pos=[0.4, 0.0, 0.6],
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
        
        # add props
        self.props = add_objects(
            self._arena,
            shapes=cfg.arena.props.shapes,
            colours=cfg.arena.props.colours,
            textures=cfg.arena.props.textures,
            min_object_size=cfg.arena.props.min_object_size,
            max_object_size=cfg.arena.props.max_object_size,
            min_objects=cfg.arena.props.min_objects,
            max_objects=cfg.arena.props.max_objects,
            sample_size=cfg.arena.props.sample_size,
            sample_colour=cfg.arena.props.sample_colour,
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
            if camera.name == "overhead_camera":
                self.overhead_camera_height = camera.height
                self.overhead_camera_width = camera.width
        
        # compile environment
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)
        self.renderer = mujoco.Renderer(self._physics.model.ptr, height=self.overhead_camera_height, width=self.overhead_camera_width)
        self.seg_renderer = mujoco.Renderer(self._physics.model.ptr, height=self.overhead_camera_height, width=self.overhead_camera_width)
        self.seg_renderer.enable_segmentation_rendering()
        self.depth_renderer = mujoco.Renderer(self._physics.model.ptr, height=self.overhead_camera_height, width=self.overhead_camera_width)
        self.depth_renderer.enable_depth_rendering()
        self.passive_view = None
        
        # instantiate initializers
        prop_position = distributions.Uniform(
            self._cfg.task.initializers.workspace.min_pose,
            self._cfg.task.initializers.workspace.max_pose,
        )
        
        prop_quaternion = rotations.QuaternionFromAxisAngle(
            axis=[0, 0, 1], 
            angle=(np.pi) * distributions.Uniform(0, 1)[0]
        )
        
        self.prop_initializer = PropPlacer(
            props=self.props,
            position=prop_position,
            quaternion=prop_quaternion,
            ignore_collisions=False,
            settle_physics=True,
        )
        self.prop_random_state = np.random.RandomState(seed=self._cfg.task.initializers.seed)
        self.prop_place_random_state = np.random.RandomState(seed=self._cfg.task.initializers.seed+1)

        self.mode = None
        
    def close(self) -> None:
        if self.passive_view is not None:
            self.passive_view.close()

    def time_limit_exceeded(self) -> bool:
        return self._physics.data.time >= cfg.time_limit

    @property
    def model(self) -> mujoco.MjModel:
        return self.physics.model

    @property
    def data(self) -> mujoco.MjData:
        return self.physics.data
    
    @property
    def props_info(self) -> dict:
       """
       Gets domain model.
       
       The domain model is a dictionary of objects and their properties.
       """
       shapes = self._cfg.arena.props.shapes

       # get prop object names
       prop_names = [
           self._physics.model.id2name(i, "geom")
           for i in range(self._physics.model.ngeom)
           if any(keyword in self._physics.model.id2name(i, "geom") for keyword in shapes)
       ]
       prop_ids = [self._physics.model.name2id(name, "geom") for name in prop_names]

       # get object information
       prop_positions = self._physics.named.data.geom_xpos[prop_names]
       prop_orientations = self._physics.named.data.geom_xmat[prop_names]
       prop_orientations = [R.from_matrix(mat.reshape((3, 3))).as_quat() for mat in prop_orientations]
       prop_rgba = self._physics.named.model.geom_rgba[prop_names]
       prop_names = [name.split("/")[0] for name in prop_names]

       # get object bounding box information
       def get_bbox(prop_id, segmentation_map):
           """Get the bounding box of an object (PASCAL VOC)."""
           prop_coords = np.argwhere(segmentation_map[:, :, 0] == prop_id)
           # Sometimes this fails, not sure why adding try/except for now and review later
           # the error relates to prop_coords being empty which suggests object is outside of camera view
           try:
               bbox_corners = np.array(
                [
                   np.min(prop_coords[:, 0]),
                   np.min(prop_coords[:, 1]),
                   np.max(prop_coords[:, 0]),
                   np.max(prop_coords[:, 1]),
                ]
                )
           except Exception as e:
                print(e)
                print("prop_id: ", prop_id)

           return bbox_corners

       # TODO: consider vectorizing this
       camera_id = mj_name2id(self._physics.model.ptr, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_camera/overhead_camera")
       self.seg_renderer.update_scene(self._physics.data.ptr, camera_id)
       segmentation_map = self.seg_renderer.render()
       
       prop_bbox = []
       for idx in prop_ids:
           bbox = get_bbox(idx, segmentation_map)
           prop_bbox.append(bbox)

       # extacting entity id and symbols from prop names
       entities = [prop_name.split("_")[-1] for prop_name in prop_names]
       symbols = [prop_name.split("_")[:-1] for prop_name in prop_names]

       # create a dictionary with all the data
       props_info = {
            entity: {
               "prop_name": prop_names[i], 
               "position": prop_positions[i],
               "orientation": prop_orientations[i],
               "rgba": prop_rgba[i],
               "bbox": prop_bbox[i],
               "symbols": symbols[i],
           }
           for i, entity in enumerate(entities)  
       }

       return props_info
    
    def reset(self) -> dm_env.TimeStep:
        """Resets the environment to an initial state and returns the first
        `TimeStep` of the new episode.
        """
        # reset the lation instance
        self._physics.reset()
        
        # reset arm to home position
        # Note: for other envs we may want random sampling of initial arm positions
        self.arm.set_joint_angles(self._physics, self.arm.named_configurations["home"])

        # sample new object positions with initializers
        self.prop_initializer(self._physics, self.prop_random_state)

        # configure viewer
        if self.has_viewer:
            if self.passive_view is not None:
                self.passive_view.close()
            self.passive_view = viewer.launch_passive(self._physics.model.ptr, self._physics.data.ptr)

        # create an instance of a robot interface for robot and controllers
        self._robot = RobotArm(
                arm=self.arm, 
                gripper=self.end_effector, 
                physics=self._physics,
                passive_viewer=self.passive_view,
                )
        
        # set the initial eef pose to home
        self.eef_home_pose = self._robot.eef_pose.copy()
        self.eef_home_pose[0] -= 0.1 # move up 10cm back so it is out of view of camera

        # start in pick mode 
        self.mode="pick"
        
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
        if self.mode == "pick":
            self.pick(action_dict['pose'])
            self.mode="place"
        else:
            self.place(action_dict['pose'])
            self.mode="pick"

        return dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=0.0,
                discount=0.0,
                observation=observation,
            )

    def pick(self, pose):
        """
        Scripted pick behaviour.
        """
        pose[2] = 0.575 # hardcode for now :(
        pre_pick = pose.copy()
        pre_pick[2] = 0.9
        self._robot.arm_controller.set_target(
            position=pre_pick[:3],
            velocity=np.zeros(3),
            quat=pre_pick[3:],
            angular_velocity=np.zeros(3),
            )
        if not self._robot.run_controller(2.0):
            raise RuntimeError("Failed to move arm to pre pick position")

        # move arm to pick position
        self._robot.arm_controller.set_target(position=pose[:3])
        if not self._robot.run_controller(2.0):
            raise RuntimeError("Failed to move arm to pick position")

        # close gripper
        self._robot.end_effector_controller.status = "max"
        if not self._robot.run_controller(1.0):
            raise RuntimeError("Failed to close gripper")

        
        # move arm to pre grasp position
        self._robot.arm_controller.set_target(position=pre_pick[:3])
        if not self._robot.run_controller(2.0):
            raise RuntimeError("Failed to move arm to pre grasp position")

        # move arm to home position
        quat = np.zeros(4,)
        rot_mat = R.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix().flatten()
        mujoco.mju_mat2Quat(quat, rot_mat)
        self._robot.arm_controller.set_target(
                position=self.eef_home_pose,
                quat=quat,
                )
        if not self._robot.run_controller(2.0):
            raise RuntimeError("Failed to move arm to home position")

    def place(self, pose):
        """
        Scripted place behaviour. 
        """
        pose[2] = 0.575 # hardcode for now :(
        # move arm to pre place position
        pre_place = pose.copy()
        pre_place[2] = 0.9
        self._robot.arm_controller.set_target(
            position=pre_place[:3],
            quat=pre_place[3:],
            )
        if not self._robot.run_controller(2.0):
            raise RuntimeError("Failed to move arm to pre place position")

        # move arm to place position
        self._robot.arm_controller.set_target(position=pose[:3])
        if not self._robot.run_controller(2.0):
            raise RuntimeError("Failed to move arm to place position")

        # open gripper
        self._robot.end_effector_controller.status = "min"
        if not self._robot.run_controller(1.0):
            raise RuntimeError("Failed to open gripper")
        
        # move arm to pre place position
        self._robot.arm_controller.set_target(position=pre_place[:3])
        if not self._robot.run_controller(2.0):
            raise RuntimeError("Failed to move arm to pre place position")

        # move arm to home position
        quat = np.zeros(4,)
        rot_mat = R.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix().flatten()
        mujoco.mju_mat2Quat(quat, rot_mat)
        self._robot.arm_controller.set_target(
                position=self.eef_home_pose,
                quat=quat,
                )
        if not self._robot.run_controller(2.0):
            raise RuntimeError("Failed to move arm to home position")

    def observation_spec(self) -> dm_env.specs.Array:
        """Returns the observation spec."""
        # get shape of overhead camera
        camera = self._arena.mjcf_model.find("camera", "overhead_camera/overhead_camera")
        camera_shape = self.overhead_camera_height, self.overhead_camera_width, 3
        return {
                "overhead_camera/depth": dm_env.specs.Array(shape=camera_shape[:-1], dtype=np.float32),
                "overhead_camera/rgb": dm_env.specs.Array(shape=camera_shape, dtype=np.float32),
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
        # get overhead camera
        camera_id = mj_name2id(self._physics.model.ptr, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_camera/overhead_camera")
        self.renderer.update_scene(self._physics.data.ptr, camera_id)
        self.depth_renderer.update_scene(self._physics.data.ptr, camera_id)
        
        # get rgb data
        rgb = self.renderer.render()

        # get depth data
        depth = self.depth_renderer.render()
        
        # add to observation
        obs = {}
        obs["overhead_camera/rgb"] = rgb
        obs["overhead_camera/depth"] = depth

        return obs
    
    def _get_camera_intrinsics(self, camera_name, h, w, inverse = False):
        """Returns the transform from pixels to camera coordinates."""
        camera_type_idx = mujoco.mjtObj.mjOBJ_CAMERA
        camera_id = mj_name2id(self._physics.model.ptr, camera_type_idx, camera_name)
        fov = self._physics.model.cam_fovy[camera_id]
        f = (1. / np.tan(np.deg2rad(fov)/2)) * self.overhead_camera_height/2.0
        
        return np.array([
            [-f, 0, (w - 1) / 2],
            [0, f, (h-1) / 2],
            [0, 0, 1]
            ])

    def _get_camera_extrinsics(self, camera_name):
        """Returns the transform from world coordinates to camera coordinates."""
        camera_type_idx = mujoco.mjtObj.mjOBJ_CAMERA
        camera_id = mj_name2id(self._physics.model.ptr, camera_type_idx, camera_name)
        pos = self._physics.data.cam_xpos[camera_id]
        rot_mat = self._physics.data.cam_xmat[camera_id].reshape(3, 3)
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rot_mat.T
        extrinsics[:3, 3] = -rot_mat.T @ pos

        return extrinsics

    def pixel_2_world(self, camera_name, coords):
        """Returns the world coordinates for a given pixel."""

        # get camera parameters
        # Note: in mujoco viewport aligns with -z
        # this was confusing on first pass over both APIs
        coords_rounded = np.round(np.copy(coords)).astype(np.int32)
        width, height = self.overhead_camera_width, self.overhead_camera_height
        intrinsics = self._get_camera_intrinsics(camera_name, height, width)
        extrinsics = self._get_camera_extrinsics(camera_name)

        # render depth value for projection purposes 
        camera_type_idx = mujoco.mjtObj.mjOBJ_CAMERA
        camera_id = mj_name2id(self._physics.model.ptr, camera_type_idx, camera_name)
        self.depth_renderer.update_scene(self._physics.data.ptr, camera_id)
        depth_vals = self.depth_renderer.render()
        depth_val = depth_vals[coords_rounded[1], coords_rounded[0]]
        
        # convert pixels to camera frame coordinates
        image_coords = np.concatenate([coords, np.ones(1)])
        camera_coords =  np.linalg.inv(intrinsics) @ image_coords
        camera_coords *= -depth_val # negative sign due to mujoco camera convention

        # convert camera coordinates to world coordinates
        camera_coords = np.concatenate([camera_coords, np.ones(1)])
        world_coords = np.linalg.inv(extrinsics) @ camera_coords
        world_coords = world_coords[:3] / world_coords[3]
        
        return world_coords

    def world_2_pixel(self, camera_name, coords):
        """Returns the pixel coordinates for a given world coordinate."""
        intrinsics = self._get_camera_intrinsics(camera_name, self.overhead_camera_height, self.overhead_camera_width)
        extrinsics = self._get_camera_extrinsics(camera_name)
        
        # convert world coordinates to camera coordinates
        camera_coords = extrinsics @ np.concatenate([coords, np.ones(1)])
        camera_coords = camera_coords[:3] / camera_coords[3]

        # convert camera coordinates to pixel coordinates
        image_coords = intrinsics @ camera_coords
        image_coords = image_coords[:2] / image_coords[2]
            
        return jnp.round(image_coords).astype(jnp.int32)
   
    def get_camera_params(self, camera_name):
        """Returns the camera parameters."""
        intrinsics = self._get_camera_intrinsics(camera_name, self.overhead_camera_height, self.overhead_camera_width)  
        extrinsics = self._get_camera_extrinsics(camera_name)
        return {"intrinsics": intrinsics, "extrinsics": extrinsics}

    def get_camera_metadata(self):
        """Returns the camera parameters."""
        intrinsics = self._get_camera_intrinsics("overhead_camera/overhead_camera", self.overhead_camera_height, self.overhead_camera_width)  
        extrinsics = self._get_camera_extrinsics("overhead_camera/overhead_camera")
        # convert rotation to quat
        quat = R.from_matrix(extrinsics[:3, :3]).as_quat()
        return {
            "intrinsics": {
                "fx": intrinsics[0, 0],
                "fy": intrinsics[1, 1],
                "cx": intrinsics[0, 2],
                "cy": intrinsics[1, 2],  
            },
            "extrinsics": {
                "x": extrinsics[3, 0],
                "y": extrinsics[3, 1],
                "z": extrinsics[3, 2],
                "qx": quat[0],
                "qy": quat[1],
                "qz": quat[2],
                "qw": quat[3],   
            }}

    def prop_pick(self, prop_id):
        """Returns pick pose for a given prop."""
        # get prop pose information
        obj_pose = self.props_info[prop_id]["position"]
        obj_quat = self.props_info[prop_id]["orientation"]
        
        # generate pick pose from object
        obj_rot = R.from_quat(obj_quat)
        obj_rot_mat = obj_rot.as_matrix()
        obj_rot_z = abs(np.rad2deg(np.arctan2(obj_rot_mat[1,0], obj_rot_mat[0,0])))
        obj_rot_z = min([obj_rot_z, obj_rot_z - 90])

        obj_rot = R.from_euler('xyz', [0, 180, obj_rot_z], degrees=True).as_matrix().flatten()
        grasp_quat = np.zeros(4,)
        mujoco.mju_mat2Quat(grasp_quat, obj_rot)

        return np.concatenate([obj_pose, grasp_quat])

    def prop_place(self, prop_id, min_pose=None, max_pose=None):
        """Returns collision free place pose for a given prop."""
        # don't want to mess with actual physics
        dummy_physics = deepcopy(self._physics)
        prop_name = f"{prop_id}/{prop_id}"
        prop_mjcf = [x for x in self.props if x.name == prop_id][0]
        prop = dummy_physics.model.geom(prop_name)
        
        # workspace bounds for place
        if (min_pose is None) and (max_pose is None):
            min_pose = self._cfg.task.initializers.workspace.min_pose.copy()
            max_pose = self._cfg.task.initializers.workspace.max_pose.copy()
        
        def _has_collisions_with_prop(physics, prop):
            prop_geom_ids = [prop.id] # for now only one geom per prop
            contacts = dummy_physics.data.contact
            for prop_id, attributes in self.props_info.items():
                prop_name = attributes["prop_name"]
                check_prop_ = dummy_physics.model.geom(f"{prop_name}/{prop_name}")
            for contact in contacts:
              # Ignore contacts with the table (this is super unclean need to refactor)
              if mj_id2name(dummy_physics.model.ptr, 5, contact.geom1) == "table/table" or mj_id2name(dummy_physics.model.ptr, 5, contact.geom2) == "table/table":
                continue

              if contact.dist <= 0.05 and (contact.geom1 in prop_geom_ids or
                                        contact.geom2 in prop_geom_ids):
                return True

            return False

    
        # attempt to place object in new location and check for collisions
        place_attempts = 0
        while place_attempts < 10000: # TODO: move to config param
            # generate random place pose
            place_position_sampler = distributions.Uniform(
                    min_pose,
                    max_pose,
                    )
            place_position = place_position_sampler(random_state=self.prop_place_random_state)

            # generate place orientation
            obj_rot = R.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix().flatten()
            place_quat = np.zeros(4,)
            mujoco.mju_mat2Quat(place_quat, obj_rot)

            # set object to sampled place pose
            prop_mjcf.set_pose(dummy_physics, place_position, place_quat)
            
            # this step is also required to update contact data
            try:
            # If this pose results in collisions then there's a chance we'll
            # encounter a PhysicsError error here due to a full contact buffer,
            # in which case reject this pose and sample another.
                dummy_physics.forward()
                # check for collisions and if none return place pose
                if not _has_collisions_with_prop(dummy_physics, prop):
                    del dummy_physics
                    return np.concatenate([place_position, place_quat])    
            except Exception as e:
                print(e)
                continue
            
            # increment place attempts
            place_attempts += 1

        raise Exception("Failed to find collision free place pose.")

    def random_pick_and_place(self):
        """Generate pick/place poses for one object."""
        # pick a random prop
        random_prop = list(self.props_info.keys())[0]
        
        # get prop pose information
        obj_pose = self.props_info[random_prop]["position"]
        obj_quat = self.props_info[random_prop]["orientation"]
        
        # generate pick pose from object
        obj_rot = R.from_quat(obj_quat)
        obj_rot_mat = obj_rot.as_matrix()
        obj_rot_z = np.rad2deg(np.arctan2(obj_rot_mat[1,0], obj_rot_mat[0,0]))
        obj_rot = R.from_euler('xyz', [0, 180, obj_rot_z], degrees=True).as_matrix().flatten()
        grasp_quat = np.zeros(4,)
        mujoco.mju_mat2Quat(grasp_quat, obj_rot)
        
        
        # sample place pose from workspace
        place_position_sampler = distributions.Uniform(
            self._cfg.task.initializers.workspace.min_pose,
            self._cfg.task.initializers.workspace.max_pose,
        )
        place_position = place_position_sampler(random_state=np.random.RandomState())
        place_quaternion = grasp_quat
        

        pick_pose = np.concatenate([obj_pose, grasp_quat])
        place_pose = np.concatenate([place_position, place_quaternion])
        

        return pick_pose, place_pose

    def sort_colours(self): 
        """Generates pick/place action for sorting coloured blocks"""  

        def get_location_bounds(target_location):
            min_x = target_location['location'][0] - target_location['size'][0]/2
            max_x = target_location['location'][0] + target_location['size'][0]/2
            min_y = target_location['location'][1] - target_location['size'][1]/2
            max_y = target_location['location'][1] + target_location['size'][1]/2
            min_z = 0.4  # Hardcoded value for Z
            max_z = 0.4  # Hardcoded value for Z
            return np.array((min_x, min_y, min_z)), np.array((max_x, max_y, max_z))

        def is_within_target(target, position):
            """
            Check if point is within bounds of target location (rectangle).
            """
            min_pose, max_pose = get_location_bounds(target_location)
            min_x = min_pose[0]
            max_x = max_pose[0]
            min_y = min_pose[1]
            max_y = max_pose[1]
            x, y, _ = position

            if min_x <= x <= max_x and min_y <= y <= max_y:
                return True
            else:
                return False

            
        for prop_id, attributes in self.props_info.items():
            prop_pos = attributes["position"]
            prop_name = attributes["prop_name"]
            prop_colour = prop_name.split("_")[1]

            # check if prop is within bounds of target location
            target_location_name = self._cfg.task.colour_target_map[prop_colour]
            target_location = self._cfg.task.target_locations[target_location_name]

            if not is_within_target(target_location, prop_pos):
                # get the pick pose
                pick_pose = self.prop_pick(prop_id)
                
                # sample collision-free place pose within target
                min_pose, max_pose = get_location_bounds(target_location)
                place_pose = self.prop_place(prop_name, min_pose, max_pose)

                return True, pick_pose, place_pose
            else:
                continue

        return False, None, None

    def interactive_tuning(self):
        """
        Interactively control arm to tune simulation parameters. 
        """

        # get difference between eef site and mocap body
        mocap_pos = self._physics.data.mocap_pos[0]
        mocap_quat = self._physics.data.mocap_quat[0]

        # update control target
        self._robot.arm_controller.set_target(
            position=mocap_pos + [0.0, 0.0, 0.175],
            quat=mocap_quat, 
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            )

        arm_command = self._robot.arm_controller.compute_control_output()
        gripper_command = np.array([self._robot.end_effector_controller.compute_control_output()])
        control_command = np.concatenate((arm_command, gripper_command))

        # step the simulation
        for _ in range(5):
            self._physics.set_control(control_command)
            self._physics.step()
            if self.passive_view is not None:
                self.passive_view.sync()
    
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
                "simulation_tuning_mode=True"
                ]
                )

    # instantiate color separation task
    env = RearrangementEnv(viewer=True, cfg=COLOR_SEPARATING_CONFIG) 

    # expert demonstration
    _, _, _, obs = env.reset()
    while env.sort_colours()[0]:
        _, pick_pose, place_pose = env.sort_colours()
        
        pick_action = {
            "pose": pick_pose,
            "pixel_coords": env.world_2_pixel("overhead_camera/overhead_camera", pick_pose[:3]),
            "gripper_rot": None,
        }

        place_action = {
            "pose": place_pose,
            "pixel_coords": env.world_2_pixel("overhead_camera/overhead_camera", place_pose[:3]),
            "gripper_rot": None,
        }

        _, _, _, obs = env.step(pick_action)
        _, _, _, obs = env.step(place_action)

    # interactive control of robot with mocap body
    _, _, _, obs = env.reset()
    while True:
        env.interactive_tuning()

    env.close()
    
