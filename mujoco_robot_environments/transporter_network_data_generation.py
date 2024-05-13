"""Generating data for the transporter network."""

import os
import sys
import subprocess
from absl import logging
import time

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from mujoco_robot_environments.tasks.rearrangement import RearrangementEnv

import envlogger
from envlogger.backends import tfds_backend_writer

import hydra
from hydra import compose, initialize

# clear hydra global state to avoid conflicts with other hydra instances
hydra.core.global_hydra.GlobalHydra.instance().clear()
initialize(version_base=None, config_path="./config", job_name="default_config")

# define known task configurations
COLOR_SEPERATOR_TASK_CONFIG = compose(
        config_name="rearrangement",
        overrides=[
            "+name=colour_splitter",
            "arena/props=colour_splitter"
            ]
        )

#COLOR_SEPERATOR_TASK_CONFIG.name = "colour_splitter"

if __name__=="__main__":
    
    # TODO: add argparse for task list
    TASKS = [COLOR_SEPERATOR_TASK_CONFIG]

    for task_config in TASKS:
        current_timestamp = time.localtime()
        human_readable_timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", current_timestamp)
        DATASET_NAME = f"{task_config.name}_{human_readable_timestamp}"

        # set up the data folder
        DATA_DIR = os.path.join(os.getcwd(), "data", DATASET_NAME)
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        # define base dataset configuration across all transporter tasks
        for camera in task_config.arena.cameras:
            if camera.name == "overhead_camera":
                camera_height = camera.height
                camera_width = camera.width

        ds_config = tfds.rlds.rlds_base.DatasetConfig(
                name=DATASET_NAME,
                observation_info=tfds.features.FeaturesDict({
                    "overhead_camera/rgb": tfds.features.Tensor(shape=(camera_height, camera_width, 3), dtype=tf.uint8),
                    "overhead_camera/depth": tfds.features.Tensor(shape=(camera_height, camera_width), dtype=tf.float32),
                    }),
                action_info=tfds.features.FeaturesDict({
                            "pose": tfds.features.Tensor(shape=(7,), dtype=np.float64),
                            "pixel_coords": tfds.features.Tensor(shape=(2,), dtype=np.int32),
                            "gripper_rot": np.float64,
                        }),                
                reward_info=np.float64,
                discount_info=np.float64,
                episode_metadata_info={
                    "intrinsics":{
                        "fx": tf.float64,
                        "fy": tf.float64,
                        "cx": tf.float64,
                        "cy": tf.float64,  
                        },
                    "extrinsics":{
                        "x": tf.float64,
                        "y": tf.float64,
                        "z": tf.float64,
                        "qx": tf.float64,
                        "qy": tf.float64,
                        "qz": tf.float64,
                        "qw": tf.float64,   
                    },
                },
                )

        def calibration_metadata(timestep, unused_action, unused_env):
            """
            Store camera calibration params as episode metadata.
            """
            if timestep.first:
                return unused_env.get_camera_metadata()
            else:
                return None

        episode_idx=0
        while (task_config.dataset.num_episodes - episode_idx>=0):
            # instantiate task environment
            env = RearrangementEnv(task_config)
            
            # collect data with envlogger
            with envlogger.EnvLogger(
                    env,
                    episode_fn=calibration_metadata,
                    backend=tfds_backend_writer.TFDSBackendWriter(
                        data_directory=DATA_DIR,
                        split_name="train", # for now default to train and eval on environment directly
                        max_episodes_per_file=task_config.dataset.max_episodes_per_file,
                        ds_config=ds_config),
                    ) as env:
                start_idx = episode_idx
                for i in range(10):
                    try:
                        episode_idx += 1
                        _, _, _, obs = env.reset()
                        for _ in range(task_config.dataset.max_steps):
                            in_progress, pick_pose, place_pose = env.sort_colours()
                            if not in_progress:
                                print("Task demonstration is complete")
                                break

                            pick_action = {
                                "pose": pick_pose,
                                "pixel_coords": env.world_2_pixel("overhead_camera/overhead_camera", pick_pose[:3]),
                                "gripper_rot": 0.0,
                            }

                            place_action = {
                                "pose": place_pose,
                                "pixel_coords": env.world_2_pixel("overhead_camera/overhead_camera", place_pose[:3]),
                                "gripper_rot": 0.0,
                            }

                            _, _, _, obs = env.step(pick_action)
                            _, _, _, obs = env.step(place_action)
                    except Exception as e:
                        print("Task demonstration failed with exception: {}".format(e))
                        break
                    
                    if (task_config.dataset.num_episodes - episode_idx) <= 0:
                        break
            env.close()
    
    # upload data to huggingface
    subprocess.call(['python', './hf_scripts/hf_data_upload.py', '+config=transporter'])

