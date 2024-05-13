# MuJoCo Robotics Environments 🤖

## Overview

    ├──── config                                       # hydra configs to define task environment and experiments
    ├──── environment                                  # files to set up cameras and base environment
    ├──── models                                       # models of robots and arenas (currently contains EBMs these are to be moved)
    ├──── mujoco_pkgs                                  # dependencies as gitsubmodules (control software, mjcf models etc.)
    ├──── mujoco_ros_env_generation                    # generating mjb files for ROS 2 integration
    ├──── tasks                                        # gym style rearrangement environments
    ├── transporter_network_data_generation.py         # script for generating transporter network demonstration datasets

## Running Task Environment Demos ✅

In order to run live demonstrations of task environments simply run the corresponding python script in the task folder. For instance to run the rearrangement environment one would run:

```python
python tasks/rearrangement.py
```

## Generating Transporter Datasets 🧑‍💻

Simply run the `transporter_network_data_generation.py` script. It is important to note that this will overwrite existing raw data in the repository.

```python
python transporter_network_data_generation.py
```
