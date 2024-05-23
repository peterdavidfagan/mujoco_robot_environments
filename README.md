# MuJoCo Robotics Environments 🤖

Software for creating and experimenting with robot workspaces in MuJoCo.

## Installation 

### PyPI
Install from package from PyPI with: 
```bash
pip install mujoco_robot_environments
```

### Local Development 

Clone this GitHub repository and ensure you have python version 3.10.6 using [pyenv](https://github.com/pyenv/pyenv). Following this install the python virtual environment for this project with [poetry](https://python-poetry.org/) through running:

```bash
poetry install 
```

Activate the virtual environment with the following command:

```bash
poetry shell
```

## Example Usage

### Instantiating shipped environments

```python
from mujoco_robot_environments.tasks.rearrangement import RearrangementEnv

env = RearrangementEnv(viewer=True)
env.reset()
...
```

### Overriding Environment Configurations

Most environments ship with a default set of logic for scene initialization and other settings you may wish to override/customise.
In order to override the default config you need use the [hydra override](https://hydra.cc/docs/advanced/override_grammar/basic/) and follow the hierarchical structure of the [configuration files](https://github.com/peterdavidfagan/mujoco_robot_environments/tree/main/mujoco_robot_environments/config) seen shipped with the repository. As an example to change the sampling settings for the `RearrangementEnv` one can follow the below syntax:

```python
 # clear hydra global state to avoid conflicts with other hydra instances
hydra.core.global_hydra.GlobalHydra.instance().clear()

# read hydra config
initialize(version_base=None, config_path=<relative path to config yaml files>, job_name="rearrangement")

# add task configs
COLOR_SEPARATING_CONFIG = compose(
        config_name="rearrangement",
        overrides=[
            "arena/props=colour_splitter",
            ]
            )

# instantiate color separation task
env = RearrangementEnv(viewer=True, cfg=COLOR_SEPARATING_CONFIG) 

# expert demonstration
_, _, _, obs = env.reset()
```

Where in the above example `<relative path to config yaml files>` points towards a directory of yaml files of the same structure as those shipped with this repository but containing your custom configuration files.

## Task Environments

### Colour Sorting Task Environment

An toy environment for reasoning about visual semantics through rearranging objects of varying shapes/colours into target locations.

[Screencast from 05-10-2024 11:45:47 AM.webm](https://github.com/peterdavidfagan/mujoco_robot_environments/assets/42982057/7ac279da-0268-4ef2-8d4a-85eed6a7f364)






