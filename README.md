# MuJoCo Robotics Environments 🤖

A set of software for creating and prototyping robot workspaces in MuJoCo.

## Getting Started

### Local Development 

Ensure all submodule are cloned with the following commands:

```bash
git submodule sync
git submodule update --init --recursive
```

Ensure you have python version 3.10.6 using [pyenv](https://github.com/pyenv/pyenv). Following this install the python virtual environment for this project with [poetry](https://python-poetry.org/) through running:

```bash
poetry install 
```

Activate the virtual environment with the following command:

```bash
poetry shell
```

### Package Installation 

Install from PYPI via (Note: for now only Linux is supported): 
```bash
pip install mujoco_robot_environments
```

Example usage:

```python
from mujoco_robot_environments.tasks.rearrangement import RearrangementEnv

env = RearrangementEnv(viewer=True)
env.reset()
...
```

## Task Environments

### Colour Sorting Task Environment

An toy environment for reasoning about visual semantics. This environment is compatible with training transporter networks and intended to be used in research and for robot learning education.

#### Scripted Expert Demonstrations (Data Collection)
[Screencast from 05-10-2024 11:45:47 AM.webm](https://github.com/peterdavidfagan/mujoco_robot_environments/assets/42982057/7ac279da-0268-4ef2-8d4a-85eed6a7f364)


#### Interactive Environment Tuning
[Screencast from 05-10-2024 10:44:19 AM.webm](https://github.com/peterdavidfagan/mujoco_robot_environments/assets/42982057/b4428fff-f58f-4f96-b91f-6c171afb20a2)



