"""Tuning control params with genetic algorithm."""

from typing import Tuple, Dict, Optional, Union, List

import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

import mujoco
from mujoco import viewer

import dm_env
from dm_control import composer, mjcf

from mujoco_controllers.build_env import construct_physics
from mujoco_controllers.osc import OSC

from mujoco_robot_environments.tasks.rearrangement import RearrangementEnv
from hydra import compose, initialize
from hydra.utils import instantiate

import numpy as np
import jax
import jax.numpy as jnp
from evosax import CMA_ES
from evosax.strategies.cma_es import get_cma_elite_weights, EvoState


class RearrangementEnvTuner(RearrangementEnv):
    """ITL Rearrangement task with fitness functions for tuning controllers."""
    def __init__(self, config, viewer=None):
        super().__init__(config, viewer)  
    
    def calculate_high_manipulability_config(self):
        """
        Calculate configuration with highest manipulability in order to avoid singularities.

        Use this configuration as a secondary task using the nullspace projection.
        """
        pass       

    # evaluate controller on p2p motion
    def p2p_fitness(self):
        """Run controller to single target."""
        # sample target pose within workspace
        position_target = np.random.uniform(
                low=self._cfg.task.initializers.workspace.min_pose, 
                high=self._cfg.task.initializers.workspace.max_pose, 
                size=(3,))
        position_target += np.array([0, 0, np.random.uniform(0.15, 0.55)]) # add offset to avoid collision with table
        orientation_target = np.zeros(4)
        mat = R.from_euler('xyz', [0, 180, np.random.uniform(-2*np.pi, 2*np.pi)], degrees=True).as_matrix().reshape(9,1) 
        mujoco.mju_mat2Quat(orientation_target, mat)

        # set sampled pose as target and run controller
        self._robot.arm_controller.eef_target_position = position_target
        self._robot.arm_controller.eef_target_quat = orientation_target
        self._robot.arm_controller.eef_target_velocity = np.zeros(3)
        self._robot.arm_controller.eef_target_angular_velocity = np.zeros(3)
        success_flag = self._robot.run_controller(4.0)

        # calculate fitness
        failure_penalty = 0.0 if success_flag else 1e6 
        position_error = self._robot.arm_controller.current_position_error()
        orientation_error = self._robot.arm_controller.current_orientation_error()
        print("position error: ", position_error)
        print("orientation error: ", orientation_error)
        fitness =  (1e3*position_error) + (orientation_error) + failure_penalty

        return fitness

    # evaluate controller on tracking circle
    def circle_fitness(self, radius=0.1, speed=0.1):
        """Run controller to track circle."""
        # fix the orientation
        orientation_target = np.zeros(4)
        mat = R.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix().reshape(9,1) 
        mujoco.mju_mat2Quat(orientation_target, mat)
        
        # sample 100 waypoints on circle
        waypoints = []
        for i in range(20):
            angle = i * 2 * np.pi / 100
            x = radius * np.cos(angle) + 0.25
            y = radius * np.sin(angle) 
            waypoints.append(np.array([x, y, 0.9]))
        
        # go to initial waypoint
        self._robot.arm_controller.eef_target_position = waypoints[0]
        self._robot.arm_controller.eef_target_quat = orientation_target
        self._robot.arm_controller.eef_target_velocity = np.zeros(3)
        self._robot.arm_controller.eef_target_angular_velocity = np.zeros(3)
        success_flag = self._robot.run_controller(5.0)

        # run controller to track waypoints
        success_flag = True
        for waypoint in waypoints:
            self._robot.arm_controller.eef_target_position = waypoint
            self._robot.arm_controller.eef_target_quat = orientation_target 
            self._robot.arm_controller.eef_target_velocity = np.zeros(3)
            self._robot.arm_controller.eef_target_angular_velocity = np.zeros(3)
            success_flag = self._robot.run_controller(1.0)
            if not success_flag:
                raise ValueError("Controller failed to track waypoints.")

        # calculate fitness
        failure_penalty = 0.0 if success_flag else 1e6
        fitness = failure_penalty

        return fitness

    # evaluate controller on pick and place task for rearrangement
    def _compute_reward(self, place_target):
        """Distance of placed object to target place."""
        # compute distance of placed object to target place
        object_pos = self.prop_info[next(iter(self.prop_info))]["position"]
        dist_to_target = np.linalg.norm(place_target[:3] - object_pos)
        return (1000 * dist_to_target)
    
    def step(self, pick_target, place_target):
        """Wrap parent method as fitness calculation requires place target."""
        step_type, _, discount, observation = super().step(pick_target, place_target)
        reward = self._compute_reward(place_target)    
        return dm_env.TimeStep(
                step_type, 
                reward, 
                discount, 
                observation
                )

if __name__=="__main__":
    # read hydra config
    initialize(version_base=None, config_path="../config", job_name="default_config")
    TUNING_CONFIG = compose(config_name="rearrangement", overrides=[
        "arena/props=single_block", # ensure only one block is spawned
        "physics_dt=0.001",
        "robots.arm.controller_config.controller_params.control_dt=0.005",
        ])

    # create simulation environment
    env = RearrangementEnvTuner(TUNING_CONFIG, viewer=False)
    
    # initialize evolution strategy
    rng = jax.random.PRNGKey(0)
    strategy = CMA_ES(
            popsize=20,
            num_dims=6,
            sigma_init=500,
            )
    params = strategy.params_strategy
    params = params.replace(init_min=500, init_max=500)
    state = strategy.initialize(rng, params)   

    def eval_fitness(params, method=env.step):
        """
        Evaluate time to reach target and convergence.

        For now the target is fixed but in future test more diverse targets.
        """
        # reset the environment
        env.reset()

        # joint specific task gains
        env._robot.arm_controller.controller_gains = {
                "position": {"kp": float(params.at[0].get()), "kd": float(params.at[1].get())},
                "orientation": {"kp": float(params.at[2].get()), "kd": float(params.at[3].get())},
                "nullspace": {"kp": float(params.at[4].get()), "kd": float(params.at[5].get()),}
                }

        # evaluate controller on test case
        # Note: we require try/except for official step method as it raises exception on convergence failure
        if method == env.step:
            try:
                poses = env.random_pick_and_place()
                _, fitness, _, _ = env.step(*poses)
                return fitness
            except Exception as e:
                print(f"Task execution failed with exception: {e}")
                return 1e6
        else:
            return method()

    # evaluate controller on tasks of increasing difficulty
    for eval_method in [env.p2p_fitness, env.step]:
        for generation in range(50):
            rng, rng_gen, rng_eval = jax.random.split(rng, 3)
            x, state = strategy.ask(rng_gen, state, params)
            fitness = []
            for param in x:
                param = jnp.abs(param) # only consider positive params
                fitness.append(eval_fitness(param, method=eval_method))
            state = strategy.tell(x, jnp.array(fitness), state, params)
            print(f"Generation {generation}, best fitness: ", state.best_fitness)
            print(f"Generation {generation}, best params: ", jnp.abs(state.best_member))

        if state.best_fitness >= 1e6:
            raise ValueError("Controller failed to converge on task")

    # TODO: write to file
    # get best params and fitness
    best_params = state.best_member
    best_fitness = state.best_fitness
    
    # write params to text file
    with open("rearrangement_control_params.txt", "w") as f:
        f.write("position kp, position kd, orientation kp, orientation kd, nullspace kp, nullspace kd\n")
        f.write(str(best_params))

    print("Best params: ", best_params)
    print("Best fitness: ", best_fitness) 
