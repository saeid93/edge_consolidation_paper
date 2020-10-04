import pickle
import json
import os
import sys
import time
import datetime
import random
import click

import gym
import gym_edgesimulator

import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from gym_edgesimulator.callbacks import TensorboardCallback
from gym_edgesimulator.callbacks import ObjReturnCallback


# ------ environment v0 ------
def learn_v0(initial_state, dir2save,
             total_timesteps, allowed_moves,
             penalty_illegal, penalty_normal, penalty_consolidated):

    # make the environment
    env = gym.make('EdgeSim-v0', initial_state=initial_state,
                   allowed_moves=allowed_moves,
                   penalty_illegal=penalty_illegal,
                   penalty_normal=penalty_normal,
                   penalty_consolidated=penalty_consolidated,
                   seed=1)

    # learning model initialization
    model = PPO2(MlpPolicy, env , verbose=1,
                 tensorboard_log=f'{dir2save}/logs')
    tensorboard = TensorboardCallback()
    obj_return = ObjReturnCallback()

    # learning
    model.learn(total_timesteps=total_timesteps,
                callback=[tensorboard, obj_return])

    # report
    learning_report = {
        "total_timesteps": total_timesteps,
        "allowed_moves": allowed_moves,
        "penalty_illegal": penalty_illegal,
        "penalty_normal": penalty_normal,
        "penalty_consolidated": penalty_consolidated
        }

    return model, learning_report


# ------ environment v1 ------
def learn_v1(initial_state, dir2save, total_timesteps,
             allowed_moves, SPEED_LIMIT,
             penalty_illegal, penalty_normal, penalty_latency):
    
    # make the environment    
    env = gym.make('EdgeSim-v1', initial_state=initial_state,
                   allowed_moves=allowed_moves,
                   SPEED_LIMIT=SPEED_LIMIT,
                   penalty_illegal=penalty_illegal,
                   penalty_normal=penalty_normal,
                   penalty_latency=penalty_latency,
                   seed=1)

    # learning model initialization
    model = PPO2(MlpPolicy, env, verbose=1,
                 tensorboard_log=f'{dir2save}/logs')
    tensorboard = TensorboardCallback()
    obj_return = ObjReturnCallback()

    # learning
    model.learn(total_timesteps=total_timesteps,
                callback=[tensorboard, obj_return])

    # report
    learning_report = {
        "total_timesteps" : total_timesteps,
        "allowed_moves" : allowed_moves,
        "SPEED_LIMIT" : SPEED_LIMIT,
        "penalty_illegal" : penalty_illegal,
        "penalty_normal" : penalty_normal,
        "penalty_latency" : penalty_latency
        }

    return model, learning_report


# ------ environment v2 ------
def learn_v2(initial_state, dir2save,
             total_timesteps, allowed_moves, cons_w, lat_w, SPEED_LIMIT,
             penalty_illegal, penalty_normal, penalty_consolidated,
             penalty_latency):

    # make the environment
    env = gym.make('EdgeSim-v2', initial_state=initial_state,
                   cons_w=cons_w, lat_w=lat_w,
                   allowed_moves=allowed_moves,
                   SPEED_LIMIT=SPEED_LIMIT,
                   penalty_illegal=penalty_illegal,
                   penalty_normal=penalty_normal,
                   penalty_consolidated=penalty_consolidated,
                   penalty_latency=penalty_latency,
                   seed=1)

    # learning model initialization
    model = PPO2(MlpPolicy, env, verbose=1,
                 tensorboard_log=f'{dir2save}/logs')
    tensorboard = TensorboardCallback()
    obj_return = ObjReturnCallback()

    # learning
    model.learn(total_timesteps=total_timesteps,
                callback=[tensorboard, obj_return])

    # report
    learning_report = {
        "total_timesteps": total_timesteps,
        "allowed_moves": allowed_moves,
        "cons_w": cons_w,
        "lat_w": lat_w,
        "SPEED_LIMIT": SPEED_LIMIT,
        "penalty_illegal": penalty_illegal,
        "penalty_normal": penalty_normal,
        "penalty_consolidated": penalty_consolidated,
        "penalty_latency": penalty_latency
        }

    return model, learning_report

# ------ environment v3 ------
def learn_v3(initial_state, dir2save, total_timesteps,
             allowed_moves, SPEED_LIMIT):
    
    # make the environment    
    env = gym.make('EdgeSim-v3', initial_state=initial_state,
                   allowed_moves=allowed_moves,
                   SPEED_LIMIT=SPEED_LIMIT,
                   seed=1)

    # learning model initialization
    model = PPO2(MlpPolicy, env, verbose=1,
                 tensorboard_log=f'{dir2save}/logs')
    tensorboard = TensorboardCallback()
    obj_return = ObjReturnCallback()

    # learning
    model.learn(total_timesteps=total_timesteps,
                callback=[tensorboard, obj_return])

    # report
    learning_report = {
        "total_timesteps" : total_timesteps,
        "allowed_moves" : allowed_moves,
        "SPEED_LIMIT" : SPEED_LIMIT,
        }

    return model, learning_report