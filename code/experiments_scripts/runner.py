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

from constants import DATASETS_BASE_PATH, MODELS_PATH
from learners import learn_v0, learn_v1, learn_v2, learn_v3
from event_to_csv import to_csv
from plot_graphs import plot_graphs

def learn(type_env, dataset, num_of_repeat, **params):

    # set the directories
    dir2load = os.path.join(DATASETS_BASE_PATH, str(dataset))
    dir2save = os.path.join(MODELS_PATH, str(dataset), f'v{type_env}')

    # make the directory to save the result
    if not os.path.isdir(dir2save):
        os.makedirs(dir2save)

    # initial state of the environment
    with open(f'{dir2load}/initial_state.pickle', 'rb') as in_pickle:
        initial_state = pickle.load(in_pickle)

    # setup session folder
    content = os.listdir(dir2save)
    turn = len(content)+1
    dir2save_session = f'{dir2save}/{turn}'
    if os.path.isdir(dir2save_session): turn+=1
    os.makedirs(dir2save_session)

    for i in range(num_of_repeat):
        dir2save_per_repeat = f'{dir2save_session}/{i}'
        os.makedirs(dir2save_per_repeat)

        # directoris for each single run of the session
        os.makedirs(f'{dir2save_per_repeat}/logs')

        # send the environment to approperiated learning function
        params.update(initial_state=initial_state)
        params.update(dir2save=dir2save_per_repeat)
        # params.update(turn=turn)
        if type_env == 0:
            model, learning_report = learn_v0(**params)
        elif type_env == 1:
            model, learning_report = learn_v1(**params)
        elif type_env == 2:
            model, learning_report = learn_v2(**params)
        elif type_env == 3:
            model, learning_report = learn_v3(**params)

        learning_report.update(dataset=dataset)

        # Save the Model
        model.save(f"{dir2save_per_repeat}/model")
        print(f'model saved in {dir2save_per_repeat}/model')
        # save a learning report in json format
        with open(f"{dir2save_per_repeat}/learning_report.json", 'x') as out_file:
            out_file.write(json.dumps(learning_report, indent=4))
        # convert logs into csv
        to_csv(f'{dir2save_per_repeat}/logs/PPO2_1', f"{dir2save_per_repeat}/csv_results")
    # plot averaged graphs
    plot_graphs(dir2save_session)
    

@click.command()
@click.argument('config_file')
def main(config_file):

    with open(config_file) as cf:
        config = json.loads(cf.read())

    learn(**config)


if __name__ == "__main__":
    main()