import numpy as np
import tensorflow as tf
import pickle
import json
import os
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from gym_edgesimulator.callbacks import TensorboardCallback
from gym_edgesimulator.callbacks import DebugCallback
import gym_edgesimulator

from constants import DATASETS_BASE_PATH, MODELS_PATH


class CheckScripts:
    # TODO come back here and fix the dataset path
    def __init__(self, dataset, type_env, allowed_moves=5, initial_obs=False):
        self.dataset = dataset
        self.type_env = type_env
        self.dir2load = os.path.join(DATASETS_BASE_PATH, str(dataset))
        self.dir2save = os.path.join(MODELS_PATH, str(dataset) ,f"v{type_env}")
        self.allowed_moves = allowed_moves

        if not os.path.isdir(self.dir2load):
            raise RuntimeError('dataset does not exists')

        with open('{}/info.txt'.format(self.dir2load), 'r') as in_file:
            self.info_str = in_file.read()

        with open('{}/initial_state.pickle'.format(self.dir2load), 'rb') as in_pickle:
            self.initial_state = pickle.load(in_pickle)

        # TODO write a @classmethod for this
        if initial_obs:
            with open('{}/initial_observations.pickle'.format(self.dir2load), 'rb') as in_pickle:
                self.initial_observations = pickle.load(in_pickle)
        else:
            self.initial_observations = False

        for k,v in self.initial_state.items(): exec('self.'+k+'=v') # TODO future changes:  All initial observation and other stuff as the original object not dictionary

        if type_env == 0:
            self.env = gym.make('EdgeSim-v0', initial_state=self.initial_state,
                                allowed_moves=self.allowed_moves, penalty_illegal=-100, penalty_normal=-1,
                                penalty_consolidated=1000, seed=1)
        elif type_env == 1:
            self.env = gym.make('EdgeSim-v1', initial_state=self.initial_state,
                                allowed_moves=self.allowed_moves, SPEED_LIMIT=10, penalty_illegal=-100,
                                penalty_normal=-1, penalty_latency=1000, seed=1)
        elif type_env == 2:
            self.env = gym.make('EdgeSim-v2', initial_state=self.initial_state,
                                allowed_moves=self.allowed_moves, SPEED_LIMIT=10, penalty_illegal=-100,
                                penalty_normal=-1, penalty_consolidated=1000, cons_w=0.5, lat_w=0.5,
                                penalty_latency=-100, seed=1)

    @classmethod
    def with_model(cls, dataset, type_env, model, allowed_moves=5, initial_obs=False):
        
        ins = cls(dataset=dataset, type_env=type_env, allowed_moves=allowed_moves, initial_obs=initial_obs)

        ins.dir2load_model = os.path.join(MODELS_PATH, f"{ins.dataset}/v{type_env}/{model}")
        ins.model = PPO2.load(f"{ins.dir2load_model}/model")
        with open(f"{ins.dir2load_model}/learning_report.json") as in_file:
            learning_report = json.loads(in_file.read())

        # get the number of allowed moves

        ins.allowed_moves = learning_report['allowed_moves']        
        return ins

    def is_on_GPU(self):
        """
            check if the GPU is on
        """
        if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
            print("module loaded")
        else:
            print("module not loaded, load it with:")
            print("module load cuda/10.2-cudnn7.6.5")


    def check_env(self):  
        observation = self.env.reset()
        print('\n\n===================== reseting!!! =====================')
        self.env.render(type_of='black')
        print('-------------------------------')
        for i in range(20):
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            self.env.render(type_of='black')
            # print(observation, reward, done, info)
            print('-------------------------------')
            if done:
                observation = self.env.reset()
                print('\n\n===================== reseting!!! =====================')


    def learn(self, total_timesteps):
        # TODO make it consistent with the commandline scripts
        model = PPO2(MlpPolicy, self.env, verbose=1,
                     tensorboard_log=f'{self.dir2save}/v{self.type_env}/logs')

        model.learn(total_timesteps=total_timesteps, callback=DebugCallback())


    def check_learned(self):
        if not self.initial_observations:
            obs = self.env.reset()
            done = False
            self.env.render(type_of='black')
            print('next state:\n------------------------------------------------')
            while not done:
                action, _states = self.model.predict(obs)
                print(f"action: {action}")
                obs, rewards, done, info = self.env.step(action)
                self.env.render(type_of='black')
                print('next state:\n------------------------------------------------')

        else:
            for initial_state in self.initial_observations:
                print('next initial observation:\n==========================================')
                obs = self.env.reset()
                done = False
                self.env.render(type_of='black')
                print('next state:\n------------------------------------------------')
                while not done:
                    action, _states = self.model.predict(obs)
                    print(f"action: {action}")
                    obs, rewards, done, info = self.env.step(action)
                    self.env.render(type_of='black')
                    print('next state:\n------------------------------------------------')

def main():
    # ins = CheckScripts.with_model(dataset=3, type_env=1, model=1)
    ins = CheckScripts(dataset=3, type_env=2)
    # ins.is_on_GPU()
    ins.check_env()
    # ins.learn(total_timesteps=15)
    # ins.check_learned()

if __name__ == '__main__':
    main()