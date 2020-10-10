import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import random
from gym.spaces import Box, MultiDiscrete
import copy

from gym_edgesimulator.util import plot_color
from gym_edgesimulator.util import plot_black


from gym_edgesimulator.envs.edgesim_v1 import EdgeSimV1
from gym_edgesimulator import Simulator


class EdgeSimV3(EdgeSimV1):
    """
        make a fake simulator for initial gym integration,
        it builds a random initialization of servers, stations,
        memory and then simulate movement
        observations: concatenate(services_servers, users_stations)
        actions:      does not exist (actions come from greedy algorithm)
        rewards:      dones not exist
    """

    def __init__(self, initial_state, allowed_moves,
                 SPEED_LIMIT,
                 seed):
        super().__init__(initial_state, allowed_moves, SPEED_LIMIT,
                         penalty_illegal=1, penalty_normal=1,
                         penalty_latency=1, seed=seed)


    def step(self, action):

        """ 
            Args:
                action (object): it just gets the action to just
                preserve the gym interface but the action is actually
                calculated in the step itself
        """
        # make a random movements in users in the sim object
        # (in place and without return)
        # and return new sample for users_stations
        self.users_stations = self.sim.sample_users_stations()
        self.users_distances = self.sim.users_distances
        # users's random movements
        self.observation[self.num_of_services:] = self.users_stations


        # to add movement of services to the simulator
        greedy_action = self.sim.next_greedy_action(self.servers_mem, self.services_mem)
        self.observation[0:self.num_of_services] = greedy_action
        self.sim.update_services_servers(self.observation[0:self.num_of_services])

        # no episode or done here
        done = False
        reward = 0

        # info dicts
        num_of_cons = self._num_of_consolidated(self.observation[0:self.num_of_services])
        fraction_of_latency = self._fraction_latency(self.users_distances)

        # users_servers only for debugging pruposees
        users_servers = np.array(list(map(lambda i: self.observation[i], self.users_services)))

        info = {'users_services':self.users_services,
                'services_servers':self.observation[0:self.num_of_services],
                'users_servers':users_servers,
                # 'actions_index': action[:self.allowed_moves],
                # 'actions_moves': action[self.allowed_moves:],
                'num_of_consolidated':num_of_cons,
                'fraction_of_latency':fraction_of_latency,
                'users_stations':self.users_stations,
                'users_distances':self.users_distances,
                'done': done,
                'reward': reward,
                'simulator': self.sim}
        return self.observation, reward, done, info

