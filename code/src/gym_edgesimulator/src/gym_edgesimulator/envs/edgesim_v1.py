import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import random
from gym.spaces import Box, MultiDiscrete
import copy

from gym_edgesimulator.util import plot_color
from gym_edgesimulator.util import plot_black

from gym_edgesimulator.envs.edgesim_v0 import EdgeSimV0
from gym_edgesimulator import Simulator

# TODO check learning and rewards - end to end debug to see
# if all the rewards in all states are computed correctly

class EdgeSimV1(EdgeSimV0):
    """
        The base Edge simulator environment
        with only the latency no consolidation
        the servers and services.
        EdgeSim-v1
        observations --> concatenate(services_servers, users_stations)
    """
    # inherited _reward_illegal(self, penalty):
    # inherited _reward_not_done(self, penalty):
    # inherited _reward_consolidation(self, penalty, observation):
    # inherited _num_of_consolidated(observation):
    # inherited _plot(service_server, server_mem, used_by_service):

    def __init__(self, initial_state, allowed_moves,
                 SPEED_LIMIT,
                 penalty_illegal, penalty_normal, penalty_latency,
                 seed):
        # this four variables need to be initialized
        # before the reset in the parent initializer
        self.initial_sim = initial_state['simulation']
        self.users_services = initial_state['users_services']
        self.num_of_users = initial_state['simulation'].num_of_users
        self.num_of_stations = initial_state['simulation'].num_of_stations
        # call the EdgeSimV0 initializer
        super().__init__(initial_state, allowed_moves, penalty_illegal,
                         penalty_normal, penalty_consolidated=1, seed=seed)
        # find the servers' and services' numbers
        # from the input arrays
        self.PERC_LAT = 0.4
        self.services_desired_latency = initial_state['services_desired_latency']
        self.initial_sim.SPEED_LIMIT = SPEED_LIMIT
        self.penalty_latency = penalty_latency

    def reset(self):
        """
            Resets the state of the environment and returns an initial observation.
            Returns:
                observation (object): the initial observation.
            Remarks:
                each time resets to a different initial state
        """
        self.observation = self.initial_observation.copy()
        self.sim = copy.deepcopy(self.initial_sim)
        return self.observation

    def step(self, action):
        """ 
            Args:
                action (object): an action provided by the agent
            Returns:
                observation (object): agent's observation of the current environment
                reward (float) : amount of reward returned after previous action
                done (bool): whether the episode has ended, in which case further
                step() calls will return undefined results
                info (dict): contains auxiliary diagnostic information (helpful for
                debugging, and sometimes learning)
                
                state==0: illegal_state
                state==1: normal_state
                sate==2: rewarded_state
        """
        # TODO double check the ordering of the movements
        # make a random movements in users in the sim object
        # (in place and without return)
        # and return new sample for users_stations
        self.users_stations = self.sim.sample_users_stations()
        self.users_distances = self.sim.users_distances

        # take action
        # take action for the services_servers half of the observation
        self.observation[:self.num_of_services][action[:self.allowed_moves]] = action[self.allowed_moves:]
        # take action for the the users_stations of the observation
        self.observation[self.num_of_services:] = self.users_stations

        # to add changes of services to the simulator
        self.sim.update_services_servers(self.observation[0:self.num_of_services])

        # find the state of the agent
        done, state = self._is_done()

        # calculate reward based-on state
        reward = self._reward(state)

        # info dicts
        num_of_cons = self._num_of_consolidated(self.observation[0:self.num_of_services])
        fraction_of_latency = self._fraction_latency(self.users_distances)

        # users_servers only for debugging pruposees
        users_servers = np.array(list(map(lambda i: self.observation[i], self.users_services)))

        info = {'state':state,
                'users_services':self.users_services,
                'services_servers':self.observation[0:self.num_of_services],
                'users_servers':users_servers,
                'actions_index': action[:self.allowed_moves],
                'actions_moves': action[self.allowed_moves:],
                'num_of_consolidated':num_of_cons,
                'fraction_of_latency':fraction_of_latency,
                'users_stations':self.users_stations,
                'users_distances':self.users_distances,
                'done': done,
                'reward': reward,
                'simulator': self.sim}
        return self.observation, reward, done, info

    def render(self, mode='human', type_of='color'):
        """
            render the info to the output
        """
        if type_of == 'color':
            plot_color(self.observation[0:self.num_of_services],
                       self.servers_mem ,self.services_mem)
        elif type_of=='black':
            plot_black(self.observation[0:self.num_of_services],
                       self.servers_mem ,self.services_mem)
        print('\n----services mem----\n')
        print('services resource usage: {}'.format(self.services_mem))
        print('servers capacity: {}'.format(self.servers_mem))
        print('users services: {}'.format(self.users_services))
        print('\n----observation----\n')
        print('services servers: {}'.format(self.observation[0:self.num_of_services]))
        print('users stations: {}'.format(self.observation[self.num_of_services:]))

    def _make_initial_observation(self, initial_state):
        """
            make initail observation
        """
        initial_observation = np.concatenate((initial_state['services_servers'],
                                              initial_state['users_stations']))
        return initial_observation

    def _is_done_latency(self, users_distances):
        """
            check if the desired fracition of users
            services are below a certain
            threshold (users_distances)
        """
        fraction = self._fraction_latency(users_distances)
        return fraction >= self.PERC_LAT

    def _is_done(self):
        """
            state==0: illegal_state
            state==1: normal_state
            sate==2: rewarded_state
        """
        done_illegal_state = self._is_done_illegal_state(self.observation[0:self.num_of_services])
        done_latency = self._is_done_latency(self.users_distances)

        done = False
        state = 0
        if done_illegal_state:
            state = 0
            done = True # if I want to end the episode in the illegal states
            return done, state
        if not done_latency:
            state = 1
        else:
            state = 2
            done = True
        return done, state

    def _reward_latency(self, users_distances):
        fraction = self._fraction_latency(users_distances)
        reward = fraction * self.penalty_latency
        return reward

    def _reward(self, state):
        """
            state==0: illegal_state
            state==1: normal_state
            sate==2: rewarded_state
        """
        if state==0:
            reward = self._reward_illegal()
        if state==1:
            reward = self._reward_not_done()
        if state==2:
            reward = self._reward_latency(self.users_distances)
        return reward

    def _make_spaces(self):
        """
            initialize spaces
            action space ([---services_servers-----][-------users_stations------])
        """
        # observatin space made Multidiscrete
        nvec_services_servers = np.ones(self.num_of_services)*self.num_of_servers
        nvec_users_stations = np.ones(self.num_of_users)*self.num_of_stations
        nvec = np.concatenate((nvec_services_servers, nvec_users_stations))
        observation_space = spaces.MultiDiscrete(nvec)

        # action spaces made with Multidiscrete
        nvec_indexes = np.ones(self.allowed_moves)*(self.num_of_services-1)
        nvec_values = np.ones(self.allowed_moves)*(self.num_of_servers-1)
        nvec = np.concatenate((nvec_indexes, nvec_values))
        action_space = spaces.MultiDiscrete(nvec)

        return observation_space, action_space


    def _fraction_latency(self, users_distances):
        """
            return the fraction of users reaching
            the services minimum latency
        """

        users_desired_latency = np.array(list(map(lambda a: self.services_desired_latency[a],
                                                  self.users_services)))
        check = users_distances < users_desired_latency
        fraction = np.count_nonzero(check==True) / self.num_of_users
        return fraction
