  
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import random
from gym.spaces import Box, MultiDiscrete

from gym_edgesimulator.util import plot_color
from gym_edgesimulator.util import plot_black


class EdgeSimV0(gym.Env):
    """
        The base Edge simulator environment
        without the users and stations only
        the servers and services and consolidation objective.
        EdgeSim-v0
        observations: services_servers
        actions:      concatenate(indecies in services server to moves, servers to move)
        rewards:      fraction_of_latency*penaly + fraction_of_consolidated*penalty
    """
    def __init__(self, initial_state, allowed_moves, penalty_illegal,
                 penalty_normal, penalty_consolidated, seed):

        # initialize seed to ensure reproducible resutls
        self.seed = seed
        # np.random.seed(seed)

        # find the servers' and services' numbers
        # from the input arrays 
        self.num_of_servers = len(initial_state['servers_mem'])
        self.num_of_services = len(initial_state['services_mem'])
        self.allowed_moves = allowed_moves

        # initialize servers and services mem if they
        self.servers_mem = initial_state['servers_mem']
        self.services_mem = initial_state['services_mem']

        # initialize action_space and observation space 
        self.observation_space, self.action_space = self._make_spaces()
        self.num_of_GT = initial_state['num_of_GT']

        # make random legal initial observation
        self.initial_observation = self._make_initial_observation(initial_state)
        self.observation = self.reset()
        # self.observation = self.initial_observation

        # penalties
        self.penalty_illegal = penalty_illegal
        self.penalty_normal = penalty_normal
        self.penalty_consolidated = penalty_consolidated
        # self.counter = 0


    def reset(self):
        """
            Resets the state of the environment and returns an initial observation.
            Returns:
                observation (object): the initial observation.
            Remarks:
                each time resets to a different initial state
        """
        self.observation = self.initial_observation.copy()
        # self.observation = self.observation_space.sample()
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

        # take action
        self.observation[action[:self.allowed_moves]] = action[self.allowed_moves:]

        done, state = self._is_done()
        reward = self._reward(state)
        # info dicts
        num_of_cons = self._num_of_consolidated(self.observation)
        info = {'state':state, 'num_of_consolidated':num_of_cons,
                'fraction_of_latency':0}
        return self.observation, reward, done, info

    def render(self, mode='human', type_of='color'):
        """
            render the info to the output
        """
        if type_of == 'color':
            plot_color(self.observation, self.servers_mem ,self.services_mem)
        elif type_of=='black':
            plot_black(self.observation, self.servers_mem ,self.services_mem)
        print('\n----services mem----\n')
        print('services resource usage: {}\n'.format(self.services_mem))
        print('servers capacity: {}\n'.format(self.servers_mem))
        print('\n----observation----\n')
        print('service placements: {}\n'.format(self.observation))

    def _make_initial_observation(self, initial_state):
        """
            make initail observation
        """
        initial_observation = initial_state['services_servers'] # TODO change initial_observation --> services_servers
        return initial_observation

    def _make_spaces(self):
        """
           initialize spaces
        """
        # observation spaces made with MultiDiscerete
        nvec = np.ones(self.num_of_services)*self.num_of_servers
        observation_space = spaces.MultiDiscrete(nvec)

        # action spaces made with Multidiscrete
        nvec_indexes = np.ones(self.allowed_moves)*(self.num_of_services-1)
        nvec_values = np.ones(self.allowed_moves)*(self.num_of_servers-1)
        nvec = np.concatenate((nvec_indexes, nvec_values))
        action_space = spaces.MultiDiscrete(nvec)

        return observation_space, action_space

    def _is_done(self):
        """
            state==0: illegal_state
            state==1: normal_state
            sate==2: rewarded_state
        """
        done_illegal_state = self._is_done_illegal_state(self.observation)
        done_consolidation = self._num_of_consolidated(self.observation)

        done = False
        state = 0
        if done_illegal_state:
            state = 0
            done = True # if I want to end the episode in the illegal states
            return done, state
        if not done_consolidation:
            state = 1
        else:
            state = 2
            done = True

        return done, state

    def _is_done_illegal_state(self, observation):
        """
            check for if the current memory allocation is feasible
        """
        servers_used_mem = np.zeros(len(self.servers_mem))
        for i, _ in enumerate(servers_used_mem):
            servers_used_mem[i] = np.sum(self.services_mem[observation==i])
        return np.alltrue(np.array(self.servers_mem) < servers_used_mem)

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
            reward = self._reward_consolidation(self.observation)
        return reward

    def _reward_illegal(self):
        return self.penalty_illegal

    def _reward_not_done(self):
        return self.penalty_normal

    def _reward_consolidation(self, observation):
        """
            compute the reward of consolidation
            multiply the number of the consolidated * penalty
        """
        num_of_consolidation = self._num_of_consolidated(observation)
        reward = (num_of_consolidation/self.num_of_GT) * self.penalty_consolidated
        return reward

    def _num_of_consolidated(self, observation):
        """
            return the number of consolidated servers
            for info dictionary
        """
        a = set(observation)
        b = set(np.arange(self.num_of_servers))
        intersect = b - a
        return len(intersect)
