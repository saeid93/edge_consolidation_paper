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

class EdgeSimV2(EdgeSimV1):
    """
        The base Edge simulator environment
        users and stations are also added
        to the base implementation
        with user movements and consolidation at
        the same time
        EdgeSim-v2
        observations: concatenate(services_servers, users_stations)
        actions:      concatenate(indecies in services server to moves, servers to move)
        rewards:      fraction_of_latency*penaly + fraction_of_consolidated*penalty
    """
    # inherited _is_done_illegal_state(self, observation):
    # inherited _num_of_consolidated(self, observation):
    # inherited _is_done_latency(self, users_distances):
    # inherited _reward_illegal(self, penalty):
    # inherited _reward_not_done(self, penalty):
    # inherited _reward_consolidation(self, penalty, observation):
    # inherited _make_spaces(self):
    # inherited _fraction_latency(self, users_distances):
    # inherited _num_of_consolidated(observation):
    # inherited _plot(service_server, server_mem, used_by_service):
    # inherited _reward_latency(self, penalty, users_distances):

    def __init__(self, initial_state, allowed_moves,
                 cons_w, lat_w, SPEED_LIMIT,
                 penalty_illegal, penalty_normal, penalty_consolidated,
                 penalty_latency,
                 seed):

        super().__init__(initial_state, allowed_moves, SPEED_LIMIT,
                            penalty_illegal, penalty_normal, penalty_latency,
                            seed)
        self.cons_w = cons_w
        self.lat_w = lat_w
        self.penalty_consolidated = penalty_consolidated

    def _is_done(self):
        """
            check if it's done either by consolidation
            or latency

            state==0: illegal_state
            state==1: normal_state
            state==2: rewarded_state
            state==3: rewarded_latency
            state==4: 2 & 3
        """
        done_illegal_state = self._is_done_illegal_state(self.observation[0:self.num_of_services])
        done_consolidation = self._num_of_consolidated(self.observation[0:self.num_of_services])
        done_latency = self._is_done_latency(self.users_distances)

        done = False
        state = 0
        if done_illegal_state:
            state = 0
            done = True # if I want to end the episode in the illegal states
            return done, state
        if not done_consolidation and not done_latency:
            state = 1
        elif done_consolidation and not done_latency:
            state = 2
            done = True
        elif not done_consolidation and done_latency:
            state = 3
            done = True
        elif done_consolidation and done_latency:
            state = 4
            done = True
        return done, state

    def _reward(self, state):
        """
            state==0: illegal_state --> _reward_consolidation
            state==1: normal_state --> _reward_consolidation
            state==2: rewarded_consolidation --> _reward_consolidation
            state==3: rewarded_latency --> _reward_latency
            state==4: 2 & 3 --> _reward_consolidation &
                                _reward_latency
        """
        if state==0:
            reward = self._reward_illegal()
        elif state==1:
            reward = self._reward_not_done()
        elif state==2:
            reward = self.cons_w * self._reward_consolidation(self.observation[:self.num_of_services])
        elif state==3:
            reward = self.lat_w * self._reward_latency(self.users_distances)
        elif state==4:
            reward = (self.cons_w * self._reward_consolidation(self.observation[:self.num_of_services])+ # TODO check self.observation
                      self.lat_w * self._reward_latency(self.users_distances))
        return reward
