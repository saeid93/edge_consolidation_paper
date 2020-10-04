import numpy as np
from tqdm import tqdm
import copy



class DatasetSampleGenerator:
    """
        generates random valid initial
        observations for testing
        args:
            initial_state: generate initial state with fixed
                            mem and server sizes
        return:
            list_of_samples: a list of dics of {observation, sim} of
                                valid initial states
    """
    def __init__(self, initial_state):
        # extract all keys into variables
        for k, v in initial_state.items(): setattr(self,k, v)

        self.num_of_servers = self.simulation.num_of_servers
        self.num_of_services = self.simulation.num_of_services
        self.num_of_users = self.simulation.num_of_users
        self.num_of_stations = self.simulation.num_of_stations
        self.num_of_full_servers = int(self.num_of_services / self.MERGE_FAC)

    def make_observation(self, num_of_samples):
        """
            make random initial observation for
            the testing

            The diference from the random_gt_consolidation_latency_initilizer.py
            is that the network size is fiexed during the whole simulation
        """
        list_of_obs = []
        print('\ngenerating samples:')
        pbar = tqdm(total=num_of_samples)

        while num_of_samples:
            # first determine the number of services to be moved
            # in the interval of numober of empty servers and an
            # estimation of making them ful

            # TODO why the number of moves is in the interval of [self.num_of_GT, self.MERGE_FAC*self.num_of_GT)?
            num_of_moves = np.random.randint(low=self.num_of_GT,
                                             high=self.MERGE_FAC*self.num_of_GT)
            # choose the indices of services to be moved proportial to the number
            # of moves
            move_indexes = np.random.randint(low=0, high=self.num_of_services,
                                             size=num_of_moves)
            # move them to one of the empty servers
            move_values = np.random.randint(low=0,
                                            high=self.num_of_servers,
                                            size=num_of_moves)
            # make the movements of the services
            new_services_servers = copy.deepcopy(self.services_servers)
            new_services_servers[move_indexes]=move_values
            if (self._is_legal_state(new_services_servers) and
                not self._num_of_consolidated(new_services_servers)):
                new_simulation = copy.deepcopy(self.simulation)
                # move users -- services should not be changed
                # set speed to maximum to be as different as possible
                new_simulation.SPEED_LIMIT = 100
                new_simulation.update_services_servers(new_services_servers)
                new_simulation.sample_users_stations()
                users_distances = new_simulation.users_distances
                if not self._is_done_latency(users_distances=users_distances):
                    num_of_samples -= 1
                    ins = {'servers_mem': self.servers_mem,
                           'services_mem': self.services_mem,
                           'services_servers':new_services_servers,
                           'users_stations': new_simulation.users_stations,
                           'users_distances': new_simulation.users_distances,
                           'users_services': self.users_services,
                           'services_desired_latency': self.services_desired_latency,
                           'PERC_LAT': self.PERC_LAT,
                           'simulation': new_simulation,
                           'latency_min': self.latency_min,
                           'latency_max': self.latency_max,
                           'num_of_GT': self.num_of_GT,
                           'MERGE_FAC': self.MERGE_FAC}
                    list_of_obs.append(copy.deepcopy(ins))
                    pbar.update(1)
        pbar.close()
        return list_of_obs

    def _is_legal_state(self, observation):
        """
            check for if the current memory allocation is feasible
        """
        servers_used_mem = np.zeros(len(self.servers_mem))
        for i, _ in enumerate(servers_used_mem):
            servers_used_mem[i] = np.sum(self.services_mem[observation==i])
        return np.alltrue(np.array(self.servers_mem) >= servers_used_mem)

    def _is_done_latency(self, users_distances):
        """
            check if the desired fracition of users
            services are below a certain
            threshold (users_distances)
        """
        users_desired_latency = np.array(list(map(lambda a: self.services_desired_latency[a], self.users_services)))
        check = users_distances < users_desired_latency
        fraction = np.count_nonzero(check==True) / self.num_of_users
        return fraction >= self.PERC_LAT

    def _num_of_consolidated(self, observation):
        """
            return the number of consolidated servers
            for info dictionary
        """
        a = set(observation)
        b = set(np.arange(self.num_of_servers))
        intersect = b - a
        return len(intersect)