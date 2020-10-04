import numpy as np
from gym_edgesimulator import Simulator


class DatasetGenerator:
    def __init__(self, num_of_services, MIN_SERVICE_MEM,
                 MAX_SERVICE_MEM, MERGE_FAC, num_of_GT,
                 PERC_LAT, latency_factor, num_of_users, num_of_stations,
                 seed):

        # consolidation variable
        if num_of_services % MERGE_FAC:
            raise RuntimeError('num_of_services is not divisible by MERGE_FAC')
        self.num_of_full_servers = int(num_of_services / MERGE_FAC)
        self.num_of_GT = num_of_GT                 # BUG the name of Ground Truth is misleading
        self.num_of_servers = self.num_of_full_servers + self.num_of_GT
        self.num_of_services = num_of_services
        self.MIN_SERVICE_MEM = MIN_SERVICE_MEM
        self.MAX_SERVICE_MEM = MAX_SERVICE_MEM
        self.MERGE_FAC = MERGE_FAC
        self.num_of_GT = num_of_GT

        # latency variable
        self.PERC_LAT = PERC_LAT
        self.latency_factor = latency_factor
        self.num_of_users = num_of_users
        self.num_of_stations = num_of_stations
        self.seed = seed

        # initialize seed to ensure reproducible resutls
        # np.random.seed(self.seed)


    def make_initial_observation(self):
        """
            combining state initilizer for consolidation
            and latency to make the real initial configuration
        """

        # generate servers and services memeory within the specified range
        self.servers_mem, self.services_mem = self._make_mems()
        # generate intial services-serves mapping
        self.services_servers = self._make_services_servers()

        # generate network observations
        (self.sim, self.latency_min, self.latency_max,
         self.services_desired_latency) = self._make_observation()
        self.users_services = self.sim.users_services
        # self.services_servers = self.sim.services_servers TODO why it is not necessary
        self.users_stations =self.sim.users_stations
        self.users_distances = self.sim.users_distances

        initial_state = {'servers_mem':self.servers_mem,
                         'services_mem':self.services_mem,
                         'services_servers':self.services_servers,
                         'users_stations':self.users_stations,
                         'users_distances':self.users_distances,
                         'users_services':self.users_services,
                         'services_desired_latency':self.services_desired_latency,
                         'PERC_LAT':self.PERC_LAT,
                         'simulation':self.sim,
                         'latency_min':self.latency_min,
                         'latency_max':self.latency_max,
                         'num_of_GT': self.num_of_GT,
                         'MERGE_FAC': self.MERGE_FAC}


        # number of idle services
        self.num_of_idle_services = self.num_of_services - len(set(self.users_services))

        info_str =  "\n".join(("-----Seed-----\n",
                    "type_of_dataset consolidation_and_latency",
                    f"seed: {self.seed}",
                    "\n-----Servers_and_Services-----\n",
                    "Inputs",
                    f"num_of_servers {self.num_of_servers}",
                    f"num_of_services {self.num_of_services}",
                    f"num_of_stations {self.num_of_stations}",
                    f"num_of_users {self.num_of_users}",
                    f"servers_mem_range ({self.servers_mem.min()}, {self.servers_mem.max()})",
                    f"services_mem_range ({self.services_mem.min()}, {self.services_mem.max()})",
                    "Outputs",
                    f"servers_mem {self.servers_mem}",
                    f"services_mem {self.services_mem}",
                    f"services_servers {self.services_servers}",
                    f"num_of_idle_services {self.num_of_idle_services}",
                    "\n-----Consolidation_objectives-----\n",
                    f"consolidation_ground_truth {self.num_of_GT}",
                    "\n-----Latency_objectives-----\n",
                    f"services_latency_range ({self.latency_min}, {self.latency_max})",
                    f"services_desired_latency {self.services_desired_latency}",
                    f"latency_factor {self.latency_factor}",
                    f"PERC_LAT {self.PERC_LAT}",
                    "\n-----Initial_States-----\n",
                    f"users_stations {self.users_stations}",
                    f"users_distances {self.users_distances}",
                    f"users_services {self.users_services}"))
        
        return initial_state, info_str


    def _make_mems(self):
        """
            make random initialization states until we reach a
            legal servers_mem and services_mem
        """

        # make services memories in the given range
        services_mem = np.random.randint(low=self.MIN_SERVICE_MEM,
                                         high=self.MAX_SERVICE_MEM,
                                         size=self.num_of_services)

        # sum each {num of merge factor} of the services to build servers
        servers_full_mem = np.reshape(services_mem, (self.num_of_full_servers,-1)).sum(axis=1)
        # sample empty servers sizes from the generated servers memories
        servers_empty_mem = np.random.choice(servers_full_mem, self.num_of_GT)
        # concatenate empty servers to full servers
        servers_mem = np.concatenate((servers_full_mem, servers_empty_mem))
        return servers_mem, services_mem

    def _make_services_servers(self):

        """
            generate the intitial state for services-servers
            that is also not a terminal state (not consolidated)
        """
        count = 200

        while True:
            # place services in the full servers with empty servers at the end
            services_servers = np.repeat(np.arange(self.num_of_full_servers),
                                                   self.MERGE_FAC)
            # first deremine the number of services to be moved
            # in the interval of numober of empty servers and an estimation of making them full
            num_of_moves = np.random.randint(low=self.num_of_GT,
                                             high=self.MERGE_FAC*self.num_of_GT)
            # choose the indices of services to be moved proportial to the number of moves
            move_indexes = np.random.randint(low=0, high=self.num_of_services,
                                             size=num_of_moves)
            # move them to one of the empty servers
            move_values = np.random.randint(low=self.num_of_full_servers,
                                            high=self.num_of_servers,
                                            size=num_of_moves)
            # make the movements of the services
            services_servers[move_indexes]=move_values
            if (self._is_legal_state(services_servers) and
                not self._num_of_consolidated(services_servers)):
                return services_servers
            count -= 1
            if count == 0:
                raise RuntimeError("tried 20 times to find a random suitable\n\
                                    service placement,\n\
                                    seems impossible!\n")


    def _is_legal_allocation(self, servers_mem, services_mem):
        """
            check if the total considered server's memory for servers
            is enough to allocate services
        """
        return np.sum(servers_mem) >= np.sum(services_mem)


    def _is_legal_state(self, observation):
        """
            check for if the current memory allocation is feasible
        """
        servers_used_mem = np.zeros(len(self.servers_mem))
        for i, _ in enumerate(servers_used_mem):
            servers_used_mem[i] = np.sum(self.services_mem[observation==i])
        return np.alltrue(np.array(self.servers_mem) >= servers_used_mem)


    def _num_of_consolidated(self, observation):
        """
            return the number of consolidated servers
            for info dictionary
        """
        a = set(observation)
        b = set(np.arange(self.num_of_servers))
        return len(b-a)


    def _make_observation(self):
        """
            make random initialization states until
            we reach a legal observation for the latency
            This is only by changing the users_services
            array and change which user is connected to which server

            make users_services
                 users_stations

            legal observation == not done by latency
        """
        count = 20000
        # initial desired latency for services array
        users_services = np.random.randint(0, self.num_of_services,
                                            size=self.num_of_users)

        # users_services fixed in all the environment
        sim = Simulator(services_servers=self.services_servers,
                        users_services=users_services,
                        num_of_servers=self.num_of_servers,
                        num_of_stations=self.num_of_stations)

        # get the desired latency of services from the simulation
        latency_min = sim.get_largets_station_server()
        latency_max = self.latency_factor * latency_min
        services_desired_latency = np.around(np.random.uniform(low=latency_min,
                                                               high=latency_max,
                                                               size=self.num_of_services),2)

        users_distances = sim.users_distances
        users_stations = sim.users_stations

        while self._is_done_latency(users_services, users_distances, services_desired_latency):
            print(f"latency count: {count}")
            # new users' to service connections
            users_services = np.random.randint(0, self.num_of_services,
                                               size=self.num_of_users)
            sim.update_users_services(users_services)
            sim.sample_users_stations() 
            users_distances = sim.users_distances

            count -= 1
            if count == 0:
                raise RuntimeError("tried 20 times to find a not done latency state\n")
        
        return sim, latency_min, latency_max, services_desired_latency

    def _is_done_latency(self, users_services, users_distances, services_desired_latency):
        """
            check if the desired fracition of users
            services are below a certain
            threshold (users_distances)
        """
        users_desired_latency = np.array(list(map(lambda a: services_desired_latency[a],
                                                  users_services)))
        check = users_distances < users_desired_latency
        fraction = np.count_nonzero(check==True) / self.num_of_users
        return fraction >= self.PERC_LAT
