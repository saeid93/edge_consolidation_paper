import numpy as np
import random
import pickle
import os

from gym_edgesimulator.util import plot_black
from gym_edgesimulator.util import plot_color

from constants import DATASETS_BASE_PATH, MODELS_PATH


class ConsolidationGreedy:
    def __init__(self, dataset):

        # set the directories
        self.dir2load = os.path.join(DATASETS_BASE_PATH, str(dataset))

        # with open('{}/info.txt'.format(self.dir2load), 'r') as in_file:
        #     self.info_str = in_file.read()

        with open('{}/initial_state.pickle'.format(self.dir2load), 'rb') as in_pickle:
            self.initial_state = pickle.load(in_pickle)
        
        # self.initial_state[]
        self.servers_mem = self.initial_state['servers_mem']
        self.services_mem = self.initial_state['services_mem']
        self.num_of_servers = len(self.servers_mem)
        self.num_of_services = len(self.services_mem)

        # global random ordering of the server
        self.servers = [_ for _ in range(self.num_of_servers)]
        random.shuffle(self.servers)


    def next_fit(self):
        self.servers_alloc = [[] for _ in range(self.num_of_servers)]
        servers = self.servers.copy()
        server_id = servers.pop()
        for service_id, service_mem in enumerate(self.services_mem):
            while True:
                if sum(self.services_mem[self.servers_alloc[server_id]]) + service_mem <= self.servers_mem[server_id]:
                    self.servers_alloc[server_id].append(service_id)
                    break
                else:
                    server_id = servers.pop()
        return self.num_of_consolidated(self.convert_to_plot_format(self.servers_alloc))


    def first_fit(self):
        self.servers_alloc = [[] for _ in range(self.num_of_servers)]
        servers = self.servers.copy()
        poped_servers = []
        server_id = servers.pop()
        poped_servers.append(server_id)
        for service_id, service_mem in enumerate(self.services_mem):
            for server_id in poped_servers:
                if sum(self.services_mem[self.servers_alloc[server_id]]) + service_mem <= self.servers_mem[server_id]:
                    self.servers_alloc[server_id].append(service_id)
                    break
            else: # no-break
                while True:
                    server_id = servers.pop()
                    poped_servers.append(server_id)
                    if sum(self.services_mem[self.servers_alloc[server_id]]) + service_mem <= self.servers_mem[server_id]:
                        self.servers_alloc[server_id].append(service_id)
                        break
        return self.num_of_consolidated(self.convert_to_plot_format(self.servers_alloc))


    def best_fit(self):
        self.servers_alloc = [[] for _ in range(self.num_of_servers)]
        servers = self.servers.copy()
        poped_servers = []
        server_id = servers.pop()
        poped_servers.append(server_id)
        for service_id, service_mem in enumerate(self.services_mem):
            # find remaining memory in the previous servers
            remmems = []
            for server_id in poped_servers:
                remmem = self.servers_mem[server_id] - sum(self.services_mem[self.servers_alloc[server_id]])
                remmems.append(remmem)
            # sort and allocate it in the smallest possible place
            remmems_indexes = [x for _,x in sorted(zip(remmems, poped_servers))]
            for index, server_id in enumerate(remmems_indexes):
                if service_mem <= remmems[poped_servers.index(server_id)]:
                    self.servers_alloc[server_id].append(service_id)
                    break
            else: # no-break
                server_id = servers.pop()
                poped_servers.append(server_id)
                self.servers_alloc[server_id].append(service_id)
        
        return self.num_of_consolidated(self.convert_to_plot_format(self.servers_alloc))
    
    # TODO merege all of its usecases if possible
    def num_of_consolidated(self, observation):
        """
            return the number of consolidated servers
            for info dictionary
        """
        a = set(observation)
        b = set(np.arange(self.num_of_servers))
        intersect = b - a
        return len(intersect)


    def convert_to_plot_format(self, servers_alloc):
        services_servers = np.zeros(self.num_of_services)
        for i, alloc in enumerate(servers_alloc):
            services_servers[alloc] = i
        return services_servers


a = ConsolidationGreedy(dataset=0)


print(f"-----next_fit-----")
print(f"num_of_consolidated: {a.next_fit()}")
print(f"servers' orders: {a.servers}")
plot_black(a.convert_to_plot_format(a.servers_alloc), a.servers_mem, a.services_mem)

print(f"\n-----first_fit-----")
print(f"num_of_consolidated: {a.first_fit()}")
print(f"servers' orders: {a.servers}")
plot_black(a.convert_to_plot_format(a.servers_alloc), a.servers_mem, a.services_mem)

print(f"\n-----best_fit-----")
print(f"num_of_consolidated: {a.best_fit()}")
print(f"servers' orders: {a.servers}")
plot_black(a.convert_to_plot_format(a.servers_alloc), a.servers_mem, a.services_mem)


# # TEST
# next_fit = []
# first_fit = []
# best_fit = []
# for i  in range(1000):
#     print(i)
#     a = ConsolidationGreedy(2)
#     next_fit.append(a.next_fit())
#     first_fit.append(a.first_fit())
#     best_fit.append(a.best_fit())

# print(sum(next_fit)/len(next_fit))
# print(sum(first_fit)/len(first_fit))
# print(sum(best_fit)/len(best_fit))
