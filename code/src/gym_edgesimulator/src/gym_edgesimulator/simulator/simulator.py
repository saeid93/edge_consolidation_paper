import networkx as nx
import copy
from networkx.algorithms import tree
from operator import itemgetter 
import numpy as np
import random
import math
import matplotlib.pyplot as plt



class Simulator:
    def __init__(self, services_servers,
                 users_services, num_of_servers,
                 num_of_stations):
        # fixed values
        self.SPEED_LIMIT = 30
        self.width = 1
        self.length = 1
        self.server_station_con = 1 # number of connection between station and servers

        # random.seed(1) # HACK TEMP
        # np.random.seed(1)  # HACK TEMP

        # argument values
        self.services_servers = services_servers
        # self.users_stations = users_stations
        self.users_services = users_services
        self.num_of_services = len(services_servers)
        self.num_of_users = len(users_services)
        self.num_of_servers = num_of_servers
        self.num_of_stations = num_of_stations
        self.servers_idx = list(map(lambda i: (i,'server'),
                                    range(self.num_of_servers)))
        self.stations_idx = list(map(lambda i: (i,'station'),
                                     range(self.num_of_stations)))
        self.users_idx = list(map(lambda i: (i,'user'),
                                  range(self.num_of_users)))
        self.raw_network = self._make_raw_network()        
        self._add_users()


    @classmethod
    def with_network(cls, services_servers,
                     users_services, num_of_servers,
                     num_of_stations, network):
        ins = cls(services_servers, users_services,
                  num_of_servers, num_of_stations)
        ins.raw_network = copy.deepcopy(network)
        ins._add_users()
        return ins


    def _make_raw_network(self): # TODO make raw network
        """make the raw network:
           the network only with
           servers and stations not the users
           """
        network = nx.Graph()
        
        # add raw nodes
        network.add_nodes_from(self.servers_idx)
        network.add_nodes_from(self.stations_idx)
        
        # add location to nodes
        occupied_locs = set()
        for node in network.nodes:
            while True:
                loc = (round(random.uniform(0, self.length), 2),
                       round(random.uniform(0, self.width), 2))
                if loc not in occupied_locs:
                    network.nodes[node]['loc'] = loc
                    occupied_locs.add(loc)
                    break

        # adding server-server edges to the network with spanning tree
        servers_subgraph = nx.complete_graph(network.subgraph(self.servers_idx).copy())
        for edge in servers_subgraph.edges:
            weight = self._euclidean_dis(network.nodes[edge[0]]['loc'],
                                         network.nodes[edge[1]]['loc'])
            servers_subgraph.edges[edge]['weight'] = weight
        mst = tree.minimum_spanning_edges(servers_subgraph, algorithm='kruskal',
                                          weight='weight', data=True)
        for edge in mst:
            network.add_edge(edge[0], edge[1], weight=edge[2]['weight'])

        # add station-server edges by connecting stations to their closest servers
        for station in self.stations_idx:
            dis_servers = {}
            for server in self.servers_idx:
                dis = self._euclidean_dis(network.nodes[station]['loc'],
                                          network.nodes[server]['loc'])
                dis_servers[server] = dis
            res = dict(sorted(dis_servers.items(),
                              key = itemgetter(1))[:self.server_station_con])
            for key, value in res.items():
                network.add_edge(station, key, weight=value)
        return network

    def _add_users(self):
        """
            do the initialization of the
            users and their distances in
            the network
        """
        self.network = self._make_users(copy.deepcopy(self.raw_network))
        self.users_stations = np.zeros(self.num_of_users, dtype=int)
        self._update_users_stations()
        self.users_distances = np.zeros(self.num_of_users)
        self._update_users_distances()

    def _make_users(self, network):
        network.add_nodes_from(self.users_idx)
        # add location to nodes
        occupied_loc = set(nx.get_node_attributes(network,'loc').values())
        for node in self.users_idx:
            while True:
                loc = (round(random.uniform(0, self.length), 2),
                       round(random.uniform(0, self.width), 2))
                if loc not in occupied_loc:
                    network.nodes[node]['loc'] = loc
                    occupied_loc.add(loc)
                    break
        return network


    # TODO check from this point on
    # ----------- updating and sampling funcitons -----------


    def update_users_services(self, users_services):
        """
            apply changes to the services_servers
            to the simulator
        """
        self.users_services = users_services
        self._update_users_distances()


    def update_services_servers(self, services_servers):
        """
            apply changes to the services_servers
            to the simulator
        """
        self.services_servers = services_servers
        self._update_users_distances()


    def sample_users_stations(self):
        """
            sample the next users_station array
            for the simulation
            and run the three innter functions in order
            and return the new users' stations:
            1. users_move
            2. update_users_stations
            3. update_users_distances
        """
        self._users_move()
        self._update_users_stations()
        self._update_users_distances()
        return self.users_stations


    def _update_users_distances(self):
        """
            find the distances of the users to their connected
            service/server
        """
        for user in self.users_idx:
            service_idx = self.users_services[user[0]]
            server_idx = self.services_servers[service_idx]
            station_idx = self.users_stations[user[0]]
            path_length = nx.shortest_path_length(self.network,
                                                  source=(station_idx, 'station'),
                                                  target=(server_idx, 'server'),
                                                  weight='weight',
                                                  method='dijkstra')
            self.users_distances[user[0]] = path_length


    def _update_users_stations(self):
        """
            find the nearset station to each user
        """

        for user in self.users_idx:
            dis_station = {}
            for station in self.stations_idx:
                dis = self._euclidean_dis(self.network.nodes[user]['loc'],
                                          self.network.nodes[station]['loc'])
                dis_station[station] = dis

            res = min(dis_station.items(), key=itemgetter(1))[0][0]
            self.users_stations[user[0]] = res


    def _users_move(self):
        """
            Randomly moves users with `User.SPEED_LIMIT` in 2d surface
        """
        for node_id, node_type in self.network.nodes:
            if node_type == 'user':
                user_speed = self.SPEED_LIMIT
                # user_speed = np.random.randint(SPEED_LIMIT)
                dx = (random.random() - 0.5)*user_speed/100*self.width
                dy = (random.random() - 0.5)*user_speed/100*self.length
                node_location = self.network.nodes[(node_id, node_type)]['loc']
                new_x = node_location[0] + dx
                new_y = node_location[1] + dy
                if new_x>self.width:
                    new_x = self.width
                if new_x<0:
                    new_x = 0
                if new_y>self.length:
                    new_y = self.length
                if new_y<0:
                    new_y = 0
                self.network.nodes[(node_id, node_type)]['loc'] = (round(new_x, 2), round(new_y, 2))


    # ----------- utilities -----------


    def get_largets_station_server(self):
        """
            get the largest station-server distance in the entire network
        """
        edges = nx.get_edge_attributes(self.network,'weight')
        servers_stations_edges = dict(filter(lambda edge: edge[0][0][1] != edge[0][1][1], edges.items()))
        return max(servers_stations_edges.values())


    def _euclidean_dis(self, loc1, loc2):
        dis = math.sqrt(sum([(a - b) ** 2 for a, b in zip(loc1, loc2)]))
        return round(dis, 2)

    def visualize_debug(self, raw=False):
        """ 
            simple network
            visualizer
            for debugging
            raw: if True network without users
        """

        f = plt.figure(figsize = [12.9, 9.6])
        if raw:
            all_pos = nx.get_node_attributes(self.raw_network,'loc')
            labels = {k: k[0] for k in self.raw_network.nodes}
            nx.draw_networkx(self.raw_network,
                            with_labels=True, labels=labels,
                            font_size = 9,
                            nodelist=self.servers_idx,
                            node_size=100, node_shape='s',
                            pos=all_pos,
                            node_color='#eba134')
            nx.draw_networkx(self.raw_network,
                            with_labels=True, labels=labels,
                            font_size = 9,
                            nodelist=self.stations_idx,
                            node_size=100, node_shape='^',
                            pos=all_pos,
                            node_color='#e8eb34')  
            edge_labels = {k: self.raw_network.edges[k]['weight'] for k in self.raw_network.edges}
            nx.draw_networkx_edge_labels(self.raw_network, 
                                         pos=nx.get_node_attributes(self.raw_network,'loc'),
                                         edge_labels=edge_labels, font_size=6)          
        else:
            all_pos = nx.get_node_attributes(self.network,'loc')
            labels = {k: k[0] for k in self.network.nodes}
            nx.draw_networkx(self.network,
                            with_labels=True, labels=labels,
                            font_size = 9,
                            nodelist=self.servers_idx,
                            node_size=100, node_shape='s',
                            pos=all_pos,
                            node_color='#eba134')
            nx.draw_networkx(self.network,
                            with_labels=True, labels=labels,
                            font_size = 9,
                            nodelist=self.stations_idx,
                            node_size=100, node_shape='^',
                            pos=all_pos,
                            node_color='#e8eb34')
            nx.draw_networkx(self.network,
                            with_labels=True, labels=labels,
                            font_size = 9,
                            nodelist=self.users_idx,
                            node_size=100, node_shape='o',
                            pos=all_pos,
                            node_color='#34ebdf')
            edge_labels = {k: self.network.edges[k]['weight'] for k in self.network.edges}
            nx.draw_networkx_edge_labels(self.network,
                                         pos=nx.get_node_attributes(self.network,'loc'),
                                         edge_labels=edge_labels, font_size=9)
        return f

    def visualize_paper_style(self):
        """ 
            simple network
            visualizer
            like the style of the paper
            raw: if True network without users
        """
        f = plt.figure(figsize = [12.9, 9.6], frameon=False)
        ax = f.add_axes([0, 0, 1, 1])
        ax.axis('off')
        all_pos = nx.get_node_attributes(self.network,'loc')
        labels = {k: k[0] for k in self.network.nodes}
        nx.draw_networkx_nodes(self.network,
                        with_labels=True, labels=labels,
                        font_size = 9,
                        nodelist=self.servers_idx,
                        node_size=300, node_shape='s',
                        pos=all_pos,
                        node_color='red', label='server')
        nx.draw_networkx_nodes(self.network,
                        with_labels=True, labels=labels,
                        font_size = 9,
                        nodelist=self.stations_idx,
                        node_size=300, node_shape='^',
                        pos=all_pos,
                        node_color='green', label='access')
        nx.draw_networkx_nodes(self.network,
                        with_labels=True, labels=labels,
                        font_size = 9,
                        nodelist=self.users_idx,
                        node_size=300, node_shape='o',
                        pos=all_pos,
                        node_color='blue', label='user')
        plt.legend(scatterpoints = 1, fontsize=20)
        edge_labels = {k: self.network.edges[k]['weight'] for k in self.network.edges}
        nx.draw_networkx_edges(self.network,
                                    pos=nx.get_node_attributes(self.network,'loc'),
                                    edge_labels=edge_labels, font_size=9)

        return f

    # ----------- greedy action -----------
    def next_greedy_action(self, servers_mem, services_mem):
        a = 1
        temp_services_servers = copy.deepcopy(self.services_servers)
        # TODO improvement: compute stations_distances once
        def server_station_dis(server_id, station_id):
            dis = nx.shortest_path_length(self.network,
                                          source=(station_id, 'station'),
                                          target=(server_id, 'server'),
                                          weight='weight',
                                          method='dijkstra')
            return dis
        # iterate through each service one by one to find a new placement for them
        for service in range(0, self.num_of_services):
            service_connected_users = np.argwhere(self.users_services==service).flatten()
            service_connected_users_stations = self.users_stations[service_connected_users]
            service_average_servers_penalties = []
            for server in range(0, self.num_of_servers):
                station_servers_penalty = np.array([server_station_dis(server, station_id)
                                                    for station_id in service_connected_users_stations])
                service_average_servers_penalty = np.average(station_servers_penalty)
                service_average_servers_penalties.append(service_average_servers_penalty)
            # move it to the first server with lowest latency and available memory
            sorted_servers_indices = np.argsort(service_average_servers_penalties)
            for server in sorted_servers_indices:
                services_in_server = np.argwhere(temp_services_servers==server).flatten()
                used_mem = sum(services_mem[services_in_server])
                if used_mem + services_mem[service] <= servers_mem[server]:
                    temp_services_servers[service] = server
                    break
        return temp_services_servers