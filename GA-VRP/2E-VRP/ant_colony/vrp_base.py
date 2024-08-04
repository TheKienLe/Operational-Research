import numpy as np
import copy


class Node:
    """
    Represents a node in the VRPTW problem.

    Parameters
    ----------
    id : int
        The identifier for the node.
    x : float
        The x-coordinate of the node.
    y : float
        The y-coordinate of the node.
    demand : float
        The demand at the node.
    ready_time : float
        The earliest time the node can be serviced.
    due_time : float
        The latest time the node can be serviced.
    service_time : float
        The time required to service the node.

    Attributes
    ----------
    is_depot : bool
        Indicates if the node is a depot.
    """
    def __init__(self, id: int, x: float, y: float, demand: float, ready_time: float, due_time: float, service_time: float):
        self.id = id
        self.is_depot = id == 0
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time

class VrptwGraph:
    """
    Represents the VRPTW problem.

    Parameters
    ----------
    file_path : str
        The path to the file containing the node data.
    rho : float, optional
        The pheromone evaporation rate (default is 0.1).

    Attributes
    ----------
    node_num : int
        The number of nodes.
    nodes : list of Node
        The list of nodes.
    node_dist_mat : numpy.ndarray
        The matrix of distances between nodes.
    vehicle_num : int
        The number of vehicles.
    vehicle_capacity : float
        The capacity of each vehicle.
    rho : float
        The pheromone evaporation rate.
    pheromone_mat : numpy.ndarray
        The matrix of pheromone levels.
    heuristic_info_mat : numpy.ndarray
        The matrix of heuristic information.
    nnh_travel_path : list of int
        The travel path found by the nearest neighbor heuristic.
    init_pheromone_val : float
        The initial pheromone value.

    Methods
    -------
    copy(init_pheromone_val)
        Creates a copy of the graph with a new initial pheromone value.
    create_from_file(file_path)
        Creates nodes and distance matrix from a file.
    calculate_dist(node_a, node_b)
        Calculates the Euclidean distance between two nodes.
    local_update_pheromone(start_ind, end_ind)
        Performs local pheromone update.
    global_update_pheromone(best_path, best_path_distance)
        Performs global pheromone update.
    nearest_neighbor_heuristic(max_vehicle_num=None)
        Applies the nearest neighbor heuristic to find an initial solution.
    _cal_nearest_next_index(index_to_visit, current_index, current_load, current_time)
        Finds the nearest next node index that can be visited.
    """
    def __init__(self, file_path, rho=0.1):
        self.node_num, self.nodes, self.node_dist_mat,\
        self.vehicle_num, self.vehicle_capacity = self.create_from_file(file_path)
        self.rho = rho
        self.nnh_travel_path, self.init_pheromone_val, _ = self.nearest_neighbor_heuristic()
        self.init_pheromone_val = 1 / (self.init_pheromone_val * self.node_num)
        self.pheromone_mat = np.ones((self.node_num, self.node_num)) * self.init_pheromone_val
        self.heuristic_info_mat = 1 / self.node_dist_mat

    def copy(self, init_pheromone_val):
        """
        Creates a copy of the graph with a new initial pheromone value.

        Parameters
        ----------
        init_pheromone_val : float
            The new initial pheromone value.

        Returns
        -------
        VrptwGraph
            A new instance of VrptwGraph with updated pheromone values.
        """
        new_graph = copy.deepcopy(self)
        new_graph.init_pheromone_val = init_pheromone_val
        new_graph.pheromone_mat = np.ones((new_graph.node_num, new_graph.node_num)) * init_pheromone_val
        return new_graph

    def create_from_file(self, file_path):
        """
        Creates nodes and distance matrix from a file.

        Parameters
        ----------
        file_path : str
            The path to the file containing the node data.

        Returns
        -------
        int
            The number of nodes.
        list of Node
            The list of nodes.
        numpy.ndarray
            The matrix of distances between nodes.
        int
            The number of vehicles.
        float
            The capacity of each vehicle.
        """
        node_list = []
        with open(file_path, 'rt') as f:
            count = 1
            for line in f:
                if count == 2:
                    print("line", line.split())
                    vehicle_num, vehicle_capacity = line.split()
                    vehicle_num = int(vehicle_num)
                    vehicle_capacity = int(vehicle_capacity)
                elif count >= 7:
                    node_list.append(line.split())
                count += 1
        node_num = len(node_list)
        nodes = [Node(int(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5]), float(item[6])) for item in node_list]

        node_dist_mat = np.zeros((node_num, node_num))
        for i in range(node_num):
            node_a = nodes[i]
            node_dist_mat[i][i] = 1e-8
            for j in range(i+1, node_num):
                node_b = nodes[j]
                node_dist_mat[i][j] = VrptwGraph.calculate_dist(node_a, node_b)
                node_dist_mat[j][i] = node_dist_mat[i][j]

        return node_num, nodes, node_dist_mat, vehicle_num, vehicle_capacity
    
    @staticmethod
    def calculate_dist(node_a, node_b):
        return np.linalg.norm((node_a.x - node_b.x, node_a.y - node_b.y))


    def local_update_pheromone(self, start_ind, end_ind):
        self.pheromone_mat[start_ind][end_ind] = (1-self.rho) * self.pheromone_mat[start_ind][end_ind] + \
                                                  self.rho * self.init_pheromone_val
        
    def global_update_pheromone(self, best_path, best_path_distance):
        """
        更新信息素矩阵
        :return:
        """
        self.pheromone_mat = (1-self.rho) * self.pheromone_mat

        current_ind = best_path[0]
        for next_ind in best_path[1:]:
            self.pheromone_mat[current_ind][next_ind] += self.rho/best_path_distance
            current_ind = next_ind


    def nearest_neighbor_heuristic(self, max_vehicle_num=None):
        index_to_visit = list(range(1, self.node_num))
        current_index = 0
        current_load = 0
        current_time = 0
        travel_distance = 0
        travel_path = [0]

        if max_vehicle_num is None:
            max_vehicle_num = self.node_num

        while len(index_to_visit) > 0 and max_vehicle_num > 0:
            nearest_next_index = self._cal_nearest_next_index(index_to_visit, current_index, current_load, current_time)

            if nearest_next_index is None:
                travel_distance += self.node_dist_mat[current_index][0]

                current_load = 0
                current_time = 0
                travel_path.append(0)
                current_index = 0

                max_vehicle_num -= 1
            else:
                current_load += self.nodes[nearest_next_index].demand

                dist = self.node_dist_mat[current_index][nearest_next_index]
                wait_time = max(self.nodes[nearest_next_index].ready_time - current_time - dist, 0)
                service_time = self.nodes[nearest_next_index].service_time

                current_time += dist + wait_time + service_time
                index_to_visit.remove(nearest_next_index)

                travel_distance += self.node_dist_mat[current_index][nearest_next_index]
                travel_path.append(nearest_next_index)
                current_index = nearest_next_index
        # 最后要回到depot
        travel_distance += self.node_dist_mat[current_index][0]
        travel_path.append(0)

        vehicle_num = travel_path.count(0)-1
        return travel_path, travel_distance, vehicle_num


    def _cal_nearest_next_index(self, index_to_visit, current_index, current_load, current_time):
        """
        找到最近的可达的next_index
        :param index_to_visit:
        :return:
        """
        nearest_ind = None
        nearest_distance = None

        for next_index in index_to_visit:
            if current_load + self.nodes[next_index].demand > self.vehicle_capacity:
                continue

            dist = self.node_dist_mat[current_index][next_index]
            wait_time = max(self.nodes[next_index].ready_time - current_time - dist, 0)
            service_time = self.nodes[next_index].service_time
            # 检查访问某一个旅客之后，能否回到服务店
            if current_time + dist + wait_time + service_time + self.node_dist_mat[next_index][0] > self.nodes[0].due_time:
                continue

            # 不可以服务due time之外的旅客
            if current_time + dist > self.nodes[next_index].due_time:
                continue

            if nearest_distance is None or self.node_dist_mat[current_index][next_index] < nearest_distance:
                nearest_distance = self.node_dist_mat[current_index][next_index]
                nearest_ind = next_index

        return nearest_ind

class PathMessage:
    def __init__(self, path, distance):
        if path is not None:
            self.path = copy.deepcopy(path)
            self.distance = copy.deepcopy(distance)
            self.used_vehicle_num = self.path.count(0) - 1
        else:
            self.path = None
            self.distance = None
            self.used_vehicle_num = None

    def get_path_info(self):
        return self.path, self.distance, self.used_vehicle_num