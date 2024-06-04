import copy  
import math  
import random  
import numpy as np  
import pandas as pd  

class Parameters():  
    # Filename and sheet data  
    file_path = "./data/real_data.xlsx"  
    hub_data = ["Hub", "A:D", 0, 6]  
    distance_data = ["Distance", "A:EZ", 0, 155]  
    quantity_data = ["Quantity", "A:F", 0, 6] 
    Q_vehicle = 12000   
    always_open_hubs = [1, 2, 3]  
    GAMMA_1, GAMMA_2, GAMMA_3 = 1, 1, 100  
  
class Utils():  
    @staticmethod  
    def read_data(file_path, sheet_name, usecols, start_row, end_row):  
        data = pd.read_excel(file_path, sheet_name=sheet_name, usecols=usecols,   
                            skiprows=range(0, start_row-1), nrows=end_row- 
start_row)  
        return data  
    @staticmethod  
    def calculate_O(quantity_matrix, i):  
        O = 0  
        for j, v in enumerate(quantity_matrix[i]):  
            if i != j: O += v  
        return O  
    @staticmethod  
    def calculate_D(quantity_matrix, i):  
        rows = len(quantity_matrix)  
        cols = len(quantity_matrix[0])  
        D = 0  
        for r in range(rows):  
            for c in range(cols):  
                if i == c:  
                    D += quantity_matrix[r][c]  
        return D  
      
class Hub():  
    def __init__(self, _id, capacity, fixed_cost, Q_vehicle):  
        self.id = _id  
        self.capacity = capacity   
        self.fixed_cost = fixed_cost   
        self.Q_vehicle = Q_vehicle  
  
        if _id in params.always_open_hubs: self.status = 'open'  
        else: self.status = 'close'  
        self.routes = []  
  
    def total_load(self):  
        total_load = 0  
        if not self.routes: return total_load  
        if isinstance(self.routes[0], list):  
            for route in self.routes:  
                total_load += sum(node.demand for node in route)  
        else:  
            total_load += sum(node.demand for node in self.routes)  
        return total_load   
      
    def info(self):  
        print("[INFO] HUB | ID = {}".format(self.id))  
        for route in self.routes:  
            _route_ids = [node.id for node in route]  
            if _route_ids:  
                print("\t{}".format(_route_ids))  
                print("\t\tRoute load: {}".format(sum(node.demand for node in  
route)))  
  
class NonHub():  
    def __init__(self, _id, O, D):  
        self.id = _id   
        self.status = "unassigned"  
        self.demand = D  
        self.D = D   
        self.O = O  
        self.O_add_D = O + D  
  
class Problem():  
    def __init__(self, file_path: str) -> None:  
        self.file_path = file_path  
        self.hub_df = Utils.read_data(  
            file_path=self.file_path,  
            sheet_name=params.hub_data[0],  
            usecols=params.hub_data[1],  
            start_row=params.hub_data[2],  
            end_row=params.hub_data[3]  
        )  
        Q_vehicle = params.Q_vehicle  
        hubs = []  
        for idx, row in self.hub_df.iterrows():  
            hubs.append(Hub(  
                _id=idx+1, capacity=row['Capacity (m3)'], 
                fixed_cost=row['Investment Capital (VND)'], 
                Q_vehicle=Q_vehicle  
            ))  
        self.hubs = hubs  
        self.number_of_hub = len(hubs)  
  
        self.distance_df = Utils.read_data(  
            file_path=self.file_path,  
            sheet_name=params.distance_data[0],  
            usecols=params.distance_data[1],  
            start_row=params.distance_data[2],  
            end_row=params.distance_data[3]  
        )  
        self.distance_matrix = self.distance_df.values.tolist()  
  
        self.quantity_df = Utils.read_data(  
            file_path=self.file_path,  
            sheet_name=params.quantity_data[0],  
            usecols=params.quantity_data[1],  
            start_row=params.quantity_data[2],  
            end_row=params.quantity_data[3]  
        )  
        self.quantity_matrix = self.quantity_df.values.tolist()  
  
  
        self.number_of_non_hub = len(self.quantity_matrix)  
        self.non_hubs = []  
  
        for i in range(self.number_of_non_hub):  
            self.non_hubs.append(NonHub(  
                _id=i+self.number_of_hub+1,  
O=Utils.calculate_O(self.quantity_matrix, i),  
D=Utils.calculate_D(self.quantity_matrix, i)  
            ))  
  
class TwoStageHeuristic():  
    def __init__(self, problem: Problem):  
        self.problem = problem  
        self.hubs = problem.hubs.copy()  
        self.non_hubs = problem.non_hubs.copy()  
        self.opened_hubs = [hub for hub in self.hubs if hub.status=='open']  
        _ids = [j for j, hub in enumerate(self.hubs) if hub.status=='close']  
        random.seed(SEED)  
        _selected_hub = random.choice(_ids)  
        self.hubs[_selected_hub].status = 'open'  
        params.always_open_hubs.append(self.hubs[_selected_hub].id)  
  
    def select_non_hub_node(self):  
        selected_non_hubs = [non_hub for non_hub in self.non_hubs   
            if non_hub.status == "unassigned"]  
        selected_non_hubs = sorted(selected_non_hubs,  
            key= lambda x: x.O_add_D, reverse=True)  
        return selected_non_hubs[0]  
  
    def assign_to_nearest_hub(self, node):  
        node_id = node.id  
        self.opened_hubs = [hub for hub in self.hubs if hub.status=='open']  
      
        hubs_and_dist = []  
        for hub in self.opened_hubs:  
            hub_id = hub.id  
            distance = self.problem.distance_matrix[node_id-1][hub_id-1]   
            hubs_and_dist.append((hub, distance))  
  
        hubs_and_dist = sorted(hubs_and_dist, key= lambda x: x[1])  
        for selected_hub, distance in hubs_and_dist:  
            if selected_hub.total_load() + node.demand <= selected_hub.capacity:   
                self.hubs[selected_hub.id-1].routes.append(node)  
                self.non_hubs[node_id-self.problem.number_of_hub-1].status \
                    ='assigned'  
                return True
        return False   
      
    def set_nearest_hub_open(self, node):  
        node_id = node.id  
        min_distance = float('inf')  
        selected_hub = None  
        close_hubs = [hub for hub in self.hubs if hub.status == 'close']  
        if not len(close_hubs):  
            return None  
        for hub in close_hubs:  
            hub_id = hub.id  
            distance = self.problem.distance_matrix[node_id-1][hub_id-1]   
            if distance < min_distance:  
                min_distance, selected_hub = distance, hub   
        self.hubs[selected_hub.id-1].status = 'open'  
        self.hubs[selected_hub.id-1].routes.append(node)  
        self.non_hubs[node_id-self.problem.number_of_hub-1].status = 'assigned'  
  
    def find_nearest_node(self, node, linked_nodes):  
        i = node.id   
        node_and_dist = []  
        for non_hub in linked_nodes:  
            j = non_hub.id   
            if i!=j:  
                dist = self.problem.distance_matrix[i-1][j-1]  
                node_and_dist.append((non_hub, dist))  
        node_and_dist = sorted(node_and_dist, key=lambda x: x[1])  
        return node_and_dist[0][0]  
  
    def init(self):  
        unassigned_nodes_remained = True  
        while unassigned_nodes_remained:  
            # Chọn nonhubs chưa được gán có O+D lớn nhât  
            selected_non_hub = self.select_non_hub_node()  
            # Add cho hubs gần nhất có đủ sức chứa  
            if not self.assign_to_nearest_hub(node=selected_non_hub):  
                # Nếu không thỏa điều kiện thì mở hubs gần nhất và add vào  
                self.set_nearest_hub_open(node=selected_non_hub)  
            # Kiểm tra lại hết tất cả các nút đã được add chưa  
            unassigned_nodes_remained = len([node for node in self.non_hubs if  
node.status=='unassigned'])  
        # Tạo routes  
        for hub in self.hubs:  
            # Khởi tạo routes rỗng  
            routes = [[]]  
            if hub.status == 'open':  
                linked_nodes = hub.routes.copy()  
                while len(linked_nodes):  
                    current_node_v = hub   
                    current_pickup_capacity = current_node_v.Q_vehicle  
                    current_delivery_capacity = current_node_v.Q_vehicle  
                    while len(linked_nodes):  
                        nearest_node = self.find_nearest_node(current_node_v,  
linked_nodes)  
                        current_node_v = nearest_node  
                        current_pickup_capacity = current_pickup_capacity -\
current_node_v.O   
                        current_delivery_capacity = min(  
                            current_pickup_capacity, current_delivery_capacity -\
current_node_v.D  
                        )  
                        if current_delivery_capacity > 0 and \
current_pickup_capacity > 0:  
                            routes[-1].append(current_node_v)  
                            # Update   
                            linked_nodes = [  
                                node for node in linked_nodes if node.id !=\
current_node_v.id  
                            ]  
                        else:  
                            routes.append([])  
                            break   
            # Update routes cho hubs  
            hub.routes = routes  
        for i, hub in enumerate(self.hubs):  
            length_route = hub.capacity // hub.Q_vehicle  
            if len(hub.routes) < length_route:  
                for _ in range(length_route -len(hub.routes)):  
                    self.hubs[i].routes.append([])  
        return self.hubs  
  
class SubMethod():  
    name = None  
    weight = 0  
    time = 0   
    score = 0   
  
class SubProblem():  
    name = None  
    weight = 0  
    time = 0   
    score = 0   
    destropy_methods = None  
  
class Method():  
    @staticmethod  
    def worst_usage_hub_removal(solution):  
        # Loại bỏ hubs có tỷ lệ sử dụng công suất ít nhất  
        opened_hubs = [hub for hub in solution if hub.status == 'open' and hub.id  
not in params.always_open_hubs]  
        new_solution = [copy.deepcopy(hub) for hub in solution]  
        if len(opened_hubs) == 0:  
            return solution, []  
              
        hub_and_efficiency = [  
            (index, hub, hub.capacity - hub.total_load())  
            for index, hub in enumerate(opened_hubs)  
        ]  
        hub_and_efficiency = sorted(hub_and_efficiency, key=lambda x: x[-1],  
reverse=True)  
  
        selected_hub_id = hub_and_efficiency[0][1].id  
        selected_hub = new_solution[selected_hub_id-1]  
        selected_routes = selected_hub.routes  
        selected_nonhub_nodes = []  
        for route in selected_routes:  
            selected_nonhub_nodes.extend(route)  
          
        selected_hub.status = 'close'  
        selected_hub.routes = [[]]  
  
        new_solution[selected_hub_id-1] = selected_hub  
        return new_solution, selected_nonhub_nodes  
      
    @staticmethod  
    def random_hub_opening(solution):  
        # Chọn mở một hubs đang đóng và random xóa một số node  
        new_solution = [copy.deepcopy(hub) for hub in solution]  
        closed_hub = [hub for hub in new_solution if hub.status == 'close']  
  
        selected_nonhub_nodes = []  
        for i, hub in enumerate(new_solution):  
            if hub.status == "open":  
                try:  
                    random.seed(SEED)  
                    j = random.choice([_ for _ in range(len(hub.routes))])  
                    random.seed(SEED)  
                    k = random.choice([_ for _ in range(len(hub.routes[j]))])  
                except:  
                    continue  
                selected_nonhub_nodes.append(new_solution[i].routes[j][k])    
  
        if len(closed_hub):  
            random.seed(SEED)  
            selected_hub_id = random.choice(closed_hub).id  
            new_solution[selected_hub_id-1].status = "open"  
  
        return new_solution, selected_nonhub_nodes  
          
    @staticmethod  
    def random_allocation_change(solution):  
        # Random xóa một số node  
        new_solution = [copy.deepcopy(hub) for hub in solution]  
        selected_nonhub_nodes = []  
        for i, hub in enumerate(new_solution):  
            if hub.status == "open":  
                try:  
                    random.seed(SEED)  
                    j = random.choice([_ for _ in range(len(hub.routes))])  
                    random.seed(SEED)                      
                    k = random.choice([_ for _ in range(len(hub.routes[j]))])  
                except:  
                    continue  
                selected_nonhub_nodes.append(new_solution[i].routes[j][k])    
                new_solution[i].routes[j].pop(k)  
        return new_solution, selected_nonhub_nodes  
      
    @staticmethod  
    def worst_allocation_removal(solution, distance_matrix):  
        # Phá bỏ subnode dựa vào khoảng cách đến hub  
        new_solution = [copy.deepcopy(hub) for hub in solution]  
        selected_nonhub_nodes = []  
  
        for i, hub in enumerate(new_solution):             
            if hub.status == 'open':  
                node_and_dist = []   
                for j, route in enumerate(hub.routes):  
                    for k, node in enumerate(route):  
                        distance = distance_matrix[hub.id-1][node.id-1]  
                        node_and_dist.append([(j, k), distance])  
                  
                node_and_dist = sorted(node_and_dist, key=lambda x: x[-1],  
reverse=True)  
                if not len(node_and_dist):  
                    continue  
  
                selected_node_info = node_and_dist[0][0]  
                j, k = selected_node_info  
                selected_nonhub_nodes.append(new_solution[i].routes[j][k])  
                new_solution[i].routes[j].pop(k)  
                  
        return new_solution, selected_nonhub_nodes  
      
    @staticmethod  
    def worst_cost_removal(solution, distance_matrix):  
        # Phá bỏ subnode dựa vào tổng cost khoảng cách của route  
        new_solution = [copy.deepcopy(hub) for hub in solution]  
        selected_nonhub_nodes = []  
  
        for i, hub in enumerate(new_solution):             
            if hub.status == 'open':  
                node_and_dist = []   
                for j, route in enumerate(hub.routes):  
                    for k, node in enumerate(route):  
                        distance_cost = 0  
                        route_ids = [n.id for n in route if n.id != node.id]  
                        route_id_merge = [hub.id] + route_ids + [hub.id]  
                        for z in range(len(route_id_merge[:-1])):  
                            a = route_id_merge[z]  
                            b = route_id_merge[z+1]  
                            distance_cost += distance_matrix[a-1][b-1]        
                        node_and_dist.append([(j, k), distance_cost])  
                  
                node_and_dist = sorted(node_and_dist, key=lambda x: x[-1],  
reverse=True)  
                if not len(node_and_dist):  
                    continue  
  
                selected_node_info = node_and_dist[0][0]  
                j, k = selected_node_info  
                selected_nonhub_nodes.append(new_solution[i].routes[j][k])  
                new_solution[i].routes[j].pop(k)  
                  
        return new_solution, selected_nonhub_nodes                
  
    @staticmethod   
    def shaw_removal(solution, distance_matrix, quantity_matrix):  
        # GAMMA_1, GAMMA_2, GAMMA_3 = 1, 1, 100  
        new_solution = [copy.deepcopy(hub) for hub in solution]  
        route_info = []  
        for i, hub in enumerate(new_solution):  
            if hub.status == "open":  
                for j, route in enumerate(hub.routes):  
                    for k, node in enumerate(route):  
                        route_info.append((i, j, k))  
        if not route_info:  
            return new_solution, []  
  
        relatedness = []  
        for node_a in route_info:  
            for node_b in route_info:  
                i1, j1, k1 = node_a  
                i2, j2, k2 = node_b  
                node_a_id = new_solution[i1].routes[j1][k1].id   
                node_b_id = new_solution[i2].routes[j2][k2].id  
                lij = -1 if (i1==i2) and (j1==j2) else 1  
                cij = distance_matrix[node_a_id-1][node_b_id-1]  
                dij = quantity_matrix[node_a_id-len(quantity_matrix)-1][node_b_id- 
len(quantity_matrix)]  
  
                rij = params.GAMMA_1*dij + params.GAMMA_2*cij + params.GAMMA_3*lij  
  
                relatedness.append(  
                    [(i1, j1, k1), (i2, j2, k2), rij]  
                )  
  
        relatedness = sorted(relatedness, key=lambda x: x[-1], reverse=True)  
        selected = relatedness[0]  
        i, j, k = selected[0]  
        selected_nonhub_nodes = [new_solution[i].routes[j][k]]  
        new_solution[i].routes[j].pop(k)  
        return new_solution, selected_nonhub_nodes  
  
    @staticmethod   
    def random_removal(solution):  
        current_solution = [copy.deepcopy(hub) for hub in solution]  
        nonhub_nodes = []  
        for i, hub in enumerate(current_solution):  
            for j, route in enumerate(hub.routes):  
                for k, node in enumerate(route):  
                    nonhub_nodes.append((i,j,k))  
          
        random.seed(SEED)  
        q = random.choice([i+1 for i in range(len(nonhub_nodes)//2)])  
        random.seed(SEED)  
        selected_nodes = random.sample(nonhub_nodes, q)  
  
        selected_nonhubs = []  
        for node in selected_nodes:  
            i, j, k = node   
            try:  
                selected_nonhubs.append(current_solution[i].routes[j][k])  
                current_solution[i].routes[j].pop(k)  
            except:  
                continue  
          
        return current_solution, selected_nonhubs  
  
    @staticmethod  
    def greedy_insertion(hubs, selected_nonhub_nodes, problem):  
        current_solution = [copy.deepcopy(hub) for hub in hubs]  
        for non_hub in selected_nonhub_nodes:  
            case_info = []  
            for i, hub in enumerate(current_solution):  
                if hub.status == 'open':  
                    for j, route in enumerate(hub.routes):  
                        for k, node in enumerate(route):  
                            case_info.append([i, j, k])  
            index_and_cost = []  
            print(index_and_cost)
            for case in case_info:  
                i, j, k = case   
                new_solution = [copy.deepcopy(hub) for hub in current_solution]  
                new_solution[i].routes[j].insert(k, non_hub)  
                cost = CostFunction.caculate_cost(new_solution, problem)   
                index_and_cost.append([(i,j,k), cost])  
                print(index_and_cost)
            index_and_cost = sorted(index_and_cost, key= lambda x: x[-1])  
            _solution_id = index_and_cost[0]  
            i, j, k = _solution_id[0]  
            current_solution[i].routes[j].insert(k, non_hub)  
        return current_solution  
  
    @staticmethod  
    def regret_insertion(hubs, nonhubs, problem):  
        current_solution = [copy.deepcopy(hub) for hub in hubs]  
        selected_nonhub_nodes = [copy.deepcopy(nonhub) for nonhub in nonhubs]  
        while len(selected_nonhub_nodes):  
            regreted_cost = []  
            for non_hub_id, non_hub in enumerate(selected_nonhub_nodes):  
                case_info = []  
                for i, hub in enumerate(current_solution):  
                    if hub.status == 'open':  
                        for j, route in enumerate(hub.routes):  
                            for k, node in enumerate(route):  
                                case_info.append([i, j, k])  
                index_and_cost = []  
                for case in case_info:  
                    i, j, k = case   
                    new_solution = [copy.deepcopy(hub) for hub in  
current_solution]  
                    new_solution[i].routes[j].insert(k, non_hub)  
                    cost = CostFunction.caculate_cost(new_solution, problem)   
                    index_and_cost.append([(i,j,k), cost])  
  
                index_and_cost = sorted(index_and_cost, key= lambda x: x[-1])  
                regreted = index_and_cost[1][1] - index_and_cost[0][1]  
                regreted_cost.append(  
                    [non_hub_id, index_and_cost[0][0], regreted]  
                )  
              
            regreted_cost = sorted(regreted_cost, key=lambda x: x[-1])  
            _selected_nonhub_id, _solution_id, _ = regreted_cost[0]  
            i, j, k = _solution_id  
            current_solution[i].routes[j].insert(k,  
selected_nonhub_nodes[_selected_nonhub_id])  
            selected_nonhub_nodes.pop(_selected_nonhub_id)  
        return current_solution  
  
    @staticmethod  
    def duplication_operator(solution):  
        current_solution = [copy.deepcopy(hub) for hub in solution]  
        nonhub_nodes = []  
        for i, hub in enumerate(current_solution):  
            for j, route in enumerate(hub.routes):  
                for k, node in enumerate(route):  
                    nonhub_nodes.append((i,j,k))  
  
        random.seed(SEED)  
        node = random.choice(nonhub_nodes)  
        i, j, k = node  
        while True:  
            i2, j2, k2 = random.choice(nonhub_nodes)  
            if sum([i, j, k]) != sum([i2, j2, k2]):  
                break   
        current_solution[i2].routes[j2][k2].demand =\
current_solution[i2].routes[j2][k2].demand / 2  
        current_solution[i].routes[j].insert(k,  
current_solution[i2].routes[j2][k2])  
  
        return current_solution, []  
      
    @staticmethod  
    def deduplication_operator(solution):  
        current_solution = [copy.deepcopy(hub) for hub in solution]  
        nonhub_nodes = []  
        for i, hub in enumerate(current_solution):  
            for j, route in enumerate(hub.routes):  
                for k, node in enumerate(route):  
                    nonhub_nodes.append((i,j,k))  
  
        selected_nonhubs = []  
        p = 1  
        step = 1  
        while p > 0.3:  
            random.seed(SEED)  
            node = random.choice(nonhub_nodes)  
            i, j, k = node  
            current_solution[i].routes[j][k].demand =\
current_solution[i].routes[j][k].demand/2  
            selected_nonhubs.append(current_solution[i].routes[j][k])  
            random.seed(SEED+step)  
            step += 1  
            p = random.random()  
  
        return current_solution, selected_nonhubs  
    @staticmethod  
    def random_hub_removal(solution):  
        current_solution = [copy.deepcopy(hub) for hub in solution]  
        random.seed(SEED)  
        selected_hub_id = [i for i in range(len(current_solution))]  
        selected_hub_id = [_id for _id in selected_hub_id if _id+1 not in  
params.always_open_hubs]  
        selected_hub_id = random.choice(selected_hub_id)  
  
        selected_node_ids = []  
        for route in  current_solution[selected_hub_id].routes:   
            selected_node_ids.extend(route)  
          
        current_solution[selected_hub_id].routes = [[]]  
        current_solution[selected_hub_id].status = 'close'  
  
        return current_solution, selected_node_ids  
      
    @staticmethod  
    def random_route_removal(solution):  
        current_solution = [copy.deepcopy(hub) for hub in solution]  
        route_info = []  
        for i, hub in enumerate(current_solution):  
            if i+1 in params.always_open_hubs:  
                continue  
            for j, route in enumerate(hub.routes):  
                route_info.append((i, j))  
  
        random.seed(SEED)  
        route = random.choice(route_info)  
        i, j = route  
        selected_nodes = current_solution[i].routes[j]  
        current_solution[i].routes.pop(j)  
        if len(current_solution[i].routes) == 0:  
            current_solution[i].routes = [[]]  
        return current_solution, selected_nodes  
      
class CostFunction():  
    @staticmethod   
    def caculate_cost(solution, problem, final=False):  
        total = 0  
        for hub in solution:  
            total += CostFunction.caculate_cost_hub(hub, problem, final)  
        return total   
       
    @staticmethod  
    def caculate_cost_hub(hub, problem, final=False):  
        ALPHA, MEW, PHI = 0.004, 10000, 100  
        distance_matrix = problem.distance_matrix  
        quantity_matrix = problem.quantity_matrix  
        # Fixed cost  
        if final:  
            fixed_cost = 0  
            for route in hub.routes:  
                if len(route):  
                    fixed_cost += hub.fixed_cost   
                    break  
        else:  
            fixed_cost = hub.fixed_cost  
        # Distance_cost   
        node_visited = []  
        distance_cost = 0   
        for route in hub.routes:  
            route_ids = [node.id for node in route]  
            route_merge_ids = [hub.id] + route_ids + [hub.id]  
            for k in range(len(route_merge_ids[:-1])):  
                i = route_merge_ids[k]  
                j = route_merge_ids[k+1]  
                if j in node_visited:  
                    continue  
                else:   
                    node_visited.append(j)  
                distance_cost += distance_matrix[i-1][j-1]  
        distance_cost = ALPHA*distance_cost  
        # total load   
        total_demand_load = 0  
        for route in hub.routes:  
            for node in route:  
                total_demand_load += node.demand  
            demand_remain = hub.capacity - total_demand_load  
        penalty_cost = 0  
        if demand_remain < 0:  
            penalty_cost += MEW*abs(demand_remain)  
  
        node_visited = []  
        for route in hub.routes:  
            total_quantity_load = 0  
            route_id = [node.id for node in route]  
            for k in range(len(route_id[:-1])):  
                i = route_id[k]  
                j = route_id[k+1]  
                if j in node_visited:  
                    continue   
                else:  
                    node_visited.append(j)  
                total_quantity_load += quantity_matrix[i-len(quantity_matrix)- 1][j-len(quantity_matrix)-1]  
            quantity_remain = hub.Q_vehicle - total_quantity_load  
            if quantity_remain < 0:  
                penalty_cost += PHI*abs(quantity_remain)  
  
        total_cost = fixed_cost + distance_cost + penalty_cost  
        return total_cost  
  
class ALNDS:  
    def __init__(self, initial_solution, max_iter=100, problem=None):  
        self.current_solution = [copy.deepcopy(hub) for hub in initial_solution]   
        self.best_solution = [copy.deepcopy(hub) for hub in initial_solution]  
        self.max_iter = max_iter  
        self.problem = problem  
        self.T = 0.2  
  
        # Khởi tạo biến lưu trữ các tham số của subproblems  
        sub_problem_names = ["hub_location", "nonhub_allocation",  
"vehicle_routing"]  
        sub_problems = dict()  
        for problem in sub_problem_names:  
            sub_problems[problem] = SubProblem()  
            sub_problems[problem].name = problem  
            if problem == "hub_location":  
                destropy_method_names = ["WUHR", "RHO", "RHR"]  
            if problem == "nonhub_allocation":  
                destropy_method_names = ["RAC", "WAR", "DO", "DDO"]  
            if problem == "vehicle_routing":  
                destropy_method_names = ["WCR", "SR", "RR", "RRR"]  
              
            # Khởi tạo biến lưu trữ các tham số cho các problem tương ứng với từng subproblem  
            destropy_methods = dict()  
            for method in destropy_method_names:  
                destropy_methods[method] = SubMethod()  
                destropy_methods[method].name = method  
            sub_problems[problem].destropy_methods = destropy_methods  
  
        self.sub_problems = sub_problems   
  
        # Khởi tạo biến lưu trữ các tham số cho các thuật toán của repair  
        repair_method_names = ["GI", "RI"]  
        repair_methods = dict()  
        for method in repair_method_names:  
            repair_methods[method] = SubMethod()  
            repair_methods[method].name = method   
        self.repair_methods = repair_methods  
  
    def select_with_probs(self, sublist):  
        _weights = [sublist[name].weight for name in sublist]  
        probs = []  
        if all(w==0 for w in _weights):  
            inverted_weights = [1/len(sublist)]*len(sublist)  
        else:  
            probs = [w/sum(_weights) for w in _weights]  
            inverted_weights = [1/(w+0.00001) for w in probs]  
        random.seed(SEED)  
          
        keys = list(sublist.keys())  
        selected = random.choices(keys, inverted_weights)[0]  
        return selected   
  
    def select_subproblem(self) -> str:  
        return self.select_with_probs(sublist=self.sub_problems)  
    def select_destroymethod(self, subproblem) -> str:  
        return  self.select_with_probs(sublist=self.sub_problems[subproblem].destropy_methods)  
    def select_repairmethod(self) -> str:  
        return self.select_with_probs(sublist=self.repair_methods)  
      
    def destroy(self, destroymethod):  
        try:  
            if destroymethod == "RHR":  
                result = Method.random_hub_removal(solution=self.current_solution)  
            if destroymethod == "WUHR":  
                result =  \
Method.worst_usage_hub_removal(solution=self.current_solution)  
            if destroymethod == "RHO":  
                result = Method.random_hub_opening(solution=self.current_solution)   
            if destroymethod == "RAC":  
                result =  \
Method.random_allocation_change(solution=self.current_solution)  
            if destroymethod == "WAR":  
                result =  \
Method.worst_allocation_removal(solution=self.current_solution,  
distance_matrix=self.problem.distance_matrix)  
            if destroymethod == "DO":  
                result =  \
Method.duplication_operator(solution=self.current_solution)   
            if destroymethod == "DDO":  
                result =  \
Method.deduplication_operator(solution=self.current_solution)  
            if destroymethod == "WCR":  
                result = Method.worst_cost_removal(solution=self.current_solution,  
distance_matrix=self.problem.distance_matrix)  
            if destroymethod == "SR":  
                result = Method.shaw_removal(solution=self.current_solution,  
distance_matrix=self.problem.distance_matrix,  
quantity_matrix=self.problem.quantity_matrix)  
            if destroymethod == "RR":  
                result = Method.random_removal(solution=self.current_solution)   
            if destroymethod == "RRR":  
                result =  \
Method.random_route_removal(solution=self.current_solution)  
        except Exception as err:  
            return None   
        return result   
    
    def repair(self, repairmethod, hubs, selected_nonhub_nodes):  
        if repairmethod == "GI":  
            return Method.greedy_insertion(hubs, selected_nonhub_nodes,  
self.problem)  
        if repairmethod == "RI":  
            return Method.regret_insertion(hubs, selected_nonhub_nodes,  
self.problem)  
          
    def update_scores(self, subproblem, destroymethod, repairmethod):  
        self.time_subproblem[subproblem] += 1  
        self.time_repair[repairmethod] += 1  
        self.time_destroy[subproblem][destroymethod] += 1  
  
        THETA1, THETA2, THETA3 = 5, 2, 0.5  
        random.seed(SEED)  
        score = random.choice([THETA1, THETA2, THETA3, 0])  
        self.score_subproblem[subproblem] += score  
        self.score_repair[repairmethod] += score  
        self.score_destroy[subproblem][destroymethod] += score  
  
    def update_weights(self, subproblem, destroymethod, repairmethod):  
        ETA = 0.9  
        THETA1, THETA2, THETA3 = 5, 2, 0.5  
  
        self.sub_problems[subproblem].time += 1  
        self.sub_problems[subproblem].destropy_methods[destroymethod].time += 1  
        self.repair_methods[repairmethod].time += 1  
        random.seed(SEED)  
        score = random.choice([THETA1, THETA2, THETA3, 0])  
        self.sub_problems[subproblem].score += score  
        self.sub_problems[subproblem].destropy_methods[destroymethod].score +=  \
score  
        self.repair_methods[repairmethod].score += score  
  
        self.sub_problems[subproblem].weight =  \
(1 - ETA) * self.sub_problems[subproblem].weight\
    + ETA * self.sub_problems[subproblem].score \
        / self.sub_problems[subproblem].time  
  
        self.sub_problems[subproblem].destropy_methods[destroymethod].weight \
= (1 - ETA) * self.sub_problems[subproblem].destropy_methods[destroymethod].weight \
    + ETA *  self.sub_problems[subproblem].destropy_methods[destroymethod].score \
        / self.sub_problems[subproblem].destropy_methods[destroymethod].time
        
        self.repair_methods[repairmethod].weight \
= (1 - ETA) * self.repair_methods[repairmethod].weight \
    + ETA * self.repair_methods[repairmethod].score  \
        / self.repair_methods[repairmethod].time 
        
    def SA_mechanism(self, new_solution):  
        THETA = 0.9995  
        self.T = THETA*self.T  
  
        cost_current = CostFunction.caculate_cost(self.current_solution,  
problem=self.problem)  
        cost_new = CostFunction.caculate_cost(new_solution, problem=self.problem)  
        p = math.e**(-(cost_new-cost_current)/self.T)  
        random.seed(SEED)  
        q = random.random()  
        return q < p  
          
    def better_than_best(self, new_solution):  
        return CostFunction.caculate_cost(new_solution, problem=self.problem)\
<= CostFunction.caculate_cost(self.best_solution, problem=self.problem)  
  
    def better_than_current(self, new_solution):  
        return CostFunction.caculate_cost(new_solution, problem=self.problem)\
<= CostFunction.caculate_cost(self.current_solution, problem=self.problem)  
      
    def solve(self):  
        f_logger = open("log.txt", "w")  
        global SEED  
        iter_count = 0  
        new_solution= self.current_solution.copy()  
        best_cost = 0  
  
        # Vòng lặp thuật toán  
        while iter_count < self.max_iter:  
            # Chọn subproblem, destroy method and repair method  
            subproblem = self.select_subproblem()  
            destroymethod = self.select_destroymethod(subproblem=subproblem)  
            repairmethod = self.select_repairmethod()  
            print(f"Method: {subproblem} - {destroymethod} - {repairmethod}")  
              
            destroied_result = self.destroy(destroymethod)  
            if destroied_result:    
                f_logger.write(f"Method: {subproblem} - {destroymethod} - {repairmethod}\n")  
                destroyed_hubs, selected_nonhub_nodes = destroied_result  
                new_solution = self.repair(repairmethod=repairmethod,  
hubs=destroyed_hubs, selected_nonhub_nodes=selected_nonhub_nodes)  
                  
                if self.better_than_best(new_solution):  
                    self.best_solution = new_solution  
                    best_cost = CostFunction.caculate_cost(new_solution,  
problem=self.problem)  
                    self.current_solution = new_solution  
                else:  
                    if self.better_than_current(new_solution):  
                        self.current_solution = new_solution  
                    else:  
                        if self.SA_mechanism(new_solution):  
                            self.current_solution = new_solution  
  
                self.update_weights(subproblem, destroymethod, repairmethod)  
            for i in range(len(self.current_solution)):  
                if len(self.current_solution[i].routes[0]) == 0 and i+1 not in\
                params.always_open_hubs:  
                    self.current_solution[i].status = "close"  
            iter_count += 1  
            SEED += 1  
            print(f"[INFO] Step {iter_count}/{self.max_iter}", "Cost: ", CostFunction.caculate_cost(self.current_solution, self.problem)) 
            f_logger.write(f"[INFO] Step {iter_count}/{self.max_iter} | Cost:  {CostFunction.caculate_cost(self.current_solution, self.problem)}\n")  
          
        for i, hub in enumerate(self.best_solution):  
            for j, route in enumerate(hub.routes):  
                self.best_solution[i].routes[j] = list(set(route))  
  
        return self.best_solution, CostFunction.caculate_cost(self.best_solution,  
self.problem, True)  
  
#Result display  
def process():  
    global params  
    # Khởi tạo probelm  
    problem = Problem(file_path=params.file_path)  
    inital_solution = TwoStageHeuristic(problem=problem).init()  
    print("[INFO] Initial Solution")  
  
    alnds = ALNDS(initial_solution=inital_solution, max_iter=100, problem=problem)  
    solution, cost = alnds.solve()  
  
    # [hub.info() for hub in solution]  
    print("[INFO] Final Solution - Cost: ", cost)  
      
    # Save solution  
    with open("solution.txt", "w") as f:  
        for hub in solution:  
            if sum(len(route) for route in hub.routes) > 0:  
                hub.status = "open"  
        for hub in solution:  
            i = 0  
            f.write(f"HUB | ID: {hub.id} | Status: {hub.status}\n")  
            for _, route in enumerate(hub.routes):  
                if len(route):  
                    f.write(f"\tRoute {i+1}: {list(set([node.id for node in route]))}\n")  
                    f.write("\t\tRoute load: {}\n".format(sum(node.demand for node  
in route)))  
                    i += 1  
            f.write("----------------------------------\n")  
        f.write(f"Total Cost: {CostFunction.caculate_cost(solution, problem,  
True)}")  
  
if __name__=="__main__":  
    SEED = 99  
    params = Parameters()  
    process()  
