import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from load_data import *
from utils import *


class GA():
    def __init__(self, instance, GA_params):
        # data
        self.satellite, self.customer = instance['sattelite'], instance['customer']
        self.motor_cap, self.truck_cap = instance['motor_cap'], instance['truck_cap']
        self.motor_num, self.truck_num = instance['motor_num'], instance['truck_num']
        self.truck_var_cost, self.truck_fix_cost = instance['truck_var_cost'], instance['truck_fix_cost']
        self.motor_var_cost, self.motor_fix_cost = instance['motor_var_cost'], instance['motor_fix_cost']
        self.dtm = instance['dtm']
        self.sat_cus_dict = {}
        self.truck_cost = 0

        # range
        self.SAT = range(list(self.satellite.keys())[0], 
                         list(self.satellite.keys())[-1])
        self.CUS = range(list(self.customer.keys())[0], 
                         list(self.customer.keys())[-1])
        self.MOTOR = range(0, 
                           self.motor_num+1)
        self.TRUCK = range(0, 
                           self.truck_num+1)
        self.LOC = range(0,
                         list(self.customer.keys())[-1])

        # evolutionary algorithm parameters
        self.generation = GA_params['generation']
        self.pop_size = GA_params['pop_size']
        self.mutate_prob = GA_params['mutate_prob']  
        self.tour_size = GA_params['tour_size']
        self.population = self.initialize_population()
        

    
    def initialize_population(self):
        """
        Generate an initial population for the genetic algorithm.

        Parameters
        ----------
        n_customers : int
            The number of customers.
        n_population : int
            The size of the population to generate.

        Returns
        -------
        population : list of lists
            A list of chromosomes, where each chromosome is a permutation of customer indices.

        Examples
        --------
        >>> initialize_population(5, 3)
        [[1, 4, 2, 5, 3], [2, 3, 5, 4, 1], [5, 3, 2, 1, 4]]
        """
        def cus_sat_order_dist(self):
            """
            Creates a dictionary where each key is a customer and the value is a list of depots
            sorted by distance from the closest to the furthest.

            Returns:
            dict: A dictionary with customer indices as keys and lists of depot indices as values,
                sorted by distance.

            Example:
            >>> result = heur.cus_sat_order_dist()
            >>> for customer, sorted_depots in result.items():
            >>>     print(f"Customer {customer}: Depots sorted by distance {sorted_depots}")
            Customer 3: Depots sorted by distance [2, 1, 0]
            Customer 4: Depots sorted by distance [2, 1, 0]
            Customer 5: Depots sorted by distance [2, 0, 1]
            """

            # Dictionary to store the result
            cus_sat_dict = {}

            sat_list = list(self.SAT)
            # Use numpy advanced indexing to find the nearest depot for each customer
            for cus in self.CUS:
                # Get distances from this depot to all customers using numpy indexing
                sat_distances = self.dtm[cus, sat_list]
                # Get sorted indices of the customers based on distances
                sorted_sat_indices = np.argsort(sat_distances)
                # Map the sorted indices back to customer IDs
                sorted_sat = np.array(sat_list)[sorted_sat_indices].tolist()
                # Store in dictionary
                cus_sat_dict[cus] = sorted_sat
            
            return cus_sat_dict

        def group_sat_to_cust(self, cus_sat_dict):
            """
            Groups customers to the nearest depots (satellites) without exceeding truck capacity.

            Parameters
            ----------
            cus_sat_dict : dict
                A dictionary with customer indices as keys and lists of depot indices as values,
                sorted by distance.

            Returns
            -------
            sat_cus_dict: dict
                A dictionary where each key is a depot (satellite) and the value is a list of
                customers assigned to that depot.

            Examples
            --------
            >>> grouped = heur.group_sat_to_cust(cus_sat_dict)
            >>> for depot, customers in grouped.items():
            ...     print(f"Depot {depot}: Customers {customers}")
            Depot 0: Customers [7]
            Depot 1: Customers []
            Depot 2: Customers [3, 4, 5, 6]
            """
            sat_cus_dict = { sat: {"customer": [], 
                                    "demand": 0+self.satellite[sat]["demand"]}
                                for sat in self.SAT }

            for cus in self.CUS:
                sat_close_idx = 0
                sat_nearest = cus_sat_dict[cus][sat_close_idx]
        
                demand = sat_cus_dict[sat_nearest]["demand"] + self.customer[cus]['demand']

                if demand > self.truck_cap:
                    while demand > self.truck_cap:
                        sat_close_idx +=1
                        sat_nearest = cus_sat_dict[cus][sat_close_idx]
                        demand = sat_cus_dict[sat_nearest]["demand"]  \
                        + self.customer[cus]['demand']
                    sat_cus_dict[sat_nearest]["demand"] = demand
                    sat_cus_dict[sat_nearest]["customer"].append(cus)
                else: 
                    sat_cus_dict[sat_nearest]["demand"] = demand 
                    sat_cus_dict[sat_nearest]["customer"].append(cus)
            return sat_cus_dict
    
        def truck_route(self):
            truck_route = np.arange(list(self.satellite.keys())[0], 
                                    list(self.satellite.keys())[-1])
            
            return route_permutation(truck_route)
        
        def motor_route(self):
            motor_route = {sat : route_permutation(self.sat_cus_dict[sat]["customer"]) 
                           for sat in self.SAT}
            
            return motor_route
        

        cus_sat_dict = cus_sat_order_dist(self)
        self.sat_cus_dict = group_sat_to_cust(self, cus_sat_dict)

        population = []
        
        while len(population) < self.pop_size:
            truck_chromo = truck_route(self)
            motor_chromo = motor_route(self)
            fitness = self.fitness(truck_chromo, motor_chromo)
            indi = {"truck_route": truck_chromo,
                     "motor_route": motor_chromo,
                     "fitness": fitness}
            
            population.append(indi)
        population = sort_population(population)
        return population
    
    def fitness(self, truck_chromo, motor_chromo):
        # Calculate truck cost 
        truck_dist, truck_used = evaluate(truck_chromo, self.dtm, 
                            self.sat_cus_dict, self.truck_cap, 
                            return_subroute=False)
        
        truck_cost = truck_dist*self.truck_var_cost \
                        + truck_used*self.truck_fix_cost
        
        # Calculate motor cost
        motor_dist = 0
        motor_used = 0
    
        for sat in motor_chromo.keys():
            if len(motor_chromo[sat]) > 0:
                motor_dist_temp, motor_used_temp = evaluate(motor_chromo[sat], 
                                        self.dtm, 
                                        self.customer, self.motor_cap, 
                                        return_subroute=False)
                motor_dist += motor_dist_temp
                motor_used += motor_used_temp
        motor_cost = motor_dist*self.motor_var_cost \
                        + motor_used*self.motor_fix_cost

        return truck_cost + motor_cost 
    
    def evolve(self):
        best_indi = self.population[0]

        score_history = [best_indi["fitness"]]

        # while cur_iter <= iteration:
        for i in tqdm(range(1, self.generation)):
                        
            # Pick two chromosome
            indi1 = tournament_selection(self.population, 
                                               self.tour_size)
            indi2 = tournament_selection(self.population, 
                                               self.tour_size)

            # Crossover & mutation
            truck_off1, truck_off2 = ordered_crossover(indi1["truck_route"], 
                                                       indi2["truck_route"])
            truck_off1 = mutate(truck_off1, self.mutate_prob)
            truck_off2 = mutate(truck_off2, self.mutate_prob)

            motor_off1, motor_off2 = {}, {}
            for sat in self.SAT:
                if self.sat_cus_dict[sat]["customer"] != []:
                    motor_off1[sat], motor_off2[sat] = ordered_crossover(
                                                       indi1["motor_route"][sat], 
                                                       indi2["motor_route"][sat])
                    motor_off1[sat] = mutate(motor_off1[sat], self.mutate_prob)
                    motor_off2[sat] = mutate(motor_off2[sat], self.mutate_prob)

            # Calculate score + update population
            ## Calculate score
            score1 = self.fitness(truck_off1, motor_off1)
            score2 = self.fitness(truck_off2, motor_off2)
            
            offspring1 = {"truck_route": truck_off1,
                          "motor_route": motor_off1,
                          "fitness": score1}
            offspring2 = {"truck_route": truck_off2,
                          "motor_route": motor_off2,
                          "fitness": score2}
            
            if score1 < self.population[-1]["fitness"]:
                replace(self.population, 
                        chromo_in=offspring1, 
                        chromo_out=self.population[-1])
            self.population = sort_population(self.population)

            if score2 < self.population[-1]["fitness"]:
                replace(self.population, 
                        chromo_in=offspring2, 
                        chromo_out=self.population[-1])

            # Update new best score
            self.population = sort_population(self.population)
            best_indi = self.population[0]
            score_history.append(best_indi["fitness"])

        for route in best_indi["motor_route"].values():
            print (evaluate(route, 
                            self.dtm, 
                            self.customer, self.motor_cap, 
                            return_subroute=True))
        return best_indi, score_history
    


    

if __name__ == '__main__':

    # get input
    # initialize population
    # calculate cost
    # if terminal -> finish
    # repeat iteration
    # -> select chromosomes
    # -> mutate chromosomes
    # -> replace
    # -> calculate cost
    # args = get_parser()
    instance = load_data_benchmark("./vrp_data/2evrp_instances_unified_breunig/Set4a_Instance50-53.dat")
    instance["motor_cap"] = 1000

    print(instance)

    GA_params = {"pop_size": 100,\
                 "mutate_prob": 0.3,\
                 "generation": 5000,\
                 "tour_size": 10,
                 }
    ga = GA(instance, GA_params)
    best_score, score_history = ga.evolve()
    print(best_score)

    # population = initialize_population(n_customers, n_population)
    # prev_score, chromosome = get_chromosome(
    #     population, evaluate, distance_matrix, demand, cap_vehicle)

    # score_history = [prev_score]

    # # while cur_iter <= iteration:
    # for i in tqdm(range(1, iteration+1)):
        
    #     # Get best chromosome
    #     chromosomes = get_chromosome(
    #         population, evaluate, distance_matrix, demand, cap_vehicle, k=2)
        
    #     # Pick two chromosome
    #     chromosome1 = chromosomes[0][1]
    #     chromosome2 = chromosomes[1][1]

    #     # Crossover
    #     offspring1, offspring2 = ordered_crossover(chromosome1, chromosome2)

    #     # Mutation
    #     offspring1 = mutate(offspring1, mutate_prob)
    #     offspring2 = mutate(offspring2, mutate_prob)

    #     # Calculate score + update population
    #     ## Calculate score
    #     score1 = evaluate(offspring1, distance_matrix, demand, cap_vehicle)
    #     score2 = evaluate(offspring2, distance_matrix, demand, cap_vehicle)
    #     score, chromosome = get_chromosome(
    #         population, evaluate, distance_matrix, demand, cap_vehicle, reverse=True)

    #     if score1 < score:
    #         replace(population, chromo_in=offspring1, chromo_out=chromosome)

    #     score, chromosome = get_chromosome(
    #         population, evaluate, distance_matrix, demand, cap_vehicle, reverse=True)

    #     if score2 < score:
    #         replace(population, chromo_in=offspring2, chromo_out=chromosome)

    #     # Update new best score
    #     score, chromosome = get_chromosome(
    #         population, evaluate, distance_matrix, demand, cap_vehicle)
    #     score_history.append(score)
    #     prev_score = score

    # print("Total_distance:", score)  # 1 depart
    # subroutes = evaluate(chromosome, distance_matrix,
    #                      demand, cap_vehicle, return_subroute=True)
    # print(subroutes)
