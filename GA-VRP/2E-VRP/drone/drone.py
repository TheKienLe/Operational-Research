from tqdm import tqdm
import pandas as pd
from drone_load_data import *
from drone_utils import *


class GA():
    def __init__(self, instance, GA_params):
        # data
        self.port, self.customer = instance['port'], instance['customer']
        self.cluster = instance['cluster']
        self.end_cap, self.mother_cap = instance['end_cap'], instance['mother_cap']
        self.end_num, self.mother_num = instance['end_num'], instance['mother_num']
        self.mother_var_cost, self.mother_fix_cost = instance['mother_var_cost'], instance['mother_fix_cost']
        self.end_var_cost, self.end_fix_cost = instance['end_var_cost'], instance['end_fix_cost']
        self.dtm = instance['dtm']
        self.port_cus_dict = {}
        self.mother_cost = 0
        # range
        self.PORT = range(list(self.port.keys())[0], 
                         list(self.port.keys())[-1]+1)
        self.CUS = range(list(self.customer.keys())[0], 
                         list(self.customer.keys())[-1]+1)
        self.END = range(0, 
                           self.end_num+1)
        self.MOTHER = range(0, 
                           self.mother_num+1)
        self.LOC = range(0,
                         list(self.customer.keys())[-1])
        self.CLS = range(1, len(self.cluster)+1)

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
        def cus_port_order_dist(self):
            """
            Creates a dictionary where each key is a customer and the value is a list of depots
            sorted by distance from the closest to the furthest.

            Returns:
            dict: A dictionary with customer indices as keys and lists of depot indices as values,
                sorted by distance.

            Example:
            >>> result = heur.cus_port_order_dist()
            >>> for customer, sorted_depots in result.items():
            >>>     print(f"Customer {customer}: Depots sorted by distance {sorted_depots}")
            Customer 3: Depots sorted by distance [2, 1, 0]
            Customer 4: Depots sorted by distance [2, 1, 0]
            Customer 5: Depots sorted by distance [2, 0, 1]
            """

            # Dictionary to store the result
            cus_port_dict = {}

            port_list = list(self.PORT)
            # Use numpy advanced indexing to find the nearest depot for each customer
            for cus in self.CUS:
                # Get distances from this depot to all customers using numpy indexing
                port_distances = self.dtm[cus, port_list]
                # Get sorted indices of the customers based on distances
                sorted_port_indices = np.argsort(port_distances)
                # Map the sorted indices back to customer IDs
                sorted_port = np.array(port_list)[sorted_port_indices].tolist()
                # Store in dictionary
                cus_port_dict[cus] = sorted_port
            return cus_port_dict

        def group_port_to_cust(self, cus_port_dict):
            """
            Groups customers to the nearest depots (ports) without exceeding mother capacity.

            Parameters
            ----------
            cus_port_dict : dict
                A dictionary with customer indices as keys and lists of depot indices as values,
                sorted by distance.

            Returns
            -------
            port_cus_dict: dict
                A dictionary where each key is a depot (port) and the value is a list of
                customers assigned to that depot.

            Examples
            --------
            >>> grouped = heur.group_port_to_cust(cus_port_dict)
            >>> for depot, customers in grouped.items():
            ...     print(f"Depot {depot}: Customers {customers}")
            Depot 0: Customers [7]
            Depot 1: Customers []
            Depot 2: Customers [3, 4, 5, 6]
            """
            port_cus_dict = { port: {"customer": [], 
                                    "demand": 0,
                                    "launch": 0}
                                for port in self.PORT }
            port_assign = {port: False for port in self.PORT}

            for cl_key, cl_val in sort_by_demand(self.cluster).items():
                for port in cl_val["distance"].keys():
                    if not port_assign[port]:
                        port_cus_dict[port]["customer"] = cl_val["customer"]
                        port_cus_dict[port]["demand"] = cl_val["demand"]
                        port_cus_dict[port]["launch"] = cl_val["launch"]
                        port_assign[port] = True
                        break
            print(port_cus_dict)
            return port_cus_dict
    
        def mother_route(self):
            mother_route = np.arange(list(self.port.keys())[0], 
                                    list(self.port.keys())[-1]+1)
            
            return route_permutation(mother_route)
        
        def end_route(self):
            end_route = {port:[] for port in self.PORT}
            for port in self.port.keys():
                launch_node = self.port_cus_dict[port]["launch"].copy()
                customer_end = self.port_cus_dict[port]["customer"].copy()
                customer_end.remove(launch_node)
                route = [launch_node]
                route.extend(route_permutation(customer_end))
                end_route[port] =  route
            return end_route
        

        cus_port_dict = cus_port_order_dist(self)
        self.port_cus_dict = group_port_to_cust(self, cus_port_dict)

        population = []
        
        while len(population) < self.pop_size:
            mother_chromo = mother_route(self)
            end_chromo = end_route(self)
            
            fitness = self.fitness(mother_chromo, end_chromo)
            indi = {"mother_route": mother_chromo,
                     "end_route": end_chromo,
                     "fitness": fitness}
            
            population.append(indi)
        population = sort_population(population)
        return population
    
    def fitness(self, mother_chromo, end_chromo):
        # Calculate mother cost 
        mother_dist, mother_used = evaluate(mother_chromo, end_chromo,
                                            self.dtm, 
                                            self.port_cus_dict, 
                                            self.mother_cap, 
                                            return_subroute=False,
                                            mother=True,
                                            air_port=None)
        
        mother_cost = mother_dist*self.mother_var_cost \
                        + mother_used*self.mother_fix_cost

        # Calculate end cost
        end_dist = 0
        end_used = 0
    
        for port in end_chromo.keys():
            if len(end_chromo[port]) > 0:
                end_dist_temp, end_used_temp = evaluate(end_chromo[port],end_chromo, 
                                        self.dtm, 
                                        self.customer, self.end_cap, 
                                        return_subroute=False, 
                                        mother=False,
                                        air_port=port)
                end_dist += end_dist_temp
                end_used += end_used_temp
        end_cost = end_dist*self.end_var_cost \
                        + end_used*self.end_fix_cost

        return mother_cost + end_cost 
    
    def finalize_route(self, indi):
        inital_mother_route = np.array(indi["mother_route"]).flatten()
        indi["mother_route"] = evaluate(indi["mother_route"], indi["end_route"],
                                            self.dtm, 
                                            self.port_cus_dict, 
                                            self.mother_cap, 
                                            return_subroute=True,
                                            mother=True,
                                            air_port=None)
        for port in indi["end_route"].keys():
            if len(indi["end_route"][port]) > 0:
                indi["end_route"][port] = evaluate(indi["end_route"][port],indi["end_route"], 
                                        self.dtm, 
                                        self.customer, self.end_cap, 
                                        return_subroute=True, 
                                        mother=False,
                                        air_port=port)
            indi["end_route"] = sort_end_by_mother(indi["end_route"], 
                                                   inital_mother_route)
        return indi
    
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
            mother_off1, mother_off2 = ordered_crossover(indi1["mother_route"], 
                                                         indi2["mother_route"],
                                                         mother=True)
            mother_off1 = mutate(mother_off1, self.mutate_prob)
            mother_off2 = mutate(mother_off2, self.mutate_prob)

            end_off1, end_off2 = {}, {}
            for port in self.PORT:
                if self.port_cus_dict[port]["customer"] != []:
                    end_off1[port], end_off2[port] = ordered_crossover(
                                                       indi1["end_route"][port], 
                                                       indi2["end_route"][port],
                                                       mother=False)
                    # end_off1[port] = mutate(end_off1[port], self.mutate_prob)
                    # end_off2[port] = mutate(end_off2[port], self.mutate_prob)


            # Calculate score + update population
            ## Calculate score
            score1 = self.fitness(mother_off1, end_off1)
            score2 = self.fitness(mother_off2, end_off2)
            
            offspring1 = {"mother_route": mother_off1,
                          "end_route": end_off1,
                          "fitness": score1}
            offspring2 = {"mother_route": mother_off2,
                          "end_route": end_off2,
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
        
        best_indi = self.finalize_route(best_indi)
                
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
    instance = load_data_excel("./C005_100cus.xlsx")
    print(instance)

    GA_params = {"pop_size": 100,\
                 "mutate_prob": 0,\
                 "generation": 5000,\
                 "tour_size": 10,
                 }
    ga = GA(instance, GA_params)
    best_indi, score_history = ga.evolve()
    print(best_indi)
