import pandas as pd
import numpy as np
import random
import math
import os
from utils import *


class GA:
    def __init__(self, instance, GA_params) -> None:

        # Instane Parameters
        # number of hospital
        self.num_hospital = instance["num_hospital"]

        # number of vehicle
        self.num_vehicle = instance["num_vehicle"]

        # number of compartments
        self.num_compart = instance["num_compart"]

        # hospital quantity for each compartment: hospital_qty = [compart_1 compart_2]
        self.hospital_qty = instance["hospital_qty"]

        # vehicle capacity for each compartment: vehicle_cap = [compart_1, compart_2]
        self.vehicle_cap = instance["vehicle_cap"]

        # Travel distance between hospital_i and hospital_j
        self.travel_dist = instance["travel_dist"]
 

        self.bigM = 1e10

        # GA parameters
        self.pop_size = GA_params["pop_size"]  # population size
        self.gen_max = GA_params["gen_max"]  # Max number of generation
        self.mutate_prob = GA_params["mutate_prob"]  # Probability of mutation
        # percentage of route being crossovered
        self.cross_rate = GA_params["cross_rate"]
        self.tour_size = GA_params["tour_size"] # tournament size

        # Initial population {index = [chromosome, fitness_score]}
        self.population = self.initialize_population(self.pop_size)
        # print("initial population", self.population)

        # Best individual (update after finishing fitness_score function) [chromosome, fitness_score]
        self.best_indi = self.find_best_indi(self.population)

    def initialize_population(self, population_size=50):
        # population
        ind0 = np.arange(1, self.num_hospital+1)
        population = {}

        for indi in range(population_size):
            temp_genes = np.random.permutation(
                ind0)  # shuffle to create new route
            final_genes = [0]  # initialize new route
            # initialize current cap
            current_cap = np.array([0]*self.num_compart, dtype=np.float32)

            for host in temp_genes:
                # update current capacity
                current_cap += self.hospital_qty[host]

                # Check current capacity
                if (current_cap <= self.vehicle_cap).all():
                    # add new hospital to current vehicle
                    final_genes.append(host)
                else:
                    # assign to the next truck
                    final_genes.extend([0, host])
                    # update current capacity
                    current_cap = self.hospital_qty[host].copy()

            final_genes.append(0)  # return to collection center

            # add new individual to the truck
            population[indi] = [final_genes, self.fittest_score(final_genes)]

        return population

    def fittest_score(self, individual):

        total_distance = 0
        for node_index in range(len(individual) - 1):
            d1 = individual[node_index]
            d2 = individual[node_index+1]
            total_distance += self.travel_dist[(d1, d2)]  # {(i, k) : value}

        return total_distance

    def find_best_indi(self, population):

        values_arr = np.array([value[1] for value in population.values()])
        best_indi_index = np.argmin(values_arr)
        return population[best_indi_index]

    def tournament_selection(self, population, tournament_size, parent_pool_size):
        parent_pool = []
        for _ in range(parent_pool_size):
            # create tournament sample
            tournament_sample = random.choices(population, k=tournament_size)
            # find the best individual in tournament sample --> return individual [[route], fitness_score]
            best_ind_tournament = self.find_best_indi(
                lst_to_dict(tournament_sample))
            # find the index of best individual in tournament sample inside population
            population_id = [
                idx for idx in population if population[idx] == best_ind_tournament][0]
            # add population id above to parent pool
            parent_pool.append(population_id)
        return parent_pool

    # processed_individual is parent after removing cross route [[1,2], [3, 4]]
    # inserted_node is a list contain # ["I" , "J"]
    def capacity_constraint(self, processed_individual, cost, inserted_node):
        test_route = total_route(processed_individual, 0)
        cost = self.bigM
        # offspring = []
        # print("processed_individual", processed_individual)
        # print("test_route", test_route)
        for isn in inserted_node:
            # print("Inserting: ", isn)
            cp_test_route = test_route.copy()
            # print("Testing at whole route: ", cp_test_route)
            temp_off = []  # [[[I, 1,2], [3,4]] , [[1,I, 2], [3,4]]] -> insert I find min distance -> test_route
            for route in cp_test_route:
                # cp_test_route = test_route.copy()
                # print("    On route: ", route)
                # print("route", route)
                for idx in range(len(route)+1):
                    # print("        Inserting at idx ", idx)
                    cp_route = route.copy()
                    #
                    cp_test_route2 = cp_test_route.copy()
                    # [1,2] --> idx = 0 [I, 1, 2] idx = 1 [1, I, 2] idx = 2 [1, 2 I]
                    cp_route.insert(idx, isn)
                    cp_test_route2[cp_test_route.index(
                        route)] = cp_route  # [[1,2], [3, 4]]

                    # calculate capacity
                    current_cap = np.array(
                        [0]*self.num_compart, dtype=np.float32)

                    for node in cp_route:
                        # update current capacity
                        current_cap += self.hospital_qty[node]
                    # print("current_cap", current_cap)
                    # print("vehicle_cap", self.vehicle_cap)

                    if (current_cap <= self.vehicle_cap).all():
                        temp_off.append(
                            [cp_test_route2, self.fittest_score(join_lst_lst(cp_test_route2))])
                    else:
                        temp_off.append([cp_test_route2, self.bigM])
                        break
                    # print("        Temp Offspring: ", temp_off)
            # print("temp_off", temp_off)
            test_route, cost = min_lst(temp_off)
            # print("minlst", min_lst(temp_off))
            # print("test_route", test_route)
            # print("cost", cost)
            # print("        Test_route:, ", test_route)
        return join_lst_lst(test_route), cost

    # parent type [chromosome, fitness]
    def crossover(self, parent1, parent2):
        num_total_route = np.array([
            len(total_route(parent1[0], 0)), len(total_route(parent2[0], 0))])
        num_route_cross = np.floor(
            num_total_route * self.cross_rate).astype(int)  # [2 , 2]
        # print("num route and cross:", num_total_route, num_route_cross)

        parent_route_1 = total_route(parent1[0], 0)
        parent_route_2 = total_route(parent2[0], 0)

        # print("parent_route_1", parent_route_1)
        # print("parent_route_2", parent_route_2)
        
        cross_route1 = select_route_cross(
            parent_route_1, num_route_cross[0])  # [[1,2], [3,4]] -> [1,2,3,4]
        cross_route2 = select_route_cross(
            parent_route_2, num_route_cross[1])

        # print("cross_route1:", cross_route1)
        # print("cross_route2:", cross_route2)

        result = []  # the place to contain all offsprings
        # print(parent2[0], parent1[0])
        # print(cross_route1, cross_route2)

        # remove cross route inside parent
        chromosome2 = parent2[0].copy()
        for route1 in cross_route1:  # [[1,2], [3,4]]
            for route in route1:
                if route in chromosome2:
                    chromosome2.remove(route)

        chromosome1 = parent1[0].copy()
        for route2 in cross_route2:
            for route in route2:
                if route in chromosome1:
                    chromosome1.remove(route)

        print("chromosome1", chromosome1)
        print("chromosome2", chromosome2)

        result.append(self.capacity_constraint(
            chromosome1, parent1[1],join_cross_route(cross_route2)))

        result.append(self.capacity_constraint(
            chromosome2, parent2[1],join_cross_route(cross_route1)))
        return result

    # swap
    # input offspring as [offspring, fitness_score]
    def swap_two_node(self, offspring):
        # create list contain all route [0,1,2,0,3,4,0] --> [[1,2], [3,4]]
        mutation_route = total_route(offspring[0], 0)
        # only subroute with len >= 2 can be swapped
        while True:
            # select a route to swap
            selected_route = random.choice(mutation_route)
            if len(selected_route) >= 2:
                idx1, idx2 = random.sample(range(len(selected_route)), k=2)
                selected_route[idx1], selected_route[idx2] = selected_route[idx2], selected_route[idx1]
                offspring[0] = join_lst_lst(mutation_route)
                break
        if offspring[1] <= self.bigM:
            offspring[1] = self.fittest_score(offspring[0])
        return offspring

    # inversion
    # input offspring as [offspring, fitness_score]
    def inversion_node(self, offspring):
        # create list contain all route [0,1,2,0,3,4,0] --> [[1,2], [3,4]]
        mutation_route = total_route(offspring[0], 0)
        # only subroute with len >= 2 can be swapped
        while True:
            # select a route to swap
            selected_route = random.choice(mutation_route)
            if len(selected_route) >= 2:
                selected_route.reverse()
                offspring[0] = join_lst_lst(mutation_route)
                break

        if offspring[1] <= self.bigM:
            offspring[1] = self.fittest_score(offspring[0])
        return offspring

    # evolution
    def evol(self):

        # loop through all generation
        for gen in range(self.gen_max):
            print("gen", gen)
            new_population = dict()  # []
            # Create pool
            pool = self.tournament_selection(
                self.population, tournament_size=self.tour_size, parent_pool_size=self.pop_size)
            print("pool", pool)
            print(len(pool))

            # population --> {idx: [individual, fittest_score]}
            for i in range(len(self.population) // 2):
                idx1 = 2*i
                idx2 = 2*i+1

                # select parent
                parent1 = self.population[pool[idx1]]
                parent2 = self.population[pool[idx2]]
                # print("parent1", parent1)
                # print("parent2", parent2)

                # crossover
                [route1, cost1], [route2, cost2] = self.crossover(parent1, parent2)
                off1 = [route1, cost1]
                off2 = [route2, cost2]
                # print("off1", off1)
                # print("off2", off2)
                
                # mutation
                # random in [0, 1]
                # swap_rate = 0.05, inversion_rate = 0.05 --> rate
                rate = np.random.rand()
                if rate <= self.mutate_prob:  # rate > mutate_prob -->
                    # off1 = np.random.choice(
                    #     [self.swap_two_node(off1), self.inversion_node(off1)])
                    # off2 = np.random.choice(
                    #     [self.swap_two_node(off2), self.inversion_node(off2)])
                    off1 = self.swap_two_node(off1)
                    off2 = self.swap_two_node(off2)

                new_population[idx1] = off1
                new_population[idx2] = off2
            new_population = add_key_dict(new_population, self.pop_size)
            self.population.update(new_population)
            self.population = sort_population(self.population, self.pop_size)
            self.best_indi = self.population[0]
        return self.population


def read_data(url):
    df = pd.read_excel(url, "parameters")
    n = int(df.iloc[0, 1])
    nc = int(df.iloc[1, 1])
    k = int(df.iloc[2, 1])
    p = int(df.iloc[3, 1])

    N = range(n+1)            # root node
    NC = range(1, nc+1)       # destination node
    K = range(1, k+1)       # set of vehicle
    P = range(1, p+1)       # set of compartment

    # parameters

    # quantity of type p at hospital i
    q = np.array(pd.read_excel(
        "data.xlsx", "coor").iloc[:, [3, 4],], dtype=np.float32)

    # distance
    x = np.array(pd.read_excel("data.xlsx", "coor"))[:, 1]
    y = np.array(pd.read_excel("data.xlsx", "coor"))[:, 2]

    d = dict()
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                d[(i, j)] = 0
            else:
                d[(i, j)] = float(
                    round(math.sqrt((x[i-1] - x[j-1])**2 + (y[i-1] - y[j-1])**2), 2))

    # Capacity
    Q = np.transpose(
        np.delete(
            np.array(pd.read_excel("data.xlsx", "capacity"), dtype=np.float32), 0, 1))
    Q = Q.squeeze()

    instance = {}
    instance["num_hospital"] = n-1
    instance["num_vehicle"] = k
    instance["num_compart"] = p
    instance["hospital_qty"] = q
    instance["vehicle_cap"] = Q
    instance["travel_dist"] = d

    return instance


if __name__ == "__main__":

    GA_params = {
        "pop_size": 15,
        "gen_max": 5,
        "mutate_prob": 0.1,
        "cross_rate": 0.5,
        "tour_size": 2
    }

    instance = read_data("data.xlsx")

    ga = GA(instance, GA_params)
    os.system('cls')
    print(ga.population)
    parent1 = [[0, 5, 3, 6, 0, 7, 0, 1, 2, 0, 4, 0], 324.42999999999995]
    parent2 = [[0, 7, 6, 0, 2, 0, 3, 4, 0, 5, 1, 0], 293.85]
    test = [[0, 1, 0, 4, 6, 7, 0, 4, 2, 0], 100]
    # print(ga.tournament_selection(ga.population, 50, 100))
    # print(ga.evol())
    # print(ga.crossover(parent1, parent2))
    # print(ga.population)
    # print(ga.tournament_selection(ga.population, 10, 100))
    print(ga.evol())
    print(ga.best_indi)
    # print(ga.capacity_constraint([0, 7, 5, 4, 0], 1e10, [1, 2]))

