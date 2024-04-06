import pandas as pd
import numpy as np
import random
import math


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

        # GA parameters
        self.pop_size = GA_params["pop_size"]  # population size
        self.gen_max = GA_params["gen_max"]  # Max number of generation
        self.mutate_prob = GA_params["mutate_prob"]  # Probability of mutation
        # percentage of route being crossovered
        self.cross_rate = GA_params["cross_rate"]

        # Initial population
        self.population = self.initialize_population(self.pop_size)

        # Best individual (update after finishing fitness_score function)

    def initialize_population(self, population_size=50):
        # population
        ind0 = np.arange(1, self.num_hospital+1)
        population = []

        for _ in range(population_size):
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
                    final_genes.extend([0, host])  # assign to the next truck
                    # update current capacity
                    current_cap = self.hospital_qty[host].copy()

            final_genes.append(0)  # return to collection center
            population.append(final_genes)  # add new individual to the truck

        def lst_to_dict(lst):
            q = dict()
            for id in range(len(lst)):
                q[id+1] = lst[id]
            return q

        return lst_to_dict(population)

    def fittest_score(self, population, distance):
        result = dict()
        dict_value = list(population.values())
        for index in range(len(dict_value)):
            total_distance = 0
            for node_index in range(len(dict_value[index])):
                d1 = dict_value[index][node_index]
                d2 = dict_value[index][node_index+1]
                total_distance += distance[(d1, d2)]
                if node_index == len(dict_value[index]) - 2:
                    break
            result[index + 1] = [dict_value[index], total_distance]
        return result

    def minimize(self, fittest_score):
        values = [value[1] for value in fittest_score.values()]
        min_value = min(values)
        index = values.index(min_value)
        self.best_ind = index + 1
        return self.best_ind

        # lst1 = [0, 7, 5, 0, 1, 4, 3, 0, 2, 0, 6, 0]
        # lst2 = [0, 7, 4, 1, 0, 3, 0, 2, 0, 6, 5, 0]
        # distance1 = 0
        # distance2 = 0
        # for i in range(len(lst1)):
        #     distance1 += distance[(lst1[i], lst1[i+1])]
        #     if i == len(lst1) - 2:
        #         break

        # for i in range(len(lst2)):
        #     distance2 += distance[(lst2[i], lst2[i+1])]
        #     if i == len(lst2) - 2:
        #         break
        # print(distance1, distance2)


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
    for i in N:
        for j in N:
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
        "pop_size": 100,
        "gen_max": 50,
        "mutate_prob": 0.1,
        "cross_rate": 0.5,
    }

    instance = read_data("data.xlsx")

    ga = GA(instance, GA_params)
    # print(ga.population[:3])
    population = ga.population
    distance = ga.travel_dist
    fittest_score = ga.fittest_score(population, distance)
    print(fittest_score)
    print(ga.minimize(fittest_score))
    print(ga.best_ind)
