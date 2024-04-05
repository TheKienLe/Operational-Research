import pandas as pd
import numpy as np
import random


class GA:
    def __init__(self, chromosome, truck_cap, num_fac, factory_qty) -> None:
        # initial solution: represent by a list
        self.chromosome = chromosome
        # truck capacity for each compartment: truck_cap = [compart_1, compart_2]
        self.truck_cap = truck_cap
        # number of factory
        self.factory = num_fac
        # factory quantity for each compartment: factory_qty = {factory : [compart_1, compart_2]}
        self.factory_qty = factory_qty

    def shuffle(self, genes):
        random.shuffle(genes)
        return genes

    def initialize_population(self, genes, population_size=50):
        # population
        data = {"population": []}

        for _ in range(population_size):
            temp_genes = self.shuffle(genes)
            result = [0]
            current_id = 0  # first factory
            self.num_truck = 0

            while temp_genes[-1] not in result:
                temp_cap = [0, 0]  # initial capacity of truck
                for id in range(current_id, self.factory):
                    # update qty of compart type 1
                    temp_cap[0] += self.factory_qty[temp_genes[id]][0]
                    # update qty of compart type 2
                    temp_cap[1] += self.factory_qty[temp_genes[id]][1]
                    if temp_cap[0] > self.truck_cap[0] or\
                            temp_cap[1] > self.truck_cap[1]:
                        result.append(0)
                        current_id = id
                        self.num_truck += 1
                        break
                    else:
                        result.append(temp_genes[id])
            if result not in data["population"]:
                data["population"].append(result)

        df = pd.DataFrame(data)

        return df


chromosome = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
truck_cap = np.transpose(
    np.delete(np.array(pd.read_excel("data.xlsx", "capacity")), 0, 1))[0]
num_fac = 10


def lst_to_dict(lst):
    q = dict()
    for i in range(num_fac):
        q[i+1] = lst[i]
    return q


factory_qty = np.array(pd.read_excel("data.xlsx", "test"))[:, [1, 2]]
factory_qty = lst_to_dict(factory_qty)
print(factory_qty)

ga = GA(chromosome, truck_cap, num_fac, factory_qty)
print(ga.initialize_population(chromosome))
