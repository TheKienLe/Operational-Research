import numpy as np
import math
import random


def lst_to_dict(lst):
    q = dict()
    for i in range(len(lst)):
        q[i] = lst[i]
    return q


def total_route(parent, split_at_value):  # parent input lst []
    my_lst = np.array(parent)
    # np.where(condition) return a tupple like: (result that satisfied condition, data_type)
    # np.where(condition)[0] access first element of tuple
    index = np.where(my_lst == split_at_value)[0]
    result = [parent[index[i]+1:index[i+1]]
              for i in range(len(index)) if i <= len(index) - 2]
    # remove 0 next to each other i.e [0, 1, 2, 4, 0, 0, 0, 8, 7, 0]
    result = [i for i in result if i != []]
    # return list of route for each parent
    return result


# total_round_of_parent from total_route func
def select_route_cross(total_round_of_parent, num_route_cross):
    cross_route = random.sample(total_round_of_parent, k=num_route_cross)
    return cross_route


def remove_zeros(individual):
    # for i in range(len(individual)):
    #     if individual[i] == 0 and individual[i+1] == 0:
    #         individual.remove(individual[i])
    pass

# testing function


def crossover(parent1, parent2):  # parent type [chromosome, fitness]
    parent_route_1 = total_route(parent1[0], 0)
    parent_route_2 = total_route(parent2[0], 0)
    cross_route1 = select_route_cross(parent_route_1, 2)
    cross_route2 = select_route_cross(parent_route_2, 2)
    result = []  # the place to contain all offsprings
    # remove element in route1 out of parent2[0]
    for route1 in cross_route1:
        chromosome2 = [node for node in parent2[0] if node not in route1]
        # insert element from route1 to parent2[0] to get new offspring
        for node in route1:
            pass
            # remove element in route2 out of parent1[0]
    for route2 in cross_route2:
        chromosome1 = [node for node in parent2[0] if node not in route2]


parent1 = [[0, 7, 0, 2, 0, 6, 4, 1, 0, 3, 5, 0], 322.96]
parent2 = [[0, 1, 6, 0, 7, 0, 2, 0, 3, 5, 0, 4, 0], 338.49]
test_route = [0, 1, 2, 4, 0, 0, 0, 6, 8, 0]
print(total_route(test_route, 0))
