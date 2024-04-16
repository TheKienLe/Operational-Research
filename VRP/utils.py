import numpy as np
import math
import random


def lst_to_dict(lst):
    q = dict()
    for i in range(len(lst)):
        q[i] = lst[i]
    return q


def total_route(parent, split_at_value):  # parent input list []
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
def select_route_cross(total_route_of_parent, num_route_cross):
    cross_route = []
    while len(cross_route) < 2:
        random_route = random.choice(total_route_of_parent)
        if random_route not in cross_route:
            cross_route.append(random_route)
    return cross_route

# concatenate list


def join_lst_lst(lst):  # input is a list in list i.e [[1, 2], [3,4] ,...]
    result = [0]
    for element in lst:
        result += element
        result += [0]
    return result


def join_cross_route(lst):
    result = []
    for element in lst:
        result += element
    return result

# min list to find offspring input is list with format [[[[1, 2, 4], [3,5], [10]], fitness_score]]


def min_lst(lst):
    initial_arr = np.array([value[1] for value in lst])
    min_idx = np.argmin(initial_arr)
    return lst[min_idx][0]

# testing function


def crossover(parent1, parent2):  # parent type [chromosome, fitness]
    parent_route_1 = total_route(parent1[0], 0)
    parent_route_2 = total_route(parent2[0], 0)
    cross_route1 = select_route_cross(parent_route_1, 2)
    cross_route2 = select_route_cross(parent_route_2, 2)
    result = []  # the place to contain all offsprings

    chromosome2 = parent2[0].copy()
    for route1 in cross_route1:
        for route in route1:
            chromosome2.remove(route)

    chromosome1 = parent1[0].copy()
    for route2 in cross_route2:
        for route in route2:
            chromosome1.remove(route)

    # print(join_lst_lst(total_route(chromosome1, 0)),
    #       join_lst_lst(total_route(chromosome2, 0)))
    # print(cross_route2, cross_route1)


parent1 = [[0, 7, 0, 2, 0, 6, 4, 1, 0, 3, 5, 0], 322.96]
parent2 = [[0, 1, 6, 0, 7, 0, 2, 0, 3, 5, 0, 4, 0], 338.49]

# print(crossover(parent1, parent2))
test = [0, 1, 0, 4, 6, 9, 0, 4, 2, 0]
print(total_route(test, 0))
print(join_lst_lst(total_route(test, 0)))

# print(min_lst([[[[1, 2], [3, 4]], 10], [[[4, 5], [7, 8], [9, 10]], 100]]))
