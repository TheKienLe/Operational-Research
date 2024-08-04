import numpy as np
import random
from copy import deepcopy

def sort_distance(distance):
    return dict(sorted(distance.items(), key=lambda item: item[1]))

def clus_port_distance(dtms, cluster_cus, ports):
    # Initialize the result dictionary
    distance_from_port = {port: 0 for port in ports}

    for port in ports:
        for cus in cluster_cus:
            distance_from_port[port] += dtms[port, cus]

    distance_from_port = sort_distance(distance_from_port)
    return distance_from_port

def sort_by_demand(cluster, reverse=True):
    return dict(sorted(cluster.items(), key=lambda item: item[1]['demand'], reverse=reverse))

def sort_end_by_mother(end, mother):
    return {key: end[key] for key in mother if key in end}


def sort_population(population):
    population = sorted(population, key=lambda x: x["fitness"])
    # print(population[:3])
    return population

def tournament_selection(population, tour_size):
    tour_pop = random.sample(population, tour_size)
    return deepcopy(sort_population(tour_pop)[0])

def route_permutation(route):
    return list(np.random.permutation(route))

def evaluate(chromosome, end_chromo, distance_matrix, 
             demand, cap_vehicle, 
             return_subroute=False, mother=True, air_port=None):
    """
    Evaluate a chromosome to calculate its total distance and number of vehicles.

    Parameters
    ----------
    chromosome : list
        A list representing a route of customers.
    distance_matrix : list of lists
        A 2D list representing distances between customers.
    demand : list
        A list of demands for each customer.
    cap_vehicle : int
        The capacity of each vehicle.
    return_subroute : bool, optional
        Whether to return sub-routes instead of total distance and vehicle count (default is False).

    Returns
    -------
    total_distance : float
        The total distance of the route.
    n_vehicle : int
        The number of vehicles used.
    sub_route : list of lists, optional
        A list of sub-routes if `return_subroute` is True.

    Examples
    --------
    >>> chromosome = [1, 2, 3, 4, 5]
    >>> distance_matrix = [[0, 2, 9, 10], [1, 0, 6, 4], [15, 7, 0, 8], [6, 3, 12, 0]]
    >>> demand = [0, 2, 4, 8, 1]
    >>> cap_vehicle = 10
    >>> evaluate(chromosome, distance_matrix, demand, cap_vehicle)
    42
    """
    total_distance = 0
    cur_load = 0
    n_vehicle = 0
    sub_route = []
    route = []
    cust_index_begin = 0
    for cus in chromosome:
        cur_load += demand[cus]['demand']

        if cur_load > cap_vehicle:
            if return_subroute:
                sub_route.append(route[:])
            total_distance += calculate_distance(route, air_port,
                                                 distance_matrix, 
                                                 mother)
            n_vehicle += 1
            cur_load = demand[cus]['demand']
            route = [end_chromo[cus][0]] if mother  else [cus]
        else:
            if mother:
                route.append(end_chromo[cus][0])
            else:  
                route.append(cus)
    total_distance += calculate_distance(route, air_port, 
                                         distance_matrix, 
                                         mother)

    n_vehicle += 1
    if return_subroute:
        sub_route.append(route[:])
        return sub_route
    return total_distance, n_vehicle

def calculate_distance(route, air_port, distance_matrix, mother=True):
    """
    Calculate the total distance of a given route.

    Parameters
    ----------
    route : list
        A list representing a route of customers.
    distance_matrix : list of lists
        A 2D list representing distances between customers.

    Returns
    -------
    distance : float
        The total distance of the route.

    Examples
    --------
    >>> route = [1, 2, 3]
    >>> distance_matrix = [[0, 2, 9, 10], [1, 0, 6, 4], [15, 7, 0, 8], [6, 3, 12, 0]]
    >>> calculate_distance(route, distance_matrix)
    19
    """
    distance = 0
    depot = 0 if mother else air_port
    distance += distance_matrix[depot][route[0]] if mother else 0
    distance += distance_matrix[depot][route[-1]]
    for i in range(0, len(route)-1):
        distance += distance_matrix[route[i]][route[i+1]]
    return distance

def replace(population, chromo_in, chromo_out):
    """
    Replace a chromosome in the population.

    Parameters
    ----------
    population : list of lists
        A list of chromosomes.
    chromo_in : list
        The chromosome to add to the population.
    chromo_out : list
        The chromosome to remove from the population.

    Returns
    -------
    None

    Examples
    --------
    >>> population = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> replace(population, [10, 11, 12], [4, 5, 6])
    >>> population
    [[1, 2, 3], [10, 11, 12], [7, 8, 9]]
    """
    population[-1] = chromo_in


def check_validity(chromosome, length):
    """
    Check if a chromosome is valid.

    Parameters
    ----------
    chromosome : list
        A list representing a route of customers.
    length : int
        The expected number of customers in the chromosome.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If the chromosome is invalid.

    Examples
    --------
    >>> chromosome = [1, 2, 3, 4, 5]
    >>> check_validity(chromosome, 5)
    >>> check_validity([1, 2, 4, 5], 5)
    Exception: invalid chromosome
    """

    for i in range(1, length+1):
        if i not in chromosome:
            raise Exception("invalid chromosome")
        
def ordered_crossover(chromo1, chromo2, mother):
    """
    Perform ordered crossover on two parent chromosomes to produce offspring.

    Parameters
    ----------
    chromo1 : list
        The first parent chromosome.
    chromo2 : list
        The second parent chromosome.

    Returns
    -------
    ind1 : listz
        The first offspring chromosome.
    ind2 : list
        The second offspring chromosome.

    Examples
    --------
    >>> random.seed(42)
    >>> chromo1 = [3, 5, 9, 2, 4]
    >>> chromo2 = [9, 2, 5, 4, 3]
    >>> ordered_crossover(chromo1, chromo2)
    ([3, 2, 5, 4, 9], [9, 5, 2, 4, 3])
    """
    if not mother:
        launch_node = chromo1[0].copy()
        chromo1.remove(launch_node)
        chromo2.remove(launch_node)

    size = min(len(chromo1), len(chromo2))
    a, b = random.sample(range(size), 2) 
    if a > b:
        a, b = b, a

    ind1 = [None] * size
    ind2 = [None] * size

    # Copy the segment from a to b
    ind1[a:b+1] = chromo1[a:b+1].copy()
    ind2[a:b+1] = chromo2[a:b+1].copy()
    
        
    # Create the mapping of the remaining elements
    def fill_remaining(ind, chromo_source, start, end):
        current_index = (end + 1) % size
        for i in range(end + 1, end + size + 1):
            index = i % size
            if (chromo_source[index] not in ind):
                ind[current_index] = chromo_source[index].copy()
                current_index = (current_index + 1) % size


    fill_remaining(ind1, chromo2, a, b)
    fill_remaining(ind2, chromo1, a, b)
    if not mother:
        route1 = [launch_node]
        route2 = [launch_node]
        route1.extend(ind1)
        route2.extend(ind2)
        ind1 = route1
        ind2 = route2

    return ind1, ind2

def mutate(chromosome, probability):
        """
        Perform mutation on a chromosome with a given probability.

        Parameters
        ----------
        chromosome : list
            A list representing a route of customers.
        probability : float
            The probability of mutation.

        Returns
        -------
        chromosome : list
            The mutated chromosome.

        Examples
        --------
        >>> random.seed(42)
        >>> chromosome = [1, 2, 3, 4, 5]
        >>> mutate(chromosome, 0.5)
        [1, 3, 2, 4, 5]
        """

        if random.random() < probability:
            # Randomly select two distinct indices (`index1` and `index2`) from the chromosome.
            index1, index2 = random.sample(range(len(chromosome)), 2)
            
            # Swap the genes at the randomly selected indices `index1` and `index2`.
            chromosome[index1], chromosome[index2] = chromosome[index2], chromosome[index1]
            
            # Select two new random indices and sort them to ensure `index1` is less than `index2`.
            index1, index2 = sorted(random.sample(range(len(chromosome)), 2))
            
            # Reverse the sublist of genes between `index1` and `index2` (inclusive) and concatenate with the unchanged part before `index1`.
            mutated = chromosome[:index1] + \
                list(reversed(chromosome[index1:index2+1]))
        
            # If `index2` is not the last index, concatenate the unchanged part after `index2` to the mutated list.
            if index2 < len(chromosome) - 1:
                mutated += chromosome[index2+1:]
            

            return mutated
            # Return the mutated chromosome.
        
        return chromosome
        # If the random number is not less than `probability`, return the original chromosome without any changes.

def port_in_used():
    pass

