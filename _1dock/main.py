import csv
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
# Function to calculate the travel time


def calculate_travel_time(start_node, end_node):
    # Calculate the travel time between two nodes (e.g., usingthe assumed velocity)
    distance = calculate_distance(start_node, end_node)
    travel_time = distance / 2
    return travel_time


# Function to calculate the distance between two nodes
def calculate_distance(start_node, end_node):
    # Calculate the Euclidean distance between two nodes
    x1, y1 = start_node['x'], start_node['y']
    x2, y2 = end_node['x'], end_node['y']
    distance = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
    return distance


# Step 1: Load data
# Function to load instance data from a file
def load_instance_data(instance_file):

    with open(instance_file, 'r') as file:
        lines = file.readlines()
        num_vehicles = int(lines[0].split()[1])
        num_customers = int(lines[0].split()[2])
        num_depots = int(lines[0].split()[3])
        customers = []
        depots = []
        vehicles = []
        # Extract vehicle data
        for i in range(1, num_vehicles+1):
            vehicle_data = lines[i].split()
            vehicle = {
                'Max Duration': int(vehicle_data[0]),
                'Max Load': int(vehicle_data[1])
            }
            vehicles.append(vehicle)

        # Extract customer data
        for i in range(num_vehicles+1, num_vehicles + num_customers+1):
            customer_data = lines[i].split()
            customer = {
                'index': int(customer_data[0]),
                'x': float(customer_data[1]),
                'y': float(customer_data[2]),
                'service_time': int(customer_data[3]),
                'demand': int(customer_data[4]),
                'ready_time': int(customer_data[-2]),
                'due_time': int(customer_data[-1]),
            }
            customers.append(customer)

        # Extract depot data
        for i in range(num_customers+num_vehicles+1, num_vehicles+num_customers+num_depots+1):
            depot_data = lines[i].split()
            depot = {
                'index': int(depot_data[0]),
                'x': float(depot_data[1]),
                'y': float(depot_data[2]),
                'Max Capacity': int(depot_data[-1])
            }
            depots.append(depot)
            return num_vehicles, num_customers, num_depots, customers, depots, vehicles


# Generate the distance matrix and travel time matrix
def generate_distance_matrix():
    num_nodes = num_customers + num_depots
    distance_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    travel_time_matrix = [[0] * num_nodes for _ in
                          range(num_nodes)]
    # Calculate distances between customers
    for i in range(num_customers):
        for j in range(i+1, num_customers):
            distance = calculate_distance(customers[i], customers[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
            travel_time = distance / (2)  # Assuming velocity is 5/6

            travel_time_matrix[i][j] = travel_time
            travel_time_matrix[j][i] = travel_time

    # Calculate distances and travel time between depots and customers
    for i in range(num_customers):
        for j in range(num_customers, num_nodes):

            distance = calculate_distance(
                customers[i], depots[j - num_customers])

            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
            travel_time = distance / (2)  # Assuming velocity is 5/6
            travel_time_matrix[i][j] = travel_time
            travel_time_matrix[j][i] = travel_time
    return distance_matrix, travel_time_matrix


def save_distance_matrix(distance_matrix, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in distance_matrix:
            writer.writerow(row)


def save_travel_time_matrix(travel_time_matrix, filename):

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in travel_time_matrix:
            writer.writerow(row)


# Step 2: Define parameters
population_size = 50
num_generations = 100
tournament_size = 10
crossover_rate = 0.8
mutation_rate = 0.2
threshold = 0.8
penalty_weight_1 = 10000
penalty_weight_2 = 10000
penalty_weight_3 = 10000
penalty_weight_4 = 10000
penalty_weight_5 = 10000
penalty_weight_6 = 10000
penalty_weight_7 = 10000
penalty_weight_8 = 10000


# Step 4: Define fitness calculation
def calculate_fitness(individual):

    total_distance = 0.0
    total_duration = 0.0
    visited_customers = set()

    penalty = 0.0
    for vehicle_route in individual:
        if len(vehicle_route) <= 2:
            break
        else:
            total_distance, total_duration = calculate_route_distance_duration(
                vehicle_route)

        # Constraint 1: Ensuring the vehicle that leaves the customer is the same as the one that visits the customer
        if vehicle_route[0] != vehicle_route[-1]:
            penalty += penalty_weight_1

        # Constraint 2: Each customer is assigned to a vehicle
        for i in range(1, len(vehicle_route) - 2):
            if vehicle_route[i] == vehicle_route[i + 1]:
                penalty += penalty_weight_2

        # Constraint 3: Determining whether each customer is assigned to a depot or not
        for customer in vehicle_route[1:-1]:
            if customer in visited_customers:
                penalty += penalty_weight_3
            else:
                visited_customers.add(customer)

        # Constraint 4: Maximum load for each vehicle
        total_demand = 0
        for i in range(len(vehicle_route) - 2):
            customer = customers[vehicle_route[i + 1]]
            total_demand += customer['demand']
            if total_demand > vehicles[0]['Max Load']:
                penalty += penalty_weight_4

        # Constraint 5: Maximum duration for each vehicle
        if total_duration > vehicles[0]['Max Duration']:
            penalty += penalty_weight_5

        # Constraint 6,7,8: Time constraints between consecutive customers
        current_time = 0.0
        for i in range(len(vehicle_route) - 1):
            if vehicle_route[i] > num_customers:
                start_node = vehicle_route[i]
                end_node = vehicle_route[i + 1]
                current_time += travel_time_matrix[start_node - 1][end_node]
            elif vehicle_route[i + 1] > num_customers:
                start_node = vehicle_route[i]
                end_node = vehicle_route[i + 1]
                current_time += travel_time_matrix[start_node][end_node - 1]
            else:
                start_node = customers[vehicle_route[i]]
                end_node = customers[vehicle_route[i + 1]]
                distance = calculate_distance(start_node, end_node)
                current_time += start_node['service_time']
                current_time += distance / (2)
                current_time = max(current_time, end_node['ready_time'])
                if 'ready_time' in end_node:
                    if current_time < end_node['ready_time']:
                        penalty += penalty_weight_6
                if 'ready_time' and 'service_time' in end_node:
                    if current_time + end_node['service_time'] < end_node['ready_time']:
                        penalty += penalty_weight_7
                if 'due_time' and 'service_time' in end_node:
                    if current_time + end_node['service_time'] > end_node['due_time']:
                        penalty += penalty_weight_8
        fitness = total_duration + penalty
    return fitness


def calculate_route_distance_duration(vehicle_route):
    distance = 0
    duration = 0.0
    current_time = 0.0
    if len(vehicle_route) <= 2:
        distance = 0
        duration = 0
    else:
        for i in range(len(vehicle_route) - 1):
            if vehicle_route[i] > num_customers:
                start_node = vehicle_route[i]
                end_node = vehicle_route[i + 1]
                distance += distance_matrix[start_node - 1][end_node]
                duration += travel_time_matrix[start_node - 1][end_node]
            elif vehicle_route[i+1] > num_customers:
                start_node = vehicle_route[i]
                end_node = vehicle_route[i + 1]
                distance += distance_matrix[start_node][end_node - 1]

                duration += travel_time_matrix[start_node][end_node-1]
            else:
                customer1 = customers[vehicle_route[i]]
                customer2 = customers[vehicle_route[i + 1]]
                distance += calculate_distance(customer1, customer2)
                current_time += distance/(4)
                current_time = max(current_time, customer2['ready_time'])
                current_time += customer2['service_time']
                duration = max(duration, current_time -
                               customer2['ready_time'])
    return distance, duration


# Step 5: Generate initial population
def generate_initial_population(population_size, num_vehicles, num_customers, customers, depots, vehicles):
    population = []
    for _ in range(population_size):
        remaining_customers = list(range(num_customers))
        random.shuffle(remaining_customers)
        random.shuffle(depots)
        individual = create_vehicle_sequence(num_vehicles,
                                             remaining_customers, customers, depots, vehicles)
        population.append(individual)
    return population


def create_vehicle_sequence(num_vehicles, remaining_customers, customers, depots, vehicles):
    vehicle_sequence = []
    depot_capacity = 1000
    for vehicle_idx in range(num_vehicles):
        depot_idx = vehicle_idx % len(depots)
        depot_index = depots[depot_idx]['index']
        sequence = [depot_index]
        maxLoad = vehicles[vehicle_idx]['Max Load']
        maxDuration = vehicles[vehicle_idx]['Max Duration']
        current_capacity = 0
        current_time = 0
        while remaining_customers:
            feasible_customers = []
            distances = []
            durations = []

            for customer in remaining_customers:
                if customer <= num_customers:
                    distance = calculate_distance(
                        customers[sequence[-1] - num_customers], customers[customer])
                else:
                    distance = calculate_distance(
                        depots[depot_idx], customers[customer])

                arrival_time = current_time + distance / 4
                waiting_time = max(
                    0, customers[customer]['ready_time'] - arrival_time)
                service_time = customers[customer]['service_time']
                update_current_time = arrival_time + waiting_time + service_time

                if (update_current_time <= maxDuration and current_capacity + customers[customer]['demand'] <= maxLoad and current_capacity + customers[customer]['demand'] <= depot_capacity):
                    feasible_customers.append(customer)
                    distances.append(distance)
                    durations.append(update_current_time)

            if feasible_customers:
                best_customer = feasible_customers[np.argmin(distances)]
                sequence.append(best_customer)
                remaining_customers.remove(best_customer)
                current_capacity += customers[best_customer]['demand']
                current_time = durations[np.argmin(distances)]
            else:
                break

        sequence.append(depot_index)
        vehicle_sequence.append(sequence)

    return vehicle_sequence


# Step 6: Genetic operators (Selection, Crossover, Mutation)
def tournament_selection(population, fitness_values, tournament_size, threshold):

    # choose random k individuals in the population
    k = tournament_size
    p = threshold
    new_gen = []
    for _ in range(k):
        tournament_indices = random.sample(range(len(population)), k)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        sorted_indices = sorted(range(
            len(tournament_fitness)), key=lambda x: tournament_fitness[x], reverse=True)
        # Compute the selection probabilities
        probabilities = np.array([p * (1 - p) ** i for i in range(k)])
        # Normalize theprobabilities to sum to 1
        probabilities /= np.sum(probabilities)
        # Select the best individuals according to the probabilities
        selected_index = np.random.choice(sorted_indices, p=probabilities)

    new_gen.append(population[tournament_indices[selected_index]])

    return new_gen


def best_cost_route_crossover(parent1, parent2):

    # Randomly select a depot
    depot = random.choice(depots)['index']
    # Randomly select p1,p2 from each parent
    p1 = random.choice(parent1)
    p2 = random.choice(parent2)
    # Randomly select a route from each parent, r1 from p1, r2
    r1 = random.choice(p1)
    r2 = random.choice(p2)
    # Remove all customers of route1 from parent2, and route2
    new_p1 = remove_customer(p1, r2)
    new_p2 = remove_customer(p2, r1)
    # Recreate individual
    offspring1 = recreate_vehicle_sequence(new_p1, r2, depot)
    offspring2 = recreate_vehicle_sequence(new_p2, r1, depot)
    return offspring1, offspring2


def remove_customer(p, r):
    new_routes = []
    removed_customers = set()
    for route in p:
        new_route = route[:]
        for customer in r[1:-1]:
            if customer in new_route[1:-1]:
                new_route.remove(customer)
                removed_customers.add(customer)
        new_routes.append(new_route)
    for sequence in new_routes:
        if len(sequence) == 2:
            new_routes.remove(sequence)
    return new_routes


def recreate_vehicle_sequence(newp, remove_route, depot):

    num_routes = len(newp)
    if num_routes < num_vehicles:
        # Add new routes with the removed customers
        remaining_depots = [depot]
        remaining_customers = list(range(num_customers))
        random.shuffle(remaining_customers)
        for route in newp:
            remaining_depots.append(route[0])
        depot_info = []
        for i in remaining_depots:
            depot_info.append(depots[i-num_customers-1])
        # Adjust existing routes by inserting the removed customers
        new_sequence = create_vehicle_sequence(
            num_vehicles, remaining_customers, customers, depot_info, vehicles)
    else:
        remaining_depots = []
        remaining_customers = list(range(num_customers))
        random.shuffle(remaining_customers)
        for route in newp:
            remaining_depots.append(route[0])
        depot_info = []
        for i in remaining_depots:
            depot_info.append(depots[i-num_customers-1])
            # Adjust existing routes by inserting the removed customers
            new_sequence = create_vehicle_sequence(
                num_routes, remaining_customers, customers, depot_info, vehicles)
    return new_sequence


def swap_mutation(individual):
    route = random.choice(individual)
    if len(route) >= 4:
        customer1, customer2 = random.sample(route[1:-1], 2)
        index1 = route.index(customer1)
        index2 = route.index(customer2)
        route[index1], route[index2] = route[index2], route[index1]

    return individual


def bestIndividual(offspring):
    fitness_values = [calculate_fitness(individual)
                      for individual in offspring]
    best_index = min(range(len(offspring)), key=lambda i: fitness_values[i])
    best_individual = offspring[best_index]
    return best_individual


def _get_edges_from_route(route):
    assert len(route) >= 2
    edges = []
    for i, n in enumerate(route):
        if i >= 1:
            edges.append((route[i - 1], route[i]))
    return edges


def draw_route(pos, route, names, route_number, save_file=False, file_name=None):

    fig, ax = plt.subplots(figsize=(8, 6))
    # Set plot properties
    customer_indices = [n + 1 for n in route[1:-1] if n in pos]
    # Exclude the depot index
    ax.scatter([pos[n][0] for n in customer_indices], [pos[n][1]
                                                       for n in customer_indices])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Route {route_number}')
    verts = [pos[route[0]]] + [pos[n] for n in customer_indices] + \
        [pos[route[-1]]]  # Include the depot as the start and end
    path = Path(verts)
    patch = patches.PathPatch(path, facecolor='none', lw=1,
                              zorder=0)
    ax.add_patch(patch)
    # Get x and y coordinates for customers in the route (exclude depot)
    x = [pos[n][0] for n in customer_indices]
    y = [pos[n][1] for n in customer_indices]
    ax.plot(x, y, marker='o', linestyle='-', markersize=8)
    # Add customer index labels to each customer location
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.text(xi, yi, str(route[i+1]), ha='center', va='center',
                fontsize=10, color='black')  # Add 1 to index

    # Add the depot as the first and last point in the route
    depot_x, depot_y = pos[route[0]]
    ax.plot(depot_x, depot_y, marker='D', markersize=8,
            color='red', label='Depot')
    ax.text(depot_x, depot_y, 'Depot', ha='center', va='center',
            fontsize=10, color='black')
    # Show the plot for this route
    if save_file:
        if file_name is None:
            file_name = f"Route_{route_number}"
        plt.savefig(f"{file_name}.pdf", bbox_inches='tight',
                    transparent=True, pad_inches=0.1)
    else:
        plt.show()
    plt.close()


def plot_routes(best_solution, customers, depots):

    # Combine routes for each depot
    depot_routes = {}
    for depot in depots:
        depot_routes[depot['index']] = [
            route for route in best_solution if route[0] == depot['index']]

    # Add the depot positions to the pos dictionary
    pos = {customer['index']: (customer['x'], customer['y'])
           for customer in customers}
    names = {customer['index']: customer['index'] for customer in
             customers}
    for depot in depots:
        pos[depot['index']] = (depot['x'], depot['y'])

    # Loop over depots and create separate figures for each route
    for depot_idx, depot in enumerate(depots):
        vehicle_routes = [
            route for route in depot_routes[depot['index']] if len(route) > 2]
        for i, route in enumerate(vehicle_routes):
            draw_route(pos, route, names, i + 1, save_file=True,
                       file_name=f"Depot_{depot['index']}_Route_{i + 1}")
# Genetic Algorithm


def genetic_algorithm(instance_file, population_size, num_generations, tournament_size, mutation_probability):

    # Step 1: Load instance data
    num_vehicles, num_customers, num_depots, customers, depots,
    vehicles = load_instance_data(instance_file)

    # Step 2: Initialize the population
    population = generate_initial_population(
        population_size, num_vehicles, num_customers, customers, depots, vehicles)

    # Step 3: Evaluate the fitness of the initial population
    fitness_values = []
    for individual in population:
        fitness_values.append(calculate_fitness(individual))
    generation_fitness = []
    best_solution_value = float('inf')
    best_solution = None
    initial_solution = bestIndividual(population)
    initial_solution_fitness = calculate_fitness(initial_solution)

    # Step 4: Iterate through generations
    for generation in range(num_generations):

        # Step 4.1: Perform tournament selection to select parents
        offspring = []
        new_population = []
        for _ in range(population_size):
            parent1 = tournament_selection(
                population, fitness_values, tournament_size, threshold)
            parent2 = tournament_selection(
                population, fitness_values, tournament_size, threshold)

        # Step 4.2: Create the offspring using best cost route crossover
        if random.uniform(0, 1) < crossover_rate:
            child1, child2 = best_cost_route_crossover(parent1, parent2)
            if child1 is not None:
                offspring.append(child1)
            if child2 is not None:
                offspring.append(child2)

        # Step 4.3: Perform swap mutation on the offspring
        for individual in offspring:
            if random.uniform(0, 1) < mutation_rate:
                new_individual = swap_mutation(individual)
                new_population.append(new_individual)

        best_individual = bestIndividual(new_population)
        current_solution_fitness = calculate_fitness(best_individual)

        if initial_solution_fitness < current_solution_fitness:
            current_solution_value = initial_solution_fitness
            best_individual = initial_solution
        else:
            current_solution_value = current_solution_fitness
        if current_solution_value < best_solution_value:
            best_solution_value = current_solution_value
            best_solution = best_individual
        generation_fitness.append(best_solution_value)

        # Step 4.4: Select best individual
    # Plot the best fitness value at each generation
    plt.plot(range(1, num_generations + 1), generation_fitness)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Value')
    plt.title('Convergence of Genetic Algorithm')
    plt.show()
    best_distance = sum(calculate_route_distance_duration(route)[
                        0] for route in best_solution)
    best_duration = sum(calculate_route_distance_duration(route)[
                        1] for route in best_solution)
    return best_solution, best_distance, best_duration, best_solution_value, generation_fitness


# individual = generate_random_individual()
instance_file = 'E:\\OR_Programming\\Operational-Research\\parameter.txt'
num_vehicles, num_customers, num_depots, customers, depots, vehicles = load_instance_data(
    instance_file)
distance_matrix, travel_time_matrix = generate_distance_matrix()
save_distance_matrix(distance_matrix, 'distance_matrix.csv')
save_travel_time_matrix(travel_time_matrix, 'travel_time_matrix.csv')
best_Ind, best_distance, best_duration, best_solution_value, generation_fitness = genetic_algorithm(
    instance_file, population_size, num_generations, tournament_size, mutation_rate)
# Calculate statistics over all generations
mean_fitness_values = np.mean(generation_fitness)
std_fitness_values = np.std(generation_fitness)
min_fitness_value = np.min(generation_fitness)
max_fitness_value = np.max(generation_fitness)
print((best_Ind))
print(best_distance)
print(best_duration)
print(best_solution_value)
print(f"Mean Fitness Value: {mean_fitness_values}")
print(f"Standard Deviation of Fitness Values: {std_fitness_values}")
print(f"Minimum Fitness Value: {min_fitness_value}")
print(f"Maximum Fitness Value: {max_fitness_value}")
