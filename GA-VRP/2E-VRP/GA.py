
class GA:
    def __init__(self, GA_params):
        # GA parameters:
        self.pop_size = GA_params["pop_size"]

        self.div_rate = GA_params["div_rate"] # div_rate = 10% [I, I, I, I |I, I] --> [I, I, I, I| I, I]

        self.pop_div = int(self.pop_size * self.div_rate)
        self.p_selection = GA_params["p_selection"] # isertion = 0.05
        self.p_routing = GA_params["p_routing"]
        # self.p_isertion = ...
        
        self.max_iter = GA_params["max_iter"]
     

        # Initialize population
        self.fitness_var, \
            self.truck_chromo_var, \
                self.motor_chromo_var = self.initialize_solu()
        self.best_chromo = [self.truck_chromo_var[-1], self.motor_chromo_var[-1]]


    def initialize_solu(self):
        truck_var = []
        motor_var = []
        cust_var = []
        truck_chromo_var = []
        motor_chromo_var = []
        fitness_var = []
        total_cost_var = []

        for i in range(self.pop_size):
            indi_var = initial_solution(data_url)
            truck_chromosome, motor_chromosome \
                = encode_chromosome(indi_var["customers"])
            truck_chromo_var.append(truck_chromosome)
            motor_chromo_var.append(motor_chromosome)
            fitness_var.append(total_cost(indi_var["customers"], indi_var["trucks"], indi_var["motors"], indi_var["depots"]))
        sorted_lists = [list(t) for t in zip(*sorted(zip(fitness_var, truck_chromo_var, motor_chromo_var), key=lambda x: x[0], reverse=True))]

        return sorted_lists

    def create_mating_pool(self):
        pool1 = []
        pool2 = []
        for i in range(self.pop_div):
            parent1 = [i] * curve(self.fitness_var[i])
            pool1.extend(parent1)

            parent2 = [i] * curve(self.fitness_var[i])
            pool2.extend(parent2)

        return pool1, pool2
    
    def crossover(self, truck1, truck2, motor1, motor2):
        # truck1
        num_truck = len(truck1)
        num_merge = round(np.random.uniform(1/2, 2/3)*(num_truck))

        ind1 = np.random.choice(num_truck, num_merge, replace=False)
        VMX1 = [truck1[i] for i in ind1]
        assigned_cust = [cust for route in VMX1 for cust in route]
        unassigned_cust = [cust for route in truck2 for cust in route if cust not in assigned_cust]
        # print("motor1",motor1)
        truck1 = VMX1
        for i in range(num_truck):
            if i not in ind1:
                truck1.append([0]) 
        # print("offspring1", truck1)            
        truck1, cost1 = cheapest_insertion(unassigned_cust, 
                                        truck1, motor1, 
                                        num_merge, 1)
        
        # truck2
        num_truck = len(truck2)
        num_merge = round(np.random.uniform(1/2, 2/3)*(num_truck))
        
        ind2 = np.random.choice(num_truck, num_merge, replace=False)
        VMX2 = [truck2[i] for i in ind2]
        assigned_cust = [cust for route in VMX2 for cust in route]
        unassigned_cust = [cust for route in truck1 for cust in route if cust not in assigned_cust]
        truck2 = VMX2
        for i in range(num_truck):
            if i not in ind2:
                truck2.append([0])            
        truck2, cost2 = cheapest_insertion(unassigned_cust, 
                                        truck2, motor2, 
                                        num_merge, 1)

        ##################################################
        
        # motor1
        num_motor = len(motor1)
        num_merge = round(np.random.uniform(1/2, 2/3)*(num_motor))

        ind1 = np.random.choice(num_motor, num_merge, replace=False)
        VMX1 = [motor1[i] for i in ind1]
        assigned_cust = [cust for route in VMX1 for cust in route]
        unassigned_cust = [cust for route in motor2 for cust in route[1:] if cust not in assigned_cust]
        offspring1 = VMX1
        for i in range(num_motor):
            if i not in ind1:
                offspring1.append([motor1[i][0]])             
        motor1, cost1 = cheapest_insertion(unassigned_cust, 
                                                truck1, offspring1, 
                                                num_merge, 2)
        
        # motor2
        num_motor = len(motor2)
        num_merge = round(np.random.uniform(1/2, 2/3)*(num_motor))
        
        ind2 = np.random.choice(num_motor, num_merge, replace=False)
        VMX2 = [motor2[i] for i in ind2]
        assigned_cust = [cust for route in VMX2 for cust in route]
        unassigned_cust = [cust for route in motor1 for cust in route[1:] if cust not in assigned_cust]
        offspring2 = VMX2
        for i in range(num_motor):
            if i not in ind1:
                offspring2.append([motor2[i][0]])             
        motor2, cost2 = cheapest_insertion(unassigned_cust, 
                                                truck2, offspring2, 
                                                num_merge, 2)
        

        return (truck1, motor1, cost1) if cost1 < cost2 else (truck2, motor2, cost2)

    def evolve(self):
        for it in range(self.max_iter):
            good_pop_truck = self.truck_chromo_var[self.pop_div:self.pop_size]
            good_pop_motor = self.motor_chromo_var[self.pop_div:self.pop_size]
            
            # Creat mating pool
            pool1, pool2 = self.create_mating_pool()
            new_pop_motor = []
            new_pop_truck = []
            fitness = []

             # Reproduction = Crossover
            for i in range(self.pop_div):
                # 1st echelon crossover
                ## selection
                id1 = np.random.choice(pool1)
                id2 = np.random.choice(pool1)

                truck_chromo, motor_chromo, cost \
                    = self.crossover(self.truck_chromo_var[id1], 
                                     self.truck_chromo_var[id2],
                                     self.motor_chromo_var[id1],
                                     self.motor_chromo_var[id2])
                # print(truck_chromo, motor_chromo)

                if cost < self.fitness_var[-1]:
                    self.best_chromo[0] = truck_chromo
                    self.best_chromo[1] = motor_chromo
                    
                    # Update chromosome
                    new_pop_truck.append(truck_chromo)
                    new_pop_motor.append(motor_chromo)
                    fitness.append(cost)
                    # print("better overall cost")
                else:
                    # Mutate 
                    truck_chromo, motor_chromo, cost = \
                        self.mutate(truck_chromo, motor_chromo)

                    # Update chromosome
                    new_pop_truck.append(truck_chromo)
                    new_pop_motor.append(motor_chromo)
                    fitness.append(cost)
                # print("cost", cost)

            self.truck_chromo_var = new_pop_truck
            self.truck_chromo_var.extend(good_pop_truck)
            
            self.motor_chromo_var = new_pop_motor
            self.motor_chromo_var.extend(good_pop_motor)

            fitness.extend(self.fitness_var[self.pop_div:])
            self.fitness_var = fitness.copy()

            self.fitness_var,\
                 self.truck_chromo_var,\
                     self.motor_chromo_var\
                          = [list(t) for t in zip(*sorted(zip(self.fitness_var, self.truck_chromo_var, self.motor_chromo_var), key=lambda x: x[0], reverse=True))]
        return self.best_chromo, self.fitness_var[-1]

    def mutate(self, truck_chromo, motor_chromo):
        if np.random.rand() < self.p_selection:
            motor_chromo = self.satellite_mutate(truck_chromo, motor_chromo)
        
        if np.random.rand() < self.p_routing:
            # Relocation 
            truck_chromo = self.relocation_mutate(truck_chromo)
            motor_chromo = self.relocation_mutate(motor_chromo)

            # 1-interchange
            truck_chromo = self.interchange_mutate(truck_chromo)
            motor_chromo = self.interchange_mutate(motor_chromo)


        candi = decode_chromosome(truck_chromosome, motor_chromosome)
        cost = total_cost(candi, trucks, motors, depots)

        return truck_chromo, motor_chromo, cost

    def satellite_mutate(self, truck_chromo, motor_chromo):
        mutate_satellite = np.random.randint(0,len(motor_chromo))
        unused_depot = find_unused_depots(truck_chromo, motor_chromo)

        motor_chromo[mutate_satellite][0] = np.random.choice(unused_depot)
        return motor_chromo

    def relocation_mutate(self, chromo):
        mutate_motor1, mutate_motor2 = \
                np.random.randint(0, len(chromo), 2)
        mutate_cust1, mutate_cust2_pos = \
            np.random.choice(chromo[mutate_motor1][1:]),\
            np.random.randint(1, max(2, len(chromo[mutate_motor2])))
        
        chromo[mutate_motor1].remove(mutate_cust1)
        chromo[mutate_motor2].insert(mutate_cust2_pos, mutate_cust1)

        return chromo

    def interchange_mutate(self, chromo):
        mutate_motor1, mutate_motor2 = \
                np.random.randint(0, len(chromo), 2)
        mutate_cust1, mutate_cust2 = \
            np.random.randint(1, max(2, len(chromo[mutate_motor1]))),\
            np.random.randint(1, max(2, len(chromo[mutate_motor2])))
        
        chromo[mutate_motor1][mutate_cust1], \
        chromo[mutate_motor2][mutate_cust2] = \
        chromo[mutate_motor2][mutate_cust2], \
        chromo[mutate_motor1][mutate_cust1]
        return chromo