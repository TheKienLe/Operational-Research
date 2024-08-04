
from random import randint
from random import uniform
from random import triangular
from math import exp
from copy import deepcopy
import time
import csv
#import matplotlib.pyplot as plt

#...............................................................................................................
#                                          initial solution and fitness
#...............................................................................................................

def fitness_function(pop, Rep, time_window):
    fitness = []
    for index in range(len(pop)):
        answer = [[], [], []]  # Distance, delivery tardiness, maintenance cost
        fitness.append([])

        # If the gene is not calculated
        if pop[index][3] == 0:
            for r in range(Rep):  # Repeat for the given number of repetitions

                temp_gene = deepcopy(pop[index])
                loading_list = []  # Stores steps, each step contains a list of vehicles with their specified customers
                deliver_timer = []  # Stores delivery time for each customer in each vehicle route
                load_timer = [0 for i in range(len(temp_gene[1]))]  # Real-time of every vehicle

                # Dedication phase: filling loading_list
                for l in range(0, loop):
                    loading_list.append([])
                    deliver_timer.append([])
                    for j in range(0, len(temp_gene[1])):  # Select a vehicle
                        temp_veh = []
                        current_cap = deepcopy(vehicle.cap[temp_gene[1][j]])
                        for c in range(0, len(temp_gene[0])):  # Select an order
                            if temp_gene[0][c] != 0:  # If it's not scheduled
                                volume = sum(customer.wn[temp_gene[0][c]])
                                if current_cap >= volume:
                                    current_cap -= volume
                                    temp_veh.append(temp_gene[0][c])
                                    temp_gene[0][c] = 0
                                else:
                                    temp_veh.append(0)
                                    loading_list[l].append(temp_veh)
                                    break

                maint_cost = 0  # Maintenance cost
                dist_cost = 0  # Distribution cost
                Tardiness = 0

                timer_prod = [0 for i2 in range(0, pn)]
                timer_main = [0 for i2 in range(0, pn)]

                # Production and Maintenance phases
                for step in range(0, len(loading_list)):
                    for veh in range(0, len(loading_list[step])):
                        timer_temp = [0 for i3 in range(0, pn)]  # Temporary production timer
                        deliver_timer[step].append([0 for o in range(0, len(loading_list[step][veh]))])

                        # Production phase
                        for cust in loading_list[step][veh]:
                            if cust != 0:
                                for o1 in range(0, pn):
                                    timer_temp[o1] += customer.process_time[cust][o1]
                                    timer_main[o1] += customer.process_time[cust][o1]

                        # Maintenance phase
                        for t2 in range(pn):
                            fail_prob = [0 for i3 in range(0, pn)]  # System failure probability
                            if timer_main[t2] > pop[index][2][t2]:  # Maintenance cycle
                                maint_cost += main.cost[t2]
                                timer_temp[t2] += main.duration[t2]
                                timer_main[t2] = 0

                            fail_prob[t2] = 1 - exp(-(timer_main[t2] * main.fail_landa[t2]))
                            if fail_prob[t2] > uniform(0, 1):
                                maint_cost += main.fail_cost[t2]
                                timer_temp += main.fail_dur
                                timer_main[t2] = 0

                            timer_prod[t2] += timer_temp[t2]

                        load_timer[pop[index][1][veh]] = max(timer_prod)

                # Distribution phase
                for step in range(0, len(loading_list)):
                    for veh in range(0, len(loading_list[step])):
                        for cust in range(0, len(loading_list[step][veh])):
                            dest = loading_list[step][veh][cust]  # Designated customer for next move

                            if cust == 0:  # If the current location of the vehicle is the factory
                                if dest != 0:
                                    dist_cost += vehicle.Fixed_cost[pop[index][1][veh]] + (customer.distance[0][dest] / 100) * vehicle.alpha[pop[index][1][veh]] * cost.fuel * vehicle.fuel_cons[pop[index][1][veh]]
                                    load_timer[pop[index][1][veh]] += customer.distance[0][dest] / vehicle.velocity[pop[index][1][veh]]
                                    deliver_timer[step][veh][cust] += load_timer[pop[index][1][veh]]
                                    p = customer.presumed_delivery_time[pop[index][0][dest]]
                                    if time_window < (deliver_timer[step][veh][cust] - p):
                                        Tardiness += abs(p - deliver_timer[step][veh][cust]) - time_window
                            else:
                                dist_cost += (customer.distance[loading_list[step][veh][cust-1]][dest] / 100) * vehicle.alpha[pop[index][1][veh]] * cost.fuel * vehicle.fuel_cons[pop[index][1][veh]]
                                load_timer[pop[index][1][veh]] += customer.distance[loading_list[step][veh][cust-1]][dest] / vehicle.velocity[pop[index][1][veh]]
                                deliver_timer[step][veh][cust] += load_timer[pop[index][1][veh]]
                                p = customer.presumed_delivery_time[pop[index][0][dest]]
                                if dest != 0 and time_window < (deliver_timer[step][veh][cust] - p):
                                    Tardiness += abs(p - deliver_timer[step][veh][cust]) - time_window

                answer[0].append(dist_cost)
                answer[1].append(Tardiness)
                answer[2].append(maint_cost)

            pop[index][3] = answer
        else:
            answer = pop[index][3]

        fitness[index].append(index)
        fitness[index].append([])
        fitness[index][1].append(sum(answer[0]) / len(answer[0]))
        fitness[index][1].append(sum(answer[1]) / len(answer[1]))
        fitness[index][1].append(sum(answer[2]) / len(answer[2]))
        fitness[index].append([])
        fitness[index].append(0)

    return [fitness, pop]

def initial_generation(G):
    Gene = []
    for g in range(G):
        gene = [[0], [], [randint(main.min_rec, main.max_rec) for i3 in range(pn)], 0]
        temp1 = [i for i in range(1, customer_number + 1)]
        temp2 = [vehicle.n[i] for i in range(len(vehicle.n))]

        for i in range(customer_number):
            a = randint(0, len(temp1) - 1)
            gene[0].append(temp1[a])
            temp1.pop(a)

        for i in range(len(temp2)):
            a = randint(0, len(temp2) - 1)
            gene[1].append(temp2[a])
            temp2.pop(a)

        Gene.append(gene)

    return Gene

#..................................................................................................................
#                                             Multi objective sorting
#..................................................................................................................

def Dominates(x, y):
    flag = False
    if x[1][0] <= y[1][0] and x[1][1] <= y[1][1] and x[1][2] <= y[1][2]:
        flag = True
    return flag

def Dominates1(x, y):
    flag = False
    if x[1][0] < y[1][0] and x[1][1] <= y[1][1] and x[1][2] <= y[1][2]:
            flag = True
    return flag

def Fronter(pop):
    F = [[] for f in range(len(pop))]

    for i in range(len(pop)):
        for j in range(i+1, len(pop)):
            if Dominates(pop[i],pop[j]):
                #print(f'{i} Dominated {j}')
                if pop[j][0] not in pop[i][2]:
                    pop[i][2].append(pop[j][0])
                    pop[j][3] += 1

            elif Dominates1(pop[j],pop[i]):
                #print(f'{j} Dominated {i}')
                if pop[i][0] not in pop[j][2]:
                    pop[j][2].append(pop[i][0])
                    pop[i][3] += 1
            #else:
             #   print(f'{i} and {j} do not dominate each other ')
        if pop[i][3] == 0:
            pop[i].append(0)
            F[0].append(pop[i])
        #print("............................................")
    #for e, sop in enumerate(pop):
     #   print("pop =", sop)

    for k in range(len(F)):
        #print("..................................")
        #print("F[k] =", F[k])
        for s in range(len(F[k])):
            #print("\n")
            #print (f'F[{k}][{s}]=, {F[k][s]}')
            for sol in F[k][s][2]:
                #print("sol =", sol)
                for i in range(len(pop)):
                    if pop[i][0] == sol:
                        #print("pop[i] =", pop[i])
                        pop[i][3] -= 1
                        #print("pop[i] =", pop[i])
                        if pop[i][3] == 0:
                            pop[i].append(0)
                            F[k+1].append(pop[i])

    F = [x for x in F if x != []]
    #print("..........................")
    #for s in range(len(F[k])):
        #print(f'F[{k}][{s}][0]= {F[k][s][0]}')
    return F

def calc_cd(input, G, PFP):

    #nObj = len(input[0][0][1])
    #pop = input[0]#copy.deepcopy(input[0])
    #for i in range(len(pop)):
     #   pop[i].append(0)
    F = input

    for k in range(len(F)):
        #print("\n")
        #temp_F_pop = []
        #print(f'len(F[{k}]) = {len(F[k])}')
        #print("len(F[k][0][1])",len(F[k][0][1]))
        for j in range(len(F[k][0][1])):
            #print(f'F[{k}] = {F[k]}')
            #print(".......................")
            L = len(F[k])
            F[k].sort(key=lambda t: t[1][j])    #sort based on objectives
            #if L > 2:
            #print("F[k] =", F[k])
            for l in range(L):
                if l == 0 or l == L-1:
                    F[k][l][-1] = float("inf")
                else:
                    #print("jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj =", j)
                    #print("(F[k][l+1][1][j] -  F[k][l-1][1][j])", F[k][l+1][1][j] -  F[k][l-1][1][j])
                    #print("(F[k][0][1][j] -  F[k][-1][1][j]) =", F[k][0][1][j] -  F[k][-1][1][j])
                    if F[k][0][1][j] -  F[k][-1][1][j] > 0:
                        F[k][l][-1] += abs(F[k][l+1][1][j] -  F[k][l-1][1][j])/ abs(F[k][0][1][j] -  F[k][-1][1][j])
                    else:
                        F[k][l][-1] += abs(F[k][l+1][1][j] -  F[k][l-1][1][j])/ (abs(F[k][0][1][j] -  F[k][-1][1][j]) + 1)

                    #print(f'F[{k}][{l}][-1] =', F[k][l][-1])
            #else:
            #   for l in range(L):
            #      F[k][l][-1] = float("inf")
        F[k].sort(key=lambda t: t[-1], reverse=True)
        #print(f'F[{k}] = {F[k]}')

    F0 = []
    counter_G = 0

    for f0 in range(len(F)):
        F0.append([])
        counter_front = 0
        for f00 in range(len(F[f0])):
            if counter_G < G:
                if counter_front < PFP:
                    counter_G += 1
                    counter_front += 1
                    F0[f0].append(F[f0][f00])
            else:
                break

    F = F0

    return F

#.................................................................................................................
#                                                 NSGA-II Operators
#.................................................................................................................

def Xover1(x1, x2):
    #print("x1 =", x1)
    #print("\n")
    L0 = len(x1[0])
    L1 = len(x1[1])

    alpha =  randint(2, L0)
    #print("L0 =",L0)
    a1 = [x2[0][i] for i in range(alpha)]
    a2 = []
    for j in range(L0):
        if x1[0][j] not in a1:
            a2.append(x1[0][j])

    y1 = [a1 + a2, x1[1] , [(x1[2][i3]+x2[2][i3])/2 for i3 in range(pn)] ,0]
    #print("y1 =", y1)
    #print(len(y1[0]))
    b1 = [x1[0][i] for i in range(alpha)]
    b2 = []
    for j in range(L0):
        if x2[0][j] not in b1:
            b2.append(x2[0][j])
    y2 = [b1 + b2, x2[1], [(x1[2][i3]+x2[2][i3])/2 for i3 in range(pn)],0]
    #print("y2 =", y2)
    #print(len(y2[0]))
    #print("\n")


    beta =  randint(1, len(x1[1]))
    #print("L1 =",L1)
    a1 = [x2[1][i] for i in range(beta)]
    a2 = []
    for j in range(L1):
        if x1[1][j] not in a1:
            a2.append(x1[1][j])
    z1 = [x2[0], a1 + a2, x2[2],0]
    #print("z1 =", z1)
    #print(len(z1[1]))
    b1 = [x1[1][i] for i in range(beta)]
    b2 = []
    for j in range(L1):
        if x2[1][j] not in b1:
            b2.append(x2[1][j] )
    z2 = [x1[0], b1 + b2, x1[2], 0]

    return y1, y2, z1, z2

def Mutate_Customer_list(x1):
    #print("x1 =", x1)
    #print("chromosome [x1]", chromosome[x1])
    L0 = len(x1[0])
    a = randint(0, L0 -1)
    a1 = x1[0][a]

    b = randint(0, L0 -1)
    while b == a:
        b = randint(0, L0 -1)
    b1 = x1[0][b]
    y = deepcopy(x1)
    y[0][a] = b1
    y[0][b] = a1
    y.append(0)                 #?????????????????????????????????????????????????????

    return y

def Mutate_Veh_list(x1):

    #print("x1 =", x1)
    #print("chromosome [x1]", chromosome[x1])
    L1 = len(x1[1])
    a =  randint(0, L1 -1)
    a1 = x1[1][a]

    b =  randint(0, L1 -1)
    while b == a:
        b =  randint(0, L1 -1)
    b1 = x1[1][b]
    y = deepcopy(x1)
    y[1][a] = b1
    y[1][b] = a1
    y.append(0)

    return y

def Mutate_maint_list(x1):

    #print("x1 =", x1)
    #print("chromosome [x1]", chromosome[x1])
    y = deepcopy(x1)

    for a in range(len(y[2])):
        uni = uniform(0, 1)
        if uni > 0.5:
            y[2][a] += 1
        else:
            y[2][a] -= 1
    y.append(0)

    return y

#.................................................................................................................
#                                               Data input and order
#.................................................................................................................

def Brain (G, PFP, Rep, MaxIt, PXover, PMutate, time_window):
    Gene = initial_generation(G)
    Res = fitness_function(Gene, Rep, time_window)

    Gene = Res[1]
    NDsorted_population = calc_cd(Fronter(Res[0]), G, PFP)

    for It in range(MaxIt):
        NXover = int(PXover * G/4)
        for X in range(NXover):
            x =  randint(0, int(G)/2)
            y =  randint(0, int(G)/2)
            while x == y:
                y =  randint(0, int(G)/2)
            for s in Xover1(Gene[x], Gene[y]):
                Gene.append(s)
                #Gene0[len(Gene0)] = s
                #print("sssssssssssssssss = ", s)

        NMutate = int(PMutate * G/3)
        for Y in range(NMutate):
            m = uniform(0, 1)
            if m > 0.05:
                x = randint(0, int(G/2))
                Gene.append(Mutate_Customer_list(Gene[x]))
                #Gene0[len(Gene0)] = Mutate_Customer_list(keey[x])
                Gene.append(Mutate_Veh_list(Gene[x]))
                #Gene0[len(Gene0)] = Mutate_Veh_list(keey[x])
                Gene.append(Mutate_maint_list(Gene[x]))
                #Gene0[len(Gene0)] = Mutate_maint_list(keey[x])

        Res = fitness_function(Gene, Rep, time_window)
        Gene = Res[1]
        NDsorted_population = calc_cd(Fronter(Res[0]), G, PFP)
        #print("Best Found Solution so far =",NDsorted_population[0][0][0], "  dist_cost =", NDsorted_population[0][0][1][0], "  tardiness =", NDsorted_population[0][0][1][1], "  maint_cost =",NDsorted_population[0][0][1][2] )

        Gene0 = []
        g = 0
        for f, front in enumerate (NDsorted_population):
            for sol in front:
                if g < G:
                    Gene0.append(Gene[sol[0]])
                    sol[0] = g
                    g += 1
        Gene = Gene0
        if It == MaxIt-1:
            for i in range(len(NDsorted_population[0])):
                print("Front ---> NDsorted_population[0][{i}]",  NDsorted_population[0][i][1])
            #ddd = fitness_function(Gene[0])

        #if It in (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99):
        #   print("\n")
        #  print(" It ", It)
        # for i in range(len(NDsorted_population[0])):
        #    print("Front ---> NDsorted_population[0][i]",  NDsorted_population[0][i][1])
    #print("maintenance periods=                             ", Gene[NDsorted_population[0][i][0]][2])
    #print("Len Front 0 =", len(NDsorted_population[0])) #, "        dist_cost, delay time, maint_cost =",NDsorted_population[0][0][1])
    #print("Front 0 solution 0 =", NDsorted_population[0][0][0], "        dist_cost, delay time, maint_cost =",NDsorted_population[0][0][1])
    #print("Front 0 solution 1 =", NDsorted_population[0][1])
    #for solu in NDsorted_population[0]:
    #   print("solu", solu[1])
    #  print("Chromo", Gene[solu[0]])
    # print("\n")

    if True:
        F_Star = [10000000000, 10000000000, 10000000000]  # Results of best solution
        for star in NDsorted_population[0]:
            # print("star", star)
            if star[1][0] < F_Star[0]:
                F_Star[0] = star[1][0]
            if star[1][1] < F_Star[1]:
                F_Star[1] = star[1][1]
            if star[1][2] < F_Star[2]:
                F_Star[2] = star[1][2]
        f1min = F_Star[0]
        f2min = F_Star[1]
        f3min = F_Star[2]
        f0 = []
        f1 = []
        f2 = []
        dsp = []
        for i in range(len(NDsorted_population[0])):
            f0.append(NDsorted_population[0][i][1][0])
            f1.append(NDsorted_population[0][i][1][1])
            f2.append(NDsorted_population[0][i][1][2])
            temp_dsp = []
            for j in range(len(NDsorted_population[0])):
                if i != j:
                    temp_dsp.append(abs(NDsorted_population[0][i][1][1] - NDsorted_population[0][j][1][1]) + abs(
                        NDsorted_population[0][i][1][0] - NDsorted_population[0][j][1][0]) + abs(
                        NDsorted_population[0][i][1][2] - NDsorted_population[0][j][1][2]))

            dsp.append(min(temp_dsp))

        RNI = (len(NDsorted_population[0]) / (G*MaxIt))
        print("RNI =", RNI)

        MID = (sum(f0) - (len(f0)) * min(f0) + sum(f1) - (len(f1)) * min(f1) + sum(f2) - (len(f2)) * min(f2)) ** (1.0 / 2.0) / len(NDsorted_population[0])
        print("MID =", MID)

        SP = 0
        for d in dsp:
            SP += ((1 / len(dsp)) * (d - (sum(dsp) / len(dsp))) ** 2) ** (1.0 / 2.0)
        print("SP =", SP)

        Duration = (time.time()) - Start_time
        print("Duration = %s seconds" % Duration)

        dGD = 0
        for i in range(len(NDsorted_population[0])):
            # print(NDsorted_population[0][i])
            dGD += (NDsorted_population[0][i][1][0] - f1min) ** 2
            dGD += (NDsorted_population[0][i][1][1] - f2min) ** 2
            dGD += (NDsorted_population[0][i][1][2] - f3min) ** 2
        GD = (1 / len(NDsorted_population[0])) * ((dGD) ** (1 / 2))
        print("GD =", GD)

        dRV = []
        for i in range(len(NDsorted_population[0])):
            dRV.append(min(abs(NDsorted_population[0][i][1][0] - f1min), abs(NDsorted_population[0][i][1][1] - f2min),abs(NDsorted_population[0][i][1][2] - f3min)))
        dRV_bar = sum(dRV) / len(dRV)
        RV_sigma = sum([(dRV_bar - drv) ** 2 for drv in dRV])
        RV = ((1 / (len(dRV) - 1)) * RV_sigma) ** (1 / 2)
        print("RV =", RV)

        MD = ((1 / 2) * (sum(f0) / (max(f0) - min(f0)) + (sum(f1) / (max(f1) - min(f1))) + (sum(f2) / (max(f2) - min(f2))) )) ** (1.0 / 2.0)
        print("MD =", MD)

        print("WM =", (RNI + 2 * MID + 2 * SP + 2 * MD) / 7)

#...................................................................................................................
#
#..................................................................................................................


#for customer_number in (10, 13, 15, 17, 21, 25, 27, 29, 30, 35, 40, 43, 47, 52, 58, 63, 66, 69, 70):
#for repetition in range(10):



for scenario in range(10, 20):

    Start_time = time.time()
    MaxIt = 100
    G = 100                                           # population number
    PXover = 0.5
    PMutate = 0.05
    PFP = 20                                            # Pareto Front Population

    Rep = 10                                            # Repetition number of the simulation
    a2 = 0.5
    b2 = 0.5
    c2 = 0.5

    PIs = [
           [20, [1, 1, 1]],
           [25, [2, 1, 2]],
           [30, [2, 1, 2]],
           [35, [2, 2, 1]],
           [40, [2, 2, 1]],
           [40, [2, 2, 3]],
           [45, [2, 3, 2]],
           [50, [3, 2, 2]],
           [55, [2, 3, 2]],
           [55, [4, 4, 2]],
           [60, [4, 4, 2]],
           [65, [4, 4, 2]],
           [70, [4, 2, 4]],
           [75, [4, 2, 4]],
           [80, [4, 2, 4]],
           [85, [4, 2, 4]],
           [90, [4, 4, 4]],
           [95, [4, 4, 4]],
           [100, [4, 4, 4]],
           [100, [5, 5, 5]],
    ]

    CDs=[                                   # Customer Demands
        [[1, 3, 8], [1, 3, 5], [1, 2, 3]],
        [[2, 4, 9], [2, 4, 6], [2, 3, 4]],
        [[3, 5, 9], [3, 5, 7], [3, 4, 5]],
        [[4, 6, 9], [4, 6, 8], [4, 5, 6]],
        [[5, 7, 10], [5, 7, 9], [5, 6, 7]],
        [[1, 3, 8], [1, 3, 5], [1, 2, 3]],
        [[2, 4, 9], [2, 4, 6], [2, 3, 4]],
        [[3, 5, 9], [3, 5, 7], [3, 4, 5]],
        [[4, 6, 9], [4, 6, 8], [4, 5, 6]],
        [[5, 7, 10], [5, 7, 9], [5, 6, 7]],
        [[1, 3, 8], [1, 3, 5], [1, 2, 3]],
        [[2, 4, 9], [2, 4, 6], [2, 3, 4]],
        [[3, 5, 9], [3, 5, 7], [3, 4, 5]],
        [[4, 6, 9], [4, 6, 8], [4, 5, 6]],
        [[5, 7, 10], [5, 7, 9], [5, 6, 7]],
        [[1, 3, 8], [1, 3, 5], [1, 2, 3]],
        [[2, 4, 9], [2, 4, 6], [2, 3, 4]],
        [[3, 5, 9], [3, 5, 7], [3, 4, 5]],
        [[4, 6, 9], [4, 6, 8], [4, 5, 6]],
        [[5, 7, 10], [5, 7, 9], [5, 6, 7]],
    ]

    PMs=[                                   # Preventive Maintenance
        [[2, 4, 7], [2, 4, 6], [2, 3, 4]],
        [[3, 4, 7], [2, 5, 6], [2, 3, 5]],
        [[3, 4, 8], [3, 5, 6], [3, 4, 5]],
        [[4, 5, 9], [3, 5, 7], [3, 4, 5]],
        [[5, 5, 9], [4, 6, 8], [4, 5, 6]],
        [[2, 4, 7], [2, 4, 6], [2, 3, 4]],
        [[3, 4, 7], [2, 5, 6], [2, 3, 5]],
        [[3, 4, 8], [3, 5, 6], [3, 4, 5]],
        [[4, 5, 9], [3, 5, 7], [3, 4, 5]],
        [[5, 5, 9], [4, 6, 8], [4, 5, 6]],
        [[2, 4, 7], [2, 4, 6], [2, 3, 4]],
        [[3, 4, 7], [2, 5, 6], [2, 3, 5]],
        [[3, 4, 8], [3, 5, 6], [3, 4, 5]],
        [[4, 5, 9], [3, 5, 7], [3, 4, 5]],
        [[5, 5, 9], [4, 6, 8], [4, 5, 6]],
        [[2, 4, 7], [2, 4, 6], [2, 3, 4]],
        [[3, 4, 7], [2, 5, 6], [2, 3, 5]],
        [[3, 4, 8], [3, 5, 6], [3, 4, 5]],
        [[4, 5, 9], [3, 5, 7], [3, 4, 5]],
        [[5, 5, 9], [4, 6, 8], [4, 5, 6]],
    ]

    CMs=[                                   # Corective Maintenance
        [[10, 12, 18], [10, 12, 15], [10, 13, 20]],
        [[12, 14, 19], [12, 14, 16], [12, 15, 20]],
        [[13, 17, 22], [12, 17, 18], [14, 17, 24]],
        [[15, 20, 28], [15, 20, 25], [15, 25, 30]],
        [[15, 25, 29], [17, 25, 28], [17, 25, 30]],
        [[10, 12, 18], [10, 12, 15], [10, 13, 20]],
        [[12, 14, 19], [12, 14, 16], [12, 15, 20]],
        [[13, 17, 22], [12, 17, 18], [14, 17, 24]],
        [[15, 20, 28], [15, 20, 25], [15, 25, 30]],
        [[15, 25, 29], [17, 25, 28], [17, 25, 30]],
        [[10, 12, 18], [10, 12, 15], [10, 13, 20]],
        [[12, 14, 19], [12, 14, 16], [12, 15, 20]],
        [[13, 17, 22], [12, 17, 18], [14, 17, 24]],
        [[15, 20, 28], [15, 20, 25], [15, 25, 30]],
        [[15, 25, 29], [17, 25, 28], [17, 25, 30]],
        [[10, 12, 18], [10, 12, 15], [10, 13, 20]],
        [[12, 14, 19], [12, 14, 16], [12, 15, 20]],
        [[13, 17, 22], [12, 17, 18], [14, 17, 24]],
        [[15, 20, 28], [15, 20, 25], [15, 25, 30]],
        [[15, 25, 29], [17, 25, 28], [17, 25, 30]],
    ]




    #customer_number in (70, 7):
    print("\n")
    #print("customer_number", customer_number)
    #customer_number = 70
    pn = 3                                            # number of products
    prd_r = [72, 75, 50]                              # production rate


    #customer_number = 70
    customer_number = PIs[scenario][0]
    print("customer_number =", customer_number)
    print("vehicles        =", PIs[scenario][1])
    class customer:
        n = [0]
        wn = [0]                                      # required product for every customer
        presumed_delivery_time = [0]
        process_time = [0]
        distance = []                                 # distance between customers

    total_wn = 0
    total_p = 0
    total_t = 0
    for c in range (customer_number):
        customer.n.append(c + 1)
        #customer.wn.append([int(triangular(1, 8, 3)), int(triangular(0, 5, 3)), int(triangular(0, 3, 2))])
        customer.wn.append([int(triangular(CDs[scenario][0][1],         # deterministic
                                           CDs[scenario][0][1],
                                           CDs[scenario][0][1])),
                            int(triangular(CDs[scenario][1][1],
                                           CDs[scenario][1][1],
                                           CDs[scenario][1][1])),
                            int(triangular(CDs[scenario][2][1],
                                           CDs[scenario][2][1],
                                           CDs[scenario][2][1]))])

    #    customer.wn.append([int(triangular(CDs[scenario][0][0],        # stochastic
    #                                       CDs[scenario][0][1],
    #                                       CDs[scenario][0][2])),
    #                        int(triangular(CDs[scenario][1][0],
    #                                       CDs[scenario][1][1],
    #                                       CDs[scenario][1][2])),
    #                        int(triangular(CDs[scenario][2][0],
    #                                       CDs[scenario][2][1],
    #                                       CDs[scenario][2][2]))])
        #customer.wn.append([8, 5, 3])
        #customer.wn.append([int(triangular(1, 8, 3)), int(triangular(0, 5, 3)), int(triangular(0, 3, 2))])
        customer.process_time.append([(customer.wn[c+1][q]*24)/prd_r[q] for q in range(pn)])
        total_wn += sum(customer.wn[c + 1])                      # Total ordered products
        total_p += max(customer.process_time[c + 1])             # Total process times

    time_window = 24
    a = 24
    b = 120
    for c in range(customer_number):
        customer.presumed_delivery_time.append(uniform(a, b))

    with open('JKMC.csv') as file:
        reader = csv.reader(file)
        cities = []
        counti = 0
        for row in reader:
            counti += 1
            countj = 0
            if counti > 1:
                customer.distance.append([])
                for item in row:
                    countj += 1
                    if countj == 1:
                        cities.append(item)
                    if countj > 1:
                        customer.distance[counti-2].append(int(item))
    #for x in customer.distance:
     #   print("distances", x)
    class info:
        #type_Number = [10, 7, 5]
        type_Number = PIs[scenario][1]
        cap = [18, 8.5, 7]
        fuel_cons = [17, 15, 12]
        velocity = [70, 80, 90]
        alpha = [(1.45, 1.35), (1.35, 1.3), (1.3, 1.2)]
        Fixed_cost = [5, 4, 3]

    total_veh = sum(info.type_Number)
    total_cap = sum([info.type_Number[i]*info.cap[i] for i in range(len(info.type_Number))])

    class vehicle:
        n = [i for i in range(total_veh)]
        cap = []
        fuel_cons = []
        velocity = []
        alpha = []
        Fixed_cost = []

    for i in range(len(info.type_Number)):
        for j in range(info.type_Number[i]):
            vehicle.cap.append(info.cap[i])
            vehicle.fuel_cons.append(info.fuel_cons[i])
            vehicle.velocity.append(info.velocity[i])
            vehicle.alpha.append(uniform(info.alpha[i][0], info.alpha[i][1]))
            vehicle.Fixed_cost.append(info.Fixed_cost[i])

    class main:
        duration = [int(triangular(PMs[scenario][0][1],         # Deterministic
                                   PMs[scenario][0][1],
                                   PMs[scenario][0][1])),
                    int(triangular(PMs[scenario][1][1],
                                   PMs[scenario][1][1],
                                   PMs[scenario][1][1])),
                    int(triangular(PMs[scenario][2][1],
                                   PMs[scenario][2][1],
                                   PMs[scenario][2][1]))]
        cost = [10, 8, 9]
        min_rec = 5             # minimum period for maintenance recommended from providers of production line
        max_rec = 35            # maximum period for maintenance recommended from providers of production line
        fail_landa = [0.0007, 0.0005, 0.0003]
        fail_dur = [int(triangular(CMs[scenario][0][1],         # deterministic
                                   CMs[scenario][0][1],
                                   CMs[scenario][0][1])),
                    int(triangular(CMs[scenario][1][1],
                                   CMs[scenario][1][1],
                                   CMs[scenario][1][1])),
                    int(triangular(CMs[scenario][2][1],
                                   CMs[scenario][2][1],
                                   CMs[scenario][2][1]))]

        fail_cost = [80, 100, 100] #[int(triangular(52, 80, 160)), int(triangular(58, 100, 170)), int(triangular(65, 100, 180))]

    class cost:
        fuel = 0.0003

    loop = int(total_wn / total_cap) + 2
    #print("loop", loop)


    Brain (G, PFP, Rep, MaxIt, PXover, PMutate, time_window)

    #........................................Validation....................................
    #Gene = initial_generation(G)
    #Res = fitness_function(Gene, Rep, time_window)