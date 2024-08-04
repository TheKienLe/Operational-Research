import numpy as np
import pandas as pd
from geopy.distance import geodesic
import classes as cls
import itertools as it
import scipy.spatial.distance as ssd
from copy import deepcopy
import math
import time
from visual_vrp import plot_2e_vrp


def read_data(name):
    f = open(name,'r')
    lines = f.readlines()
    filedata = cls.rawdata
    #truck data
    data1 = lines[2].split(',')
    filedata.number_truck = int(data1[0])
    filedata.truck_capacity = int(data1[1])
    filedata.truck_cost_distance = int(data1[2])
    filedata.truck_fixed_cost = int(data1[3])
    

    #motobike data
    data2 = lines[5].split(',')
    filedata.max_motor_per_depot = int(data2[0])
    filedata.number_motor = int(data2[1])
    filedata.motor_capacity = int(data2[2])
    filedata.motor_cost_distance = int(data2[3])
    filedata.motor_fixed_cost = int(data2[4])
    
    #depot and warehouse data
    data3 = lines[8].split('   ')
    depot_xcoord = []
    depot_ycoord = []
    for i in range(0, len(data3)):
        if i == 0:
            temp = data3[i].split(',')
            filedata.warehouse_xcoord = int(temp[0])
            filedata.warehouse_ycoord = int(temp[1])
        else:
            temp = data3[i].split(',')
            depot_xcoord.append(int(temp[0]))
            depot_ycoord.append(int(temp[1]))
            filedata.depot_xcoord = depot_xcoord
            filedata.depot_ycoord = depot_ycoord

    #customer data
    data4 = lines[11].split('   ')
    customer_xcoord = []
    customer_ycoord = []
    customer_demand = []
    for i in data4:
        temp = i.split(',')
        customer_xcoord.append(int(temp[0]))
        customer_ycoord.append(int(temp[1]))
        customer_demand.append(int(temp[2]))
        filedata.customer_xcoord = customer_xcoord
        filedata.customer_ycoord = customer_ycoord
        filedata.customer_demand = customer_demand
    return filedata

def create_classes_customer(rawdata):
    customers =[]
    for i in range(0, len(rawdata.customer_xcoord)):
        customers.append(cls.customer())
    for j in range(0, len(rawdata.customer_xcoord)):
        customers[j].id = j
        customers[j].xcoord = rawdata.customer_xcoord[j]
        customers[j].ycoord = rawdata.customer_ycoord[j]
        customers[j].opentime = 0
        customers[j].closetime = 144
        customers[j].demand = rawdata.customer_demand[j]
        customers[j].servicetime = rawdata.customer_demand[j]*1
        customers[j].assigned = 0
        customers[j].penalty = 10
    return customers


def create_classes_depot(rawdata):
    depots = []
    for i in range(0, len(rawdata.depot_xcoord)):
        depots.append(cls.depot())
    for j in range(0, len(rawdata.depot_xcoord)):
        depots[j].id = j + len(rawdata.customer_xcoord)
        depots[j].xcoord = rawdata.depot_xcoord[j]
        depots[j].ycoord = rawdata.depot_ycoord[j]
        depots[j].opentime = 0
        depots[j].closetime = 144
        depots[j].penalty = 10
    return depots

def create_classes_warehouse(rawdata):
    warehouse = cls.warehouse()
    warehouse.id = len(rawdata.customer_xcoord) + len(rawdata.depot_xcoord)
    warehouse.xcoord = rawdata.warehouse_xcoord
    warehouse.ycoord = rawdata.warehouse_ycoord
    return warehouse

def create_classes_truck(rawdata):
    trucks = []
    for i in range(0, rawdata.number_truck):
        trucks.append(cls.truck())
    for i in range(0, rawdata.number_truck):
        trucks[i].id = i
        trucks[i].type = 1
        trucks[i].capacity = rawdata.truck_capacity
        trucks[i].current_load = 0
        trucks[i].complete_time = 0
        trucks[i].variable_cost = rawdata.truck_cost_distance
        trucks[i].fixed_cost = rawdata.truck_fixed_cost
        trucks[i].penalty = 10
    return trucks

def create_classes_motor(rawdata):
    motors = []
    for i in range(0, rawdata.number_motor):
        motors.append(cls.motor())
    for i in range(0, rawdata.number_motor):
        motors[i].id = i
        motors[i].type = 1
        motors[i].capacity = rawdata.motor_capacity
        motors[i].current_load = 0
        motors[i].complete_time = 0
        motors[i].variable_cost = rawdata.motor_cost_distance
        motors[i].fixed_cost = rawdata.motor_fixed_cost
        motors[i].penalty = 10
    return motors

def create_class_solution(customer,motor,truck,total_cost):
    x = cls.solution()
    x.customers = customer
    x.motors = motor
    x.trucks = truck
    x.total_cost = total_cost
    return x

def dtm(customer,depot,warehouse):
    x = []
    y = []
    for i in range(0,warehouse.id + 1):
        if i < len(customer):
            x.append(customer[i].xcoord)
            y.append(customer[i].ycoord)
        elif i >= len(customer) and i < warehouse.id:
            x.append(depot[i - len(customer)].xcoord) # vì index depot tu 0
            y.append(depot[i - len(customer)].ycoord)
        else:
            x.append(warehouse.xcoord)
            y.append(warehouse.ycoord)
            xy = np.column_stack((x,y))
            dtm = ssd.cdist(xy, xy, "euclidean")
    return dtm

def initial_assign_customer_to_depot(customers, depots, dtms):
    order_dist_customer_to_depot = {}
    for i in np.random.permutation(len(customers)):
        value_arr = [] 
        
        for j in np.random.permutation(\
        range(len(customers),len(customers) + len(depots))):
           value_arr.append(dtms[j][i])

        
        order_dist_customer_to_depot[i] = np.argsort(value_arr) + len(customers)
        ind = order_dist_customer_to_depot[i][0]
        customers[i].depot = ind
        depots[ind - len(customers)].total_demand += customers[i].demand
    return customers, depots, order_dist_customer_to_depot 

def dist_depot_to_customer(customer, depot, dtm):
    order_dist_depot_to_customer = {} 
    for i in range(len(customer),len(customer) + len(depot)):
        value_arr = []
        for j in range(0, len(customer)):
            value_arr.append(dtm[j][i])
        order_dist_depot_to_customer[i] = np.argsort(value_arr)
    return order_dist_depot_to_customer

# func nhap--------------------------------
def sort_by_depot_of_customer(customer):
    customer_list = []
    depot_used = []
    for i in range(0, len(customer)):
        if customer[i].depot not in depot_used:
            depot_used.append(customer[i].depot)
    depot_used.sort()
    for k in depot_used:
        temp = []
        for j in range(0,len(customer)):
            if customer[j].depot == k:
                temp.append(customer[j].id)
        customer_list.append(temp)
    return customer_list
#-------------------------------------------

def list_depot_used(customer):
    depot_list = []
    for i in range(0, len(customer)):
        if customer[i].depot not in depot_list:
            depot_list.append(customer[i].depot)
    depot_list.sort()
    return depot_list

def list_truck_used(customer):
    truck_list = []
    for i in range(0, len(customer)):
        if customer[i].truck not in truck_list:
            truck_list.append(customer[i].truck)
    truck_list.sort()
    return truck_list

def list_motor_used(customer):
    motor_list = []
    for i in range(0, len(customer)):
        if customer[i].motor not in motor_list:
            motor_list.append(customer[i].motor)
    motor_list.sort()
    return motor_list

def list_position_in_truck_used(customer):
    position_in_truck_list = []
    for i in range(0, len(customer)):
        if customer[i].position_in_truck not in position_in_truck_list:
            position_in_truck_list.append(customer[i].position_in_truck)
    position_in_truck_list.sort()
    return position_in_truck_list

def list_position_in_motor_used(customer):
    position_in_motor_list = []
    for i in range(0, len(customer)):
        if customer[i].position_in_motor not in position_in_motor_list:
            position_in_motor_list.append(customer[i].position_in_motor)
    position_in_motor_list.sort()
    return position_in_motor_list


def filter_by_depot(customer, depot_id):
    temp = []
    for i in range(0, len(customer)):
        if customer[i].depot == depot_id:
            temp.append(customer[i])
    return temp

def filter_by_truck(customer, truck_id):
    temp = []
    for i in range(0, len(customer)):
        if customer[i].truck == truck_id:
            temp.append(customer[i])
    return temp

def filter_by_motor(customer, motor_id):
    temp = []
    for i in range(0, len(customer)):
        if customer[i].motor == motor_id:
            temp.append(customer[i])
    return temp

def filter_motor_to_truck(customer, motor_id):
    for i in range(0, len(customer)):
        if customer[i].motor == motor_id:
            return customer[i].truck

def filter_by_position_in_truck(customer, k):
    temp = []
    for i in range(0, len(customer)):
        if customer[i].position_in_truck == k:
            temp.append(customer[i])
    return temp

def filter_by_position_in_motor(customer, k):
    temp = []
    for i in range(0, len(customer)):
        if customer[i].position_in_motor == k:
            temp.append(customer[i])
    return temp

def filter_by_assigned(customer, assigned):
    temp = []
    for i in range(0, len(customer)):
        if customer[i].assigned == assigned:
            temp.append(customer[i])
    return temp

def filter_depot_in_truck_chromo(truck_chromosome, depot_id):
    # print(depot_id)
    truck_pos, truck_id = 0, 0
    for truck_id, route in enumerate(truck_chromosome):
        try:
            truck_pos = route.index(depot_id)
            return truck_pos, truck_id
        except: 
            pass
    return truck_pos, truck_id
 

def show_id(arr):
    result_arr = []
    for i in range(0, len(arr)):
        result_arr.append(arr[i].id)
    return result_arr

def check_arr_all_negative(arr):
    # flag = 9999
    # temp = []
    # for i in arr:
    #     if i <= 0:
    #         temp.append(0)
    #     else:
    #         temp.append(1)
    # total = sum(temp)

    # if total > 0:
    #     flag = 0
    # else:
    #     flag = 1
    
    arr = np.array(arr)
    flag = 1 if sum(arr < 0) > 0 else 0
    return flag

def choose_index_min_value(arr):
    min_value = 99999
    min_index = 99999
    for i in range(0, len(arr)):
        if arr[i] < min_value and arr[i] >= 0:
            min_index = i
            min_value = arr[i]
    return min_index

def get_cost_truck(truck, solution):
    distance_cost = 0
    time_cost = 0
    cost = 0
    finishing_time_depot = []


def total_demand_at_depot(customer, depot_id):
    total_demand = 0
    customer_by_depot = filter_by_depot(customer, depot_id)
    for cust in customer_by_depot:
        total_demand += cust.demand
    return total_demand

def total_demand_at_motor(customer, motor_id):
    total_demand = 0
    customer_by_motor = filter_by_motor(customer, motor_id)
    for cust in customer_by_motor:
        total_demand += cust.demand
    return total_demand

def total_demand_at_truck(customer, truck_id):
    total_demand = 0
    customer_by_depot = filter_by_truck(customer, truck_id)
    for cust in customer_by_depot:
        total_demand += cust.demand
    return total_demand

def routing_motor(customers, motors, depots):
    motor_id = 0
    motor_pos = 1
    for depot_id in np.random.permutation(len(depots)):
        customer_by_depot = filter_by_depot(customers, depots[depot_id].id)
        unassign_customer = filter_by_assigned(customer_by_depot, False)
        unassign_customer_id = show_id(unassign_customer)
        #Best fit######
        for id in unassign_customer_id:
                # print("before", customers[id].depot)
            if motors[motor_id].current_load + customers[id].demand \
                  <= motors[motor_id].capacity:
                motors[motor_id].current_load += customers[id].demand
                customers[id].motor = motor_id
                customers[id].position_in_motor = motor_pos
                customers[id].assigned = True
                customers[id].depot = depot_id + len(customers)
                motor_pos +=1
            else:
                motor_id += 1
                motor_pos = 1
                motors[motor_id].current_load += customers[id].demand
                customers[id].motor = motor_id
                customers[id].position_in_motor = motor_pos
                customers[id].assigned = True
                customers[id].depot = depot_id + len(customers)
                motor_pos += 1
    # for motor in motors:
        # print(motor.depot, motor.route)  
    return customers, motors

def assign_depot(customer, depot, motor, rawdata):
    cummulative_load = 0
    index_depot = 0
    motor_used = list_motor_used(customer)
    for i in motor_used:
        cummulative_load += motor[i].current_load
        customer_by_motor = filter_by_motor(customer, i)
        if cummulative_load <= rawdata.truck_capacity:
            for cust in customer_by_motor:
                cust.depot = depot[index_depot].id
        else:
            index_depot += 1
            cummulative_load = 0
            for cust in customer_by_motor:
                cust.depot = depot[index_depot].id

def routing_truck(customers, trucks, depots):
    truck_id = 0
    truck_pos = 1
    # depot_used = list_depot_used(customers)
    for depot_id in np.random.permutation(len(depots)):
        total_demand = total_demand_at_depot(customers, depots[depot_id].id)
        if total_demand + trucks[truck_id].current_load \
            <= trucks[truck_id].capacity:
    
            customer_by_depot = filter_by_depot(customers, depots[depot_id].id)
            # assign all customers in the depot for that truck
            for cust in customer_by_depot:
                cust.truck = truck_id
                cust.position_in_truck = truck_pos
            trucks[truck_id].current_load += total_demand
            depots[depot_id].truck = truck_id
            truck_pos += 1

        else:
            truck_id += 1
            truck_pos = 1
            # print("else", truck_id)
            customer_by_depot = filter_by_depot(customers, depots[depot_id].id)

            # assign all customers in the depot for the next truck
            for cust in customer_by_depot:
                cust.truck = truck_id
                cust.position_in_truck = truck_pos
            trucks[truck_id].current_load  += total_demand
            depots[depot_id].truck = truck_id
            truck_pos += 1
    return customers, trucks, depots

def check_capacity(customer, vehicle_id,  eche=1):
    if eche ==1:
        truck_demand = 0
        truck_demand = total_demand_at_truck(customer, vehicle_id)
        # print(truck_demand)
        return truck_demand > truck_capacity
    else:
        truck_id = filter_motor_to_truck(customer, vehicle_id)
        truck_demand = total_demand_at_truck(customer, truck_id)
        motor_demand = total_demand_at_motor(customer, vehicle_id)
        
    return (motor_demand > motor_capacity) or (truck_demand > truck_capacity)


def check_truck_capacity(trucks, cust1, cust2):
    truck_id_cust1 = cust1.truck
    truck_id_cust2 = cust2.truck

    load_truck1 = (trucks[truck_id_cust1].current_load\
          - cust1.demand + cust2.demand) \
            <= trucks[truck_id_cust1].capacity
    load_truck2 =  (trucks[truck_id_cust2].current_load\
          + cust1.demand - cust2.demand) \
          <= trucks[truck_id_cust2].capacity
    check = load_truck1 and load_truck2
    return check

def check_motor_capacity(motors, cust1, cust2):
    motor_id_cust1 = cust1.motor
    motor_id_cust2 = cust2.motor

    load_motor1 = (motors[motor_id_cust1].current_load\
          - cust1.demand + cust2.demand) \
            <= motors[motor_id_cust1].capacity
    load_motor2 =  (motors[motor_id_cust2].current_load\
          + cust1.demand - cust2.demand) \
          <= motors[motor_id_cust2].capacity
    check = load_motor1 and load_motor2
    return check

def find_next_depot(customers, depot_id):
    truck_chromo, _ = encode_chromosome(customers)
    
    next_depot = None
    for route in truck_chromo:
        if (depot_id in route) and (route.index(depot_id) < len(route) -1):
            next_depot = route[route.index(depot_id) + 1] 
        
    return next_depot

def truck_cost(customer, trucks, depots):
    pre_id_truck = 0
    final_id_truck = 0
    
    total_cost_truck = 0
    truck_used = list_truck_used(customer)
    total_cost_truck += len(truck_used) * trucks[0].fixed_cost
    try:
        truck_used.remove(-1)
    except:
        pass
    
    for i in truck_used:
        customer_by_truck = filter_by_truck(customer, i)
        truckpos_used = list_position_in_truck_used(customer_by_truck)
        last_arrival_time = 0

        total_cost_truck += min(
                trucks[i].capacity - total_demand_at_truck(customer_by_truck, i),
                0) * -trucks[i].penalty
        # print("capacity_cost", total_cost_truck)

        for j in truckpos_used:

            customer_by_truck_by_truckpos = \
            filter_by_position_in_truck(customer_by_truck, j)
            
            depot_in_truckpos_used = list_depot_used(customer_by_truck_by_truckpos)
            post_id_truck = depot_in_truckpos_used[0]
            
            # variable cost
            variable_cost = dtms[pre_id_truck][post_id_truck] * trucks[i].variable_cost
            # print("variable_cost", variable_cost)
            

            # Waiting cost
            travel_time = dtms[pre_id_truck][post_id_truck] / 1e3
            unload_time = total_demand_at_depot(customer_by_truck, post_id_truck) * 1e-3
            arrival_time = last_arrival_time + travel_time + unload_time

            depot_id = post_id_truck - len(customer)
            depots[depot_id].finish_time = arrival_time
            violate_cost =  depots[depot_id].penalty \
                            * (max(depots[depot_id].opentime - arrival_time, 0) \
                               + max(arrival_time - depots[depot_id].closetime, 0))
            
            # print("waiting_time_cost", violate_cost)
            total_cost_truck += variable_cost + violate_cost
            # print("total_cost_truck", total_cost_truck)
            pre_id_truck = post_id_truck

        total_cost_truck += dtms[pre_id_truck][final_id_truck] * trucks[i].variable_cost
    return total_cost_truck, depots

def motor_cost(customer, motors, depots):
    total_cost_motor = 0
    depot_used = list_depot_used(customer)
    total_cost_motor += len(depot_used) * motors[0].fixed_cost
    try:
        depot_used.remove(-1)
    except:
        pass
    for a in depot_used:

        final_id_motor = find_next_depot(customer, a)
        customer_by_depot = filter_by_depot(customer, a)
        motor_used = list_motor_used(customer_by_depot)
        
        depot = a - len(customer)
        try:
            motor_used.remove(-1)
        except:
            pass
        for b in motor_used:
            customer_by_motor = filter_by_motor(customer_by_depot, b)
            motorpos_used = list_position_in_motor_used(customer_by_motor)
            arrival_time = depots[depot].finish_time
            pre_id_motor = a

            # Capacity violated cost
            total_cost_motor += min(
                motors[b].capacity - total_demand_at_motor(customer_by_motor, b),
                0) * -motors[b].penalty
            
            for c in motorpos_used:
                customer_by_depot_motor_pos = \
                filter_by_position_in_motor(customer_by_motor, c)
                post_id_motor = customer_by_depot_motor_pos[0].id
                
                # [3, 5 , 12] [26, 3, 11]
                # Variable cost
                variable_cost = dtms[pre_id_motor][post_id_motor] * motors[b].variable_cost
                
                
                # Waiting cost
                travel_time = dtms[pre_id_motor][post_id_motor] / 1.2e3
                unload_time = customer_by_depot_motor_pos[0].demand * 5e-3 
                arrival_time += travel_time + unload_time

                waiting_cost = customer[post_id_motor].penalty \
                            * (max(customer[post_id_motor].opentime - arrival_time, 0) \
                               + max(arrival_time - customer[post_id_motor].closetime, 0))

                total_cost_motor += variable_cost + waiting_cost
                pre_id_motor = post_id_motor

            if final_id_motor:
                total_cost_motor += dtms[pre_id_motor][final_id_motor] * motors[b].variable_cost
    return total_cost_motor

def total_cost(customer, trucks, motors, depots):
    total_cost_truck, depots = truck_cost(customer, trucks, depots)
    total_cost_motor = motor_cost(customer, motors, depots)

    total_cost = total_cost_motor + total_cost_truck
    return total_cost

def initial_solution(data_url):
    raw_data = read_data(data_url)
    customers = create_classes_customer(raw_data)
    depots = create_classes_depot(raw_data)
    warehouses = create_classes_warehouse(raw_data)
    trucks = create_classes_truck(raw_data)
    motors = create_classes_motor(raw_data)
    truck_capacity = trucks[0].capacity
    motor_capacity = motors[0].capacity

    dtms = dtm(customers, depots, warehouses)

    customers, depots, order_dist_customer_to_depot = \
        initial_assign_customer_to_depot(customers, 
                                         depots, 
                                         dtms)
    
    order_dist_depot_to_customer = dist_depot_to_customer(customers,
                                                          depots,
                                                          dtms)
    for depot_id in np.random.permutation(len(depots)):
        depot_demand = total_demand_at_depot(customers, depots[depot_id].id)
        if depot_demand > truck_capacity:
            temp_cust_id = \
                order_dist_depot_to_customer[depots[depot_id].id][-1]
                 
            
            customers[temp_cust_id].depot = \
                order_dist_customer_to_depot[temp_cust_id][-2]
           
    customers, motors = routing_motor(customers, motors, depots)
    customers, trucks, depots = routing_truck(customers, trucks, depots)
    
    update_variable = {"customers": customers, 
                       "depots": depots, 
                       "warehouses": warehouses, 
                       "trucks": trucks, 
                       "motors": motors, 
                       "truck_capacity": truck_capacity,
                       "motor_capacity": motor_capacity, 
                       "dtms": dtms}
    return update_variable

def swap_customer(cust1, cust2):
    cust1.truck, cust2.truck = cust2.truck, cust1.truck
    cust1.motor, cust2.motor = cust2.motor, cust1.motor
    cust1.position_in_truck, cust2.position_in_truck = cust2.position_in_truck, cust1.position_in_truck
    cust1.position_in_motor, cust2.position_in_motor = cust2.position_in_motor, cust1.position_in_motor
    cust1.depot, cust2.depot = cust2.depot, cust1.depot
    # return cust1, cust2

# def 

def tabu_search(solution, max_iter=100, tabu_size=35):
    def getNB_rule1(candidates, trucks=trucks, motors=motors):
        list_new_candidate =[]
        initial_candidate = deepcopy(candidates)
        for id1 in range(len(initial_candidate)-1):
            for id2 in range(id1+1, len(initial_candidate)):
                condition1 = check_motor_capacity(motors, initial_candidate[id1], initial_candidate[id2])
                condition2 = check_truck_capacity(trucks, initial_candidate[id1], initial_candidate[id2])
                if condition1 and condition2:
                    swap_customer(initial_candidate[id1], initial_candidate[id2])
                    list_new_candidate.append(initial_candidate)
                    initial_candidate = deepcopy(candidates)
        return list_new_candidate

    s0 = solution.customers
    sBest = s0
    bestCandidate = s0
    tabuList = []
    tabuList.append(s0)
    for iter in range(max_iter): 
        x = np.random.rand()
        if x <= 1:
            sNeighborhood = getNB_rule1(bestCandidate)
        for sCandidate in sNeighborhood:
            if (sCandidate not in tabuList)\
                and (total_cost(sCandidate, variable["trucks"], variable["motors"], variable["depots"]) \
                     < total_cost(bestCandidate, variable["trucks"], variable["motors"], variable["depots"])):
              
                bestCandidate = sCandidate
        if total_cost(bestCandidate, variable["trucks"], variable["motors"], variable["depots"]) \
            < total_cost(sBest, variable["trucks"], variable["motors"], variable["depots"]):
            
            sBest = bestCandidate
        tabuList.append(bestCandidate)
        if len(tabuList) > tabu_size:
            tabuList.pop(0)

    return sBest

def encode_truck_route(customers):
    truck_route = []
    for id in range(len(trucks)):
        cust_in_truck = filter_by_truck(customers, id)

        if cust_in_truck != []:
            truck_route.append([0])
            list_depot = []
            pos_in_truck = []
            for cust in cust_in_truck:
                if cust.position_in_truck not in pos_in_truck:
                    pos_in_truck.append(cust.position_in_truck)
                    list_depot.append(cust.depot)
    
            route = [depot for _, depot in sorted(zip(pos_in_truck, list_depot))]
            truck_route[-1].extend(route)
    return truck_route

def encode_motor_route(customers):
    motor_route = []
    for id in range(len(motors)):
        cust_in_motor = filter_by_motor(customers, id)
        if cust_in_motor != []:
            motor_route.append([])
            motor_route[-1].append(cust_in_motor[0].depot)
            pos_in_motor = []
            list_cust = []
            for cust in cust_in_motor:
                pos_in_motor.append(cust.position_in_motor)
                list_cust.append(cust.id)
            route = [cust for _, cust in sorted(zip(pos_in_motor, list_cust))]
            motor_route[-1].extend(route)
    return motor_route

def encode_chromosome(customers):
    # encode truck chromosome
    truck_chromosome = encode_truck_route(customers)
    
    # encode motor chromosome
    motor_chromosome = encode_motor_route(customers)
    return truck_chromosome, motor_chromosome

def sort_motor_chromosome(truck_chromosome, motor_chromosome):
    truck_chromosome_flat = [depot for route in truck_chromosome for depot in route[1:]]
    # print(truck_chromosome_flat)
    
    motor_chromosome = sorted(motor_chromosome,
                              key=lambda x:truck_chromosome_flat.index(x[0]))

    return motor_chromosome

def decode_chromosome(truck_chromosome, motor_chromosome):
    
    # for route in truck_chromosome:
    candidates = create_classes_customer(read_data(data_url))
    for motor_id, route in enumerate(motor_chromosome):
        truck_pos, truck_id = filter_depot_in_truck_chromo(truck_chromosome, route[0])
        for motor_pos, cust in enumerate(route[1:]):
            candidates[cust].depot = route[0]
            candidates[cust].position_in_motor = motor_pos+1
            candidates[cust].truck = truck_id
            candidates[cust].position_in_truck = truck_pos
            candidates[cust].motor = motor_id
    return candidates

def cheapest_insertion(unassign_customer, truck_chromosome, motor_chromosome, num_merge, echelon):
    if echelon ==1:
        VCX = truck_chromosome
        total_truck = len(truck_chromosome)
        current_truck = num_merge
        # print("cheapest_begin", VCX)
        
        if unassign_customer == []:
            candi = decode_chromosome(VCX, motor_chromosome)
            final_cost = total_cost(candi, trucks, motors, depots)

        while unassign_customer != []: 
            min_cost = float("inf")

            for cust_id, cust in enumerate(unassign_customer):
                VCX[current_truck].append(cust)
                candi = decode_chromosome(VCX, motor_chromosome)
                cost = total_cost(candi, trucks, motors, depots)
                if cost < min_cost:
                    chosen_cust = cust
                    chosen_cust_id = cust_id
                    chosen_candi = deepcopy(candi) 
                    min_cost = cost
                VCX[current_truck].pop(-1)
            VCX[current_truck].append(chosen_cust)
            # print("cheapest_middle", VCX)
            if (check_capacity(chosen_candi, current_truck, 1)) \
                and (current_truck < total_truck-1):
                VCX[current_truck].pop(-1)  
                current_truck += 1
            else:
                unassign_customer.pop(chosen_cust_id)
                final_cost = min_cost.copy()
    else:
        VCX = motor_chromosome
        total_motor = len(motor_chromosome)
        current_motor = num_merge
        
        
        if unassign_customer == []:
            candi = decode_chromosome(truck_chromosome, VCX)
            final_cost = total_cost(candi, trucks, motors, depots)

        while unassign_customer != []: 
            min_cost = float("inf")

            for cust_id, cust in enumerate(unassign_customer):
                VCX[current_motor].append(cust)
                candi = decode_chromosome(truck_chromosome, VCX)
                cost = total_cost(candi, trucks, motors, depots)
                if cost < min_cost:
                    chosen_cust = cust
                    chosen_cust_id = cust_id
                    min_cost = cost
                VCX[current_motor].pop(-1)
            VCX[current_motor].append(chosen_cust)
            

            candi = decode_chromosome(truck_chromosome, VCX)
            
            if (check_capacity(candi, current_motor, 2)) \
                and (current_motor < total_motor-1):
                VCX[current_motor].pop(-1)  
                current_motor += 1
            else:
                unassign_customer.pop(chosen_cust_id)
                final_cost = min_cost.copy()

    return VCX, final_cost

def curve(x): #sửa curve tùy theo dist phù hợ
    return  round(1000*math.e**(-0.00019*(x*0.06)))

def find_unused_depots(truck_chromo, motor_chromo):
    depots = []
    for route in truck_chromo:
        depots.extend(route[1:]) 
    
    used_depots = []
    for route in motor_chromo:
        used_depots.append(route[0])

    unused_depots = [depot for depot in depots if depot not in used_depots]
    return depots


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
    
    
if __name__ == "__main__":
    data_url = r"C:\Users\Dave\Documents\GitHub\jolie-thesis\data\Set2a_E-n33-k4-s14-22.dat"
    variable = initial_solution(data_url)
    dtms = variable["dtms"]
    trucks = variable["trucks"]
    motors = variable["motors"]
    depots = variable["depots"]
    warehouse = variable["warehouses"]
    customers = variable["customers"]
    truck_capacity = variable["truck_capacity"]
    motor_capacity = variable["motor_capacity"]


    # Initial solution 
    cost = total_cost(variable["customers"], variable["trucks"], variable["motors"], variable["depots"])
    solution = create_class_solution(variable["customers"], variable["motors"], variable["trucks"], cost)
    print("initial cost:", cost)
    truck_chromosome, motor_chromosome = encode_chromosome(variable["customers"])
    motor_chromosome = sort_motor_chromosome(truck_chromosome, motor_chromosome)

    print("truck_route", truck_chromosome, 
          "\nmotor_route", motor_chromosome)



    # # Solution from Tabu Search
    sBest = tabu_search(solution, max_iter=50, tabu_size=35)

    print("tabu cost: ",total_cost(sBest, variable["trucks"], variable["motors"], variable["depots"]))    
    truck_chromosome, motor_chromosome = encode_chromosome(variable["customers"])
    variable["customers"] = decode_chromosome(truck_chromosome, motor_chromosome)
    # for cust in variable["customers"]:
    #     print(cust)
    print("truck_route", truck_chromosome, 
          "\nmotor_route", motor_chromosome)


    # Solution from GA
    # GA_params = {"pop_size": 100,\
    #              "div_rate": 0.9,\
    #              "p_selection": 0.05,\
    #              "p_routing" : 0.005,\
    #              "max_iter": 2}
    # GA_algo = GA(GA_params)
    # solution, final_cost = GA_algo.evolve()
    # solution[1] = sort_motor_chromosome(solution[0], solution[1])
    # print("truck_route", solution[0], 
    #       "\nmotor_route", solution[1]
    #       )
    # print("ga_cost",final_cost)

    # plot_2e_vrp(warehouse, depots, customers, 
    #         truck_chromosome, motor_chromosome)