import pandas as pd
import numpy as np

def load_data_excel(file_name):
    """
    Load a VRP instance from an Excel file.

    Parameters
    ----------
    file_name : str
        The name of the Excel file containing the VRP instance data.

    Returns
    -------
    instance : dict
        A dictionary containing the VRP instance data, including parameters,
        customer details, and distance matrix.

    Examples
    --------
    >>> instance = load_instance('vrp_instance.xlsx')
    >>> instance['max_vehicle_number']
    10
    >>> instance['Number_of_customers']
    50
    >>> instance['vehicle_capacity']
    100
    >>> instance['customer_1']
    {'coordinates': {'x': 30, 'y': 50}, 
    'demand': 10, 
    'due_time': 100, 
    'ready_time': 20}
    >>> instance['sattelite_1']
    {'coordinates': {'x': 30, 'y': 50}, 
    'demand': 10, 
    'due_time': 100, 
    'ready_time': 20}
    >>> instance['distance_matrix'][0][1]
    14.142135623730951
    """

    instance = {}
    sheet_names = pd.ExcelFile(file_name).sheet_names
    for sheet_name in sheet_names:
        if sheet_name == "parameters":
            instance['max_vehicle_number'] = pd.read_excel(
                file_name, sheet_name).iloc[0, 1]
            instance['Number_of_customers'] = pd.read_excel(
                file_name, sheet_name).iloc[1, 1]
            instance['vehicle_capacity'] = pd.read_excel(
                file_name, sheet_name).iloc[2, 1]
            instance['Number_of_clusters'] = pd.read_excel(
                file_name, sheet_name).iloc[10, 1]
        elif sheet_name == "customers":
            customers = pd.read_excel(file_name, sheet_name)
            for customer_id in range(len(customers["Cus No"])):
                name = ""
                if customer_id == 0:
                    name = "depart"
                else:
                    name = f"customer_{customer_id}"
                instance[name] = {
                    "coordinates": {"x": customers['x'][customer_id], "y": customers['y'][customer_id]},
                    "demand": customers["demand"][customer_id],
                    "due_time": customers["due_time"][customer_id],
                    "ready_time": customers["ready_time"][customer_id],
                    "service_time": customers["service_time"][customer_id]}


    coor_data = pd.read_excel(file_name, "customers")[["x", "y"]]
    distance = []
    x = coor_data["x"]
    y = coor_data["y"]
    for i in range(len(coor_data)):
        row = []
        for j in range(len(coor_data)):
            if i == j:
                row.append(0)
            else:
                row.append(((x[i] - x[j])**2 + (y[i] - y[j])**2)**0.5)
        distance.append(row)

    instance["distance_matrix"] = distance
    return instance

def load_data_benchmark(file_name):
    """
    Load a VRP instance from an Excel file.

    Parameters
    ----------
    file_name : str
        The name of the Excel file containing the VRP instance data.

    Returns
    -------
    instance : dict
        A dictionary containing the VRP instance data, including parameters,
        customer details, and distance matrix.

    Examples
    --------
    >>> instance = load_instance('vrp_instance.xlsx')
    >>> instance['max_vehicle_number']
    10
    >>> instance['Number_of_customers']
    50
    >>> instance['vehicle_capacity']
    100
    >>> instance['customer_1']
    {'coordinates': {'x': 30, 'y': 50}, 
    'demand': 10, 
    'due_time': 100, 
    'ready_time': 20}
    >>> instance['sattelite_1']
    {'coordinates': {'x': 30, 'y': 50}, 
    'demand': 10, 
    'due_time': 100, 
    'ready_time': 20}
    >>> instance['distance_matrix'][0][1]
    14.142135623730951
    """
    f = open(file_name,'r')
    lines = f.readlines()
    instance = {}

    #truck data
    data_truck = lines[2].split(',')
    instance['truck_num'] = int(data_truck[0])
    instance['truck_cap'] = int(data_truck[1])
    instance['truck_var_cost'] = int(data_truck[2])
    instance['truck_fix_cost'] = int(data_truck[3])

    #motobike data
    data_motor = lines[5].split(',')
    instance['motor_num'] = int(data_motor[0])
    instance['motor_cap'] = int(data_motor[2])
    instance['motor_var_cost'] = int(data_motor[3])
    instance['motor_fix_cost'] = int(data_motor[4])

    #sattelite and depot data
    data_sat = lines[8].split('   ')
    instance['sat_num'] = len(data_sat)
    instance['sattelite'] = {}

    for i in range(0, len(data_sat)):
        if i == 0:
            temp = data_sat[i].split(',')
            instance['sattelite'][i] = {'coordinate':{'x':0, 'y':0},
                                        'demand':0}
            instance['sattelite'][i]['coordinate']['x'] = int(temp[0])
            instance['sattelite'][i]['coordinate']['y'] = int(temp[1])
            instance['sattelite'][i]['demand'] = 0
        else:
            temp = data_sat[i].split(',')
            instance['sattelite'][i] = {'coordinate':{'x':0, 'y':0}}
            instance['sattelite'][i]['coordinate']['x'] = int(temp[0])
            instance['sattelite'][i]['coordinate']['y'] = int(temp[1])
            instance['sattelite'][i]['demand'] = 0

    #customer data
    data_cus = lines[11].split('   ')
    instance['cus_num'] = len(data_cus)
    instance['customer'] = {}

    for i, value in enumerate(data_cus):
        temp = value.split(',')
        instance['customer'][i+len(data_sat)] = {'coordinate':{'x':0, 'y':0},
                                                  'demand': 0,
                                                  'ready_time':0,
                                                  'due_time':0}
        instance['customer'][i+len(data_sat)]['coordinate']['x'] = int(temp[0])
        instance['customer'][i+len(data_sat)]['coordinate']['y'] = int(temp[1])
        instance['customer'][i+len(data_sat)]['demand'] = int(temp[2])
        instance['customer'][i+len(data_sat)]['ready_time'] = 0
        instance['customer'][i+len(data_sat)]['due_time'] = 144

    #distance matrix      
    distance = []
    for i in range(len(data_sat) + len(data_cus)):
        row = []
        for j in range(len(data_sat) + len(data_cus)):
            xi = instance['sattelite'][i]['coordinate']['x'] if i < len(data_sat) else instance['customer'][i]['coordinate']['x']
            xj = instance['sattelite'][j]['coordinate']['x'] if j < len(data_sat) else instance['customer'][j]['coordinate']['x']
            yi = instance['sattelite'][i]['coordinate']['y'] if i < len(data_sat) else instance['customer'][i]['coordinate']['y']
            yj = instance['sattelite'][j]['coordinate']['y'] if j < len(data_sat) else instance['customer'][j]['coordinate']['y']
            if i == j:
                row.append(0)
            else:
                row.append(((xi - xj)**2 + (yi - yj)**2)**0.5)
        distance.append(row)

    instance["dtm"] = np.array(distance)
    return instance
