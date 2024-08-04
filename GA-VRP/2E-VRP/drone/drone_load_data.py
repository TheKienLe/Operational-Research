import pandas as pd
import numpy as np
from drone_utils import *

def load_data_excel(file_name):
    """
    Load a VRP instance from an Excel file.

    Parameters
    ----------
    file_name : strx`
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
    >>> instance['port_1']
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
        if sheet_name == "Parameter":
            data = pd.read_excel(file_name, sheet_name)
            #mother data
            instance['mother_num'] = 1
            instance['mother_cap'] = data.iloc[4, 2]
            instance['mother_var_cost'] = data.iloc[8, 2]
            instance['mother_fix_cost'] = 0
            
            #end data
            instance['end_num'] = data.iloc[1, 2]
            instance['end_cap'] = data.iloc[9, 2]
            instance['end_var_cost'] = data.iloc[7, 2]
            instance['end_fix_cost'] = 0

            #customer, port and depot data
            instance['customer_num'] = data.iloc[0, 2]
            print('customer_num', data.iloc[0, 2])
            instance['port_num'] = data.iloc[2, 2]
            instance['cluster_num'] = data.iloc[10, 2]
            instance['customer'] = {}
            instance['port'] = {}

        elif sheet_name == "Customers":
            instance["customer"] = {}
            instance["depart"] = {}
            instance["port"] = {}
            customers = pd.read_excel(file_name, sheet_name)
            for customer_id in range(len(customers["Node"])):
                name = ""
                if customer_id == 0:
                    name = "depart"
                elif customer_id <= instance['customer_num'] :
                    name = "customer"
                else:
                    name = "port"
                instance[name][customer_id] = {
                    "coordinates": {"x": customers['x-coord'][customer_id], 
                                    "y": customers['y-coord'][customer_id]},
                    "demand": customers["Demand"][customer_id],
                    # "due_time": customers["due_time"][customer_id],
                    # "ready_time": customers["ready_time"][customer_id],
                    # "service_time": customers["service_time"][customer_id]
                    }
        elif sheet_name == "Distance":       
            #distance matrix      
            distance = pd.read_excel(file_name, sheet_name)
            instance["dtm"] = np.array(distance)
        
        elif sheet_name == "Time":       
            #distance matrix      
            time = pd.read_excel(file_name, sheet_name)
            instance["time"] = np.array(time)
        
        elif sheet_name == "Clusters":
            instance["cluster"] = {cl+1:{
                "customer": [],
                "demand": 0,
                "distance": {port: 0 
                             for port in instance["port"].keys()},
                "launch": 0
            } for cl in range(instance['cluster_num'])}

            clusters = pd.read_excel(file_name, sheet_name)
            for _, row in clusters.iterrows():
                
                cl = row["Cluster"]
                cus = row["Node"]

                instance["cluster"][cl]["customer"].append(cus)
                instance["cluster"][cl]["demand"] += instance["customer"][cus]["demand"]
                if row["Launch"] == 1:
                    instance["cluster"][cl]["launch"] = cus
            print(instance["cluster"])
            for cluster in instance["cluster"].values():
                cluster["distance"] = clus_port_distance(instance["dtm"], 
                                                       cluster["customer"],
                                                       instance["port"].keys())
            print(instance["cluster"])
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
    >>> instance['port_1']
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

    #mother data
    data_mother = lines[2].split(',')
    instance['mother_num'] = int(data_mother[0])
    instance['mother_cap'] = int(data_mother[1])
    instance['mother_var_cost'] = int(data_mother[2])
    instance['mother_fix_cost'] = int(data_mother[3])

    #end data
    data_end = lines[5].split(',')
    instance['end_num'] = int(data_end[0])
    instance['end_cap'] = int(data_end[2])
    instance['end_var_cost'] = int(data_end[3])
    instance['end_fix_cost'] = int(data_end[4])

    #port and depot data
    data_port = lines[8].split('   ')
    instance['port_num'] = len(data_port)
    instance['port'] = {}

    for i in range(0, len(data_port)):
        if i == 0:
            temp = data_port[i].split(',')
            instance['port'][i] = {'coordinate':{'x':0, 'y':0},
                                        'demand':0}
            instance['port'][i]['coordinate']['x'] = int(temp[0])
            instance['port'][i]['coordinate']['y'] = int(temp[1])
            instance['port'][i]['demand'] = 0
        else:
            temp = data_port[i].split(',')
            instance['port'][i] = {'coordinate':{'x':0, 'y':0}}
            instance['port'][i]['coordinate']['x'] = int(temp[0])
            instance['port'][i]['coordinate']['y'] = int(temp[1])
            instance['port'][i]['demand'] = 0

    #customer data
    data_cus = lines[11].split('   ')
    instance['cus_num'] = len(data_cus)
    instance['customer'] = {}

    for i, value in enumerate(data_cus):
        temp = value.split(',')
        instance['customer'][i+len(data_port)] = {'coordinate':{'x':0, 'y':0},
                                                  'demand': 0,
                                                  'ready_time':0,
                                                  'due_time':0}
        instance['customer'][i+len(data_port)]['coordinate']['x'] = int(temp[0])
        instance['customer'][i+len(data_port)]['coordinate']['y'] = int(temp[1])
        instance['customer'][i+len(data_port)]['demand'] = int(temp[2])
        instance['customer'][i+len(data_port)]['ready_time'] = 0
        instance['customer'][i+len(data_port)]['due_time'] = 144

    #distance matrix      
    distance = []
    for i in range(len(data_port) + len(data_cus)):
        row = []
        for j in range(len(data_port) + len(data_cus)):
            xi = instance['port'][i]['coordinate']['x'] if i < len(data_port) else instance['customer'][i]['coordinate']['x']
            xj = instance['port'][j]['coordinate']['x'] if j < len(data_port) else instance['customer'][j]['coordinate']['x']
            yi = instance['port'][i]['coordinate']['y'] if i < len(data_port) else instance['customer'][i]['coordinate']['y']
            yj = instance['port'][j]['coordinate']['y'] if j < len(data_port) else instance['customer'][j]['coordinate']['y']
            if i == j:
                row.append(0)
            else:
                row.append(((xi - xj)**2 + (yi - yj)**2)**0.5)
        distance.append(row)

    instance["dtm"] = np.array(distance)
    return instance
