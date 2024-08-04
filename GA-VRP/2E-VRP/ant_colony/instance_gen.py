import os
import random

# Ensure the directory exists
os.makedirs('./solomon-100', exist_ok=True)

# Parameters for data generation
num_customers = 100
vehicle_capacity = 200
max_coordinate = 100
max_demand = 30
max_ready_time = 200
max_due_time = 500
max_service_time = 10

# Generate depot data
depot = {
    "cust_no": 0,
    "x_coord": random.randint(0, max_coordinate),
    "y_coord": random.randint(0, max_coordinate),
    "demand": 0,
    "ready_time": 0,
    "due_date": max_due_time,
    "service_time": 0
}

# Generate customer data
customers = []
for i in range(1, num_customers + 1):
    customer = {
        "cust_no": i,
        "x_coord": random.randint(0, max_coordinate),
        "y_coord": random.randint(0, max_coordinate),
        "demand": random.randint(1, max_demand),
        "ready_time": random.randint(0, max_ready_time),
        "due_date": random.randint(max_ready_time, max_due_time),
        "service_time": random.randint(1, max_service_time)
    }
    customers.append(customer)

# Path to the file
file_path = './solomon-100/c101.txt'

# Write the data to the file
with open(file_path, 'w') as file:
    file.write("C101\n")
    file.write("VEHICLE NUMBER = 25\n")
    file.write(f"CAPACITY = {vehicle_capacity}\n\n")
    file.write("CUSTOMER\n")
    file.write("CUST NO.  XCOORD  YCOORD  DEMAND  READY TIME  DUE DATE  SERVICE TIME\n")
    file.write(f"{depot['cust_no']}         {depot['x_coord']}      {depot['y_coord']}      {depot['demand']}       {depot['ready_time']}           {depot['due_date']}      {depot['service_time']}\n")
    
    for customer in customers:
        file.write(f"{customer['cust_no']}         {customer['x_coord']}      {customer['y_coord']}      {customer['demand']}       {customer['ready_time']}           {customer['due_date']}      {customer['service_time']}\n")

print(f"File '{file_path}' has been created with synthetic data.")
