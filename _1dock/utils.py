import pandas as pd


def read_parameters(file_name, sheet_name):
    data = pd.read_excel(file_name, sheet_name)
    num_vehicles = data.iloc[0, 1]
    num_customers = data.iloc[1, 1]
    num_depots = data.iloc[2, 1]
    return num_vehicles, num_customers, num_depots


def read_parameters_data(file_name, sheet_name):
    data = pd.read_excel(file_name, sheet_name)
    results = []
    fields = data.columns

    if sheet_name == "Vehicles":
        fields = list(fields[1:])
    elif sheet_name not in pd.ExcelFile("Data.xlsx").sheet_names:
        return "Incorrect sheet_name"

    for i in range(len(data)):
        result = dict()
        for j in range(len(fields)):
            result[fields[j]] = data[fields[j]][i]
        results.append(result)
    return results
