import pandas as pd

file_name = "./data/instance.xlsx"

# coor_data = pd.read_excel(file_name, "distance")

# size = 50

# distance = []
# for i in range(size+1):
#     row = []
#     for j in range(size + 1):
#         row.append(coor_data[i][j])
#     distance.append(row)

# coor_data = pd.read_excel(file_name, "customers")[["x", "y"]]
# distance = []
# x = coor_data["x"]
# y = coor_data["y"]
# for i in range(len(coor_data)):
#     row = []
#     for j in range(len(coor_data)):
#         if i == j:
#             row.append(0)
#         else:
#             row.append(((x[i] - x[j])**2 + (y[i] - y[j])**2)**0.5)
#     distance.append(row)

# print(len(distance))

