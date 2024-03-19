from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np


def MIP_model():
    # n, r,  m, v
    # production data
    df = pd.read_excel("data.xlsx", "set")
    n = df.iloc[0, 2]
    r = df.iloc[1, 2]
    m = df.iloc[2, 2]
    v = df.iloc[3, 2]
    # set
    N = range(n)
    R = range(r)
    M = range(m)
    V = range(v)

    # parameter
    # unit processing cost on machine m per unit of time
    lamda = np.transpose(np.delete(np.array(pd.read_excel(
        "data.xlsx", "machine_cost")), 0, 1))[0]

    # fixed cost of vihecle type v
    F = np.transpose(np.delete(np.array(pd.read_excel(
        "data.xlsx", "vehicle_cost")), 0, 1))[0]

    # variable cost of vehicle v per unit of time
    Theta = np.transpose(np.delete(np.array(pd.read_excel(
        "data.xlsx", "vehicle_cost")), 0, 1))[1]

    # weight of early delivery
    mu = df.iloc[4, 2]
    # weight of tardy delivery
    fei = df.iloc[5, 2]

    # process time of operation r of job j on machine m
    # [job, operation, machine, time]
    p = lst_to_dict(np.array(pd.read_excel("data.xlsx", "processing_time")))

    # process machine matrix
    # [job, operation, machine, value]
    a = lst_to_dict(np.array(pd.read_excel(
        "data.xlsx", "process_machine_matrix")))

    # transportation time from customer i to customer j by vehicle v
    t = _2d_lst_to_dict(
        np.delete(np.array(pd.read_excel("data.xlsx", "transport_time")), 0, 1), M)

    # size of job j
    theta = np.transpose(np.delete(np.array(pd.read_excel(
        "data.xlsx", "size_job")), 0, 1))[0]

    # delivery time window of job j
    tw = np.array(pd.read_excel("data.xlsx", "delivery_time_window"))

    # capacity of vehicle v
    q = np.transpose(
        np.delete(np.array(pd.read_excel("data.xlsx", "vehicle_capacity")), 0, 1))[0]

    # decision variable
    # production

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return

    # production start time of operation r of job j
    pi = {}
    for j in N:
        for r in R:
            pi[(j, r)] = solver.IntVar(0, solver.infinity(), "")

    # production completion time of operation r of job j
    gamma = {}
    for j in N:
        for r in R:
            gamma[(j, r)] = solver.IntVar(0, solver.infinity(), "")

    # production complation time of job j
    C = []
    for j in N:
        C.append(solver.IntVar(0, solver.infinity(), ""))

    # binary variable takes the value 1 if operation r of job j is processed by machine m else 0
    X = {}
    for j in N:
        for r in R:
            for m in M:
                X[(j, r, m)] = solver.BoolVar("")

    # binary variable takes the value 1 if operation r of job j is processed immediately after the operation f of job i, both on machine m. else 0
    Y = {}
    for i in N:
        for f in R:
            for j in N:
                for m in M:
                    Y[(i, f, j, m)] = solver.BoolVar("")

    # distribution
    # deliver time of order (job) j
    D = []
    for j in N:
        D.append(solver.IntVar(0, solver.infinity(), ""))

    # visiting time of customer (order) j by vehicle v
    T = {}
    for j in N:
        for v in V:
            T[(j, v)] = solver.IntVar(0, solver.infinity(), "")

    # leaving time of vehicle v from production facility
    S = []
    for v in V:
        S.append(solver.IntVar(0, solver.infinity(), ""))

    # visiting time of the last customer (order) in the tour of vehicle v
    E = []
    for v in V:
        E.append(solver.IntVar(0, solver.infinity(), ""))

    # binary variable takes the value 1 if job j is delivered by vehicle else 0
    Z = {}
    for j in N:
        for v in V:
            Z[(j, v)] = solver.BoolVar("")

    # binary variable takes the value 1 if job j is delivered after job i, by vehicle v else 0
    U = {}
    for i in N:
        for j in N:
            for v in V:
                U[(i, j, v)] = solver.BoolVar("")

    # binary variable take the value 1 if vehicle v is used for delivery else 0
    W = []
    for v in V:
        W.append(solver.BoolVar(""))


def lst_to_dict(lst):
    mydict = dict()
    for sublist in lst:
        mydict[tuple(sublist[0:3])] = sublist[len(sublist) - 1]
    return mydict


def _2d_lst_to_dict(lst, V):
    mydict = dict()
    for v in V:
        for i in range(len(lst)):
            for j in range(len(lst)):
                mydict[(i, j, v)] = lst[i][j]
    return mydict


MIP_model()
