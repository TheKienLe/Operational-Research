from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np
import math


def vrp_model():

    # sets
    df = pd.read_excel("data.xlsx", "parameters")
    n = int(df.iloc[0, 1])
    nc = int(df.iloc[1, 1])
    k = int(df.iloc[2, 1])
    p = int(df.iloc[3, 1])

    N = range(n+1)            # root node
    NC = range(1, nc+1)       # destination node
    K = range(1, k+1)       # set of vehicle
    P = range(1, p+1)       # set of compartment
    bigM = 1e10

    # parameters
    def lst_to_dict(lst):
        q = dict()
        for i in NC:
            for p in P:
                q[(i, p)] = float((lst[i][p-1]))  # i-1 --> i
        return q

    # quantity of type p at hospital i
    q = lst_to_dict(np.array(pd.read_excel("data.xlsx", "raw"))[:, [5, 6]])
    print("Demand", q)

    # distance
    x = np.array(pd.read_excel("data.xlsx", "raw"))[:, 2]
    y = np.array(pd.read_excel("data.xlsx", "raw"))[:, 3]

    d = dict()
    for i in N:
        for j in N:
            if i == j:
                d[(i, j)] = 0  # lol :))
            else:
                d[(i, j)] = float(
                    round(math.sqrt((x[i-1] - x[j-1])**2 + (y[i-1] - y[j-1])**2)/9000000, 2))

    Q = dict()
    q_temp = np.transpose(
        np.delete(np.array(pd.read_excel("data.xlsx", "capacity")), 0, 1))

    for i in range(len(q_temp[0])):
        for k in K:
            Q[(i+1, k)] = q_temp[k-1][i]
    print("Capacity", Q)


    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return
    pass

    # variables
    # N --> NC
    x = dict()
    for i in N:
        for j in N:
            for k in K:
                x[(i, j, k)] = solver.BoolVar("")

    z = dict()
    for i in N:
        for k in K:
            for p in P:
                z[(i, k, p)] = solver.BoolVar("")

    y = dict()
    for i in N:
        for k in K:
            y[(i, k)] = solver.BoolVar("")

    u = dict()
    for i in NC:
        for k in K:
            for p in P:
                u[(i, k, p)] = solver.IntVar(0, bigM, "")

    # constraints
    # ct2
    for i in NC:
        solver.Add(solver.Sum([y[(i, k)] for k in K]) == 1, "ct2")

    # ct3
    solver.Add(solver.Sum([y[(0, k)] for k in K]) <= k, "ct3")

    # ct4
    # y_ik --> y_jk
    for j in NC:
        for k in K:
            solver.Add(solver.Sum([x[(i, j, k)]
                       for i in N if i != j]) == y[(j, k)], "ct4")

    # ct5
    for i in NC:
        for k in K:
            solver.Add(solver.Sum(x[(i, j, k)]
                       for j in N if i != j) == y[(i, k)], "ct5")

    # ct6
    for i in NC:
        for j in NC:
            for k in K:
                for p in P:
                    if i != j:
                        solver.Add(u[(i, k, p)] - u[(j, k, p)] + Q[(p, k)]
                                   * x[(i, j, k)] <= Q[(p, k)] - q[(j, p)], "ct6")

    # ct7a
    # q_jp --> q_ip
    # for i in NC:
    #     for p in P:
    #         for k in K:
    #             solver.Add(q[(i, p)] <= u[(i, k, p)], "ct7a")
    #             solver.Add(u[(i, k, p)] <= Q[(p, k)], "ct7b")

    # ct8
    for j in NC:
        for k in K:
            for p in P:
                solver.Add(z[(j, k, p)] <= solver.Sum(
                    [x[(i, j, k)] for i in N if i != j]), "ct8")

    # ct9
    for j in NC:
        for p in P:
            solver.Add(solver.Sum([z[(j, k, p)] for k in K]) == 1, "ct9")

    # ct10
    for p in P:
        for k in K:
            solver.Add(solver.Sum([z[(j, k, p)] * q[(j, p)]
                       for j in NC]) <= Q[(p, k)], "ct10")

    # objective function
    solver.Minimize(solver.Sum([d[(i, j)] * x[(i, j, k)]
                    for i in N for j in N for k in K]))
    
    # Sets a time limit of 1 hour.
    solver.SetTimeLimit(30*60*1000)

    # set a minimum gap limit for the integer solution during branch and cut
    gap = 0.01
    solverParams = pywraplp.MPSolverParameters()
    solverParams.SetDoubleParam(solverParams.RELATIVE_MIP_GAP, gap)

    print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()

    # print solution

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f"Total distance = {solver.Objective().Value()}")
        for i in N:
            for k in K:
                if y[(i, k)].solution_value() == 1:
                    print(f"y{i, k}=", y[(i, k)].solution_value())
        print()
        for i in N:
            for j in N:
                for k in K:
                    if x[(i, j, k)].solution_value() == 1:
                        print(f"x{i, j, k}=", x[(i, j, k)].solution_value())
    else:
        print("No optimal solution")


if __name__ == "__main__":
    vrp_model()
