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
                q[(i, p)] = float((lst[i][p-1])) # i-1 --> i
        return q

    # quantity of type p at hospital i
    q = lst_to_dict(np.array(pd.read_excel("data.xlsx", "coor"))[:, [3, 4]])

    # distance
    x = np.array(pd.read_excel("data.xlsx", "coor"))[:, 1]
    y = np.array(pd.read_excel("data.xlsx", "coor"))[:, 2]

    d = dict()
    for i in N:
        for j in N:
            if i == j:
                d[(i, j)] = 0 # lol :))
            else:
                d[(i, j)] = float(
                    round(math.sqrt((x[i-1] - x[j-1])**2 + (y[i-1] - y[j-1])**2), 2))

    Q = dict()
    q_temp = np.transpose(
        np.delete(np.array(pd.read_excel("data.xlsx", "capacity")), 0, 1))
    for i in range(len(q_temp[0])):
        Q[i+1] = q_temp[0][i]

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
    # Why???
    # bc6 = dict()
    # for i in NC:
    #     for j in NC:
    #         for k in K:
    #             for p in P:
    #                 if i != j:
    #                     bc6[(i, j, k, p)] = solver.IntVar(0, bigM, "")

    # for i in NC:
    #     for j in NC:
    #         for k in K:
    #             for p in P:
    #                 if i != j:
    #                     solver.Add(bc6[(i, j, k, p)] <= bigM * x[(i, j, k)])
    #                     solver.Add(bc6[(i, j, k, p)] <= Q[p] +
    #                                bigM * (1 - x[(i, j, k)]))
    #                     solver.Add(bc6[(i, j, k, p)] >= Q[p] -
    #                                bigM * (1 - x[(i, j, k)]))
    #                     solver.Add(bc6[(i, j, k, p)] <= Q[p] -
    #                                q[(j, p)] - u[(i, k, p)] + u[(j, k, p)], "ct6")
    
    for i in NC:
        for j in NC:
            for k in K:
                for p in P:
                    if i != j:
                        solver.Add(u[(i, k, p)] - u[(j, k, p)] + Q[p]*x[(i, j, k)] <= Q[p] - q[(j, p)], "ct6")

    # ct7a
    # q_jp --> q_ip
    # merge two loops
    for i in NC:
        for p in P:
            for k in K:
                solver.Add(q[(i, p)] <= u[(i, k, p)], "ct7a")
                solver.Add(u[(i, k, p)] <= Q[p], "ct7b")

    # ct8
    for j in NC:
        for k in K:
            for p in P:
                solver.Add(z[(j, k, p)] <= solver.Sum(
                    [x[(i, j, k)] for i in N if i !=j]), "ct8")

    # ct9
    for j in NC:
        for p in P:
            solver.Add(solver.Sum([z[(j, k, p)] for k in K]) == 1, "ct9")

    # ct10
    # bc10 = dict()
    # for k in K:
    #     for p in P:
    #         for j in N:
    #             if j != 0:
    #                 bc10[(j, k, p)] = solver.IntVar(0, bigM, "")

   # for k in K:
    #     for p in P:
    #         for j in N:
    #             if j != 0: 
    #                 solver.Add(bc10[(j, k, p)] <= bigM * z[(j, k, p)])
    #                 solver.Add(bc10[(j, k, p)] <= q[(j, p)] +
    #                            bigM * (1 - z[(j, k, p)]))
    #                 solver.Add(bc10[(j, k, p)] >= q[(j, p)] -
    #                            bigM * (1 - z[(j, k, p)]))
    #         solver.Add(solver.Sum([bc10[(j, k, p)]
    #                    for j in N if j != 0]) <= Q[p], "ct10")
    for p in P:
        for k in K:
            solver.Add(solver.Sum([z[(j, k, p)] * q[(j, p)] for j in NC]) <= Q[p], "ct10")

    # objective function
    solver.Minimize(solver.Sum([d[(i, j)] * x[(i, j, k)]
                    for i in N for j in N for k in K]))

    print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()

    # print solution

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f"Total makespan = {solver.Objective().Value()}")
        for i in N:
            for k in K:
                if y[(i, k)].solution_value() == 1:
                    print(f"y{i,k}=", y[(i, k)].solution_value())
        print()
        for i in N:
            for j in N:
                for k in K:
                    if x[(i, j, k)].solution_value() == 1:
                        print(f"x{i,j,k}=", x[(i, j, k)].solution_value())
    else:
        print("No optimal solution")


if __name__ == "__main__":
    vrp_model()
