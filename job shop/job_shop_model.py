from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np


def MIP_model(data_address=r"job shop/data.xlsx"):
    # n, r,  m, v
    # production data 
    # If int --> int
    df = pd.read_excel(data_address, "set")
    n = int(df.iloc[0, 2])
    r = int(df.iloc[1, 2])
    m = int(df.iloc[2, 2])
    v = int(df.iloc[3, 2])
    # set # index [1, n+1]
    N_full = range(0,n+2)
    N = range(1, n+1)
    N_plus = range(1,n+2)
    N_minus = range(0, n+1)
    R = range(1, r+1)
    R_minus = range(2, r+1)
    M = range(1, m+1)
    V = range(1, v+1)

    # parameter
    # unit processing cost on machine m per unit of time
    lamda = np.transpose(np.delete(np.array(pd.read_excel(
        data_address, "machine_cost")), 0, 1))[0]

    # fixed cost of vihecle type v
    F = np.transpose(np.delete(np.array(pd.read_excel(
        data_address, "vehicle_cost")), 0, 1))[0]

    # variable cost of vehicle v per unit of time
    Theta = np.transpose(np.delete(np.array(pd.read_excel(
        data_address, "vehicle_cost")), 0, 1))[1]

    # weight of early delivery
    mu = df.iloc[4, 2]
    # weight of tardy delivery
    fei = df.iloc[5, 2]

    # process time of operation r of job j on machine m
    # [job, operation, machine, time]
    P = lst_to_dict(np.array(pd.read_excel(
        data_address, "processing_time")))

    # process machine matrix
    # [job, operation, machine, value]
    a = lst_to_dict(np.array(pd.read_excel(
        data_address, "process_machine_matrix")))
    
    
    # transportation time from customer i to customer j by vehicle v
    t = _2d_lst_to_dict(
        np.delete(np.array(pd.read_excel(
            data_address, "transport_time")), 0, 1), V)

    # size of job j
    theta = np.transpose(np.delete(np.array(pd.read_excel(
        data_address, "size_job")), 0, 1))[0]

    # delivery time window of job j
    tw = np.array(pd.read_excel(
        data_address, "delivery_time_window"))

    # capacity of vehicle v
    q = np.transpose(
        np.delete(np.array(pd.read_excel(
            data_address, "vehicle_capacity")), 0, 1))[0]
    
    bigM = 1e10

    # decision variable
    # production

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return

    # production start time of operation r of job j
    pi = {}
    for j in N:
        for r in R:
            pi[(j, r)] = solver.NumVar(0, solver.infinity(), "")

    # production completion time of operation r of job j
    gamma = {}
    for j in N_full:
        for r in R:
            gamma[(j, r)] = solver.NumVar(0, solver.infinity(), "")

    # production complation time of job j
    C = []
    for j in N:
        C.append(solver.NumVar(0, solver.infinity(), ""))

    # binary variable takes the value 1 if operation r of job j is processed by machine m else 0
    X = {}
    for j in N:
        for r in R:
            for m in M:
                X[(j, r, m)] = solver.BoolVar("")

    # binary variable takes the value 1 if operation r of job j is processed immediately after the operation f of job i, both on machine m. else 0
    Y = {}
    for i in N_full:
        for f in R:
            for j in N_full:
                for r in R:
                    for m in M:
                        Y[(i, f, j, r, m)] = solver.BoolVar("")
                        # Y[(i, f, j, r, m)] = solver.NumVar(0, 1, "")

    # distribution
    # deliver time of order (job) j
    D = []
    for j in N:
        D.append(solver.NumVar(0, solver.infinity(), ""))

    # visiting time of customer (order) j by vehicle v
    T = {}
    for j in N_full:
        for v in V:
            T[(j, v)] = solver.NumVar(0, solver.infinity(), "")

    # leaving time of vehicle v from production facility
    S = []
    for v in V:
        S.append(solver.NumVar(0, solver.infinity(), ""))

    # visiting time of the last customer (order) in the tour of vehicle v
    E = []
    for v in V:
        E.append(solver.NumVar(0, solver.infinity(), ""))

    # binary variable takes the value 1 if job j is delivered by vehicle else 0
    Z = {}
    for j in N:
        for v in V:
            Z[(j, v)] = solver.BoolVar("")

    # binary variable takes the value 1 if job j is delivered after job i, by vehicle v else 0
    U = {}
    for i in N_full:
        for j in N_full:
            for v in V:
                U[(i, j, v)] = solver.BoolVar("")

    # binary variable take the value 1 if vehicle v is used for delivery else 0
    W = []
    for v in V:
        W.append(solver.BoolVar(""))

    
    ## Constraint
    
    # 3
    # Fix Sum constrant
    for j in N:
        for r in R:
                solver.Add(solver.Sum([X[(j, r, m)] for m in M]) == 1, "ct3")

    # 4
    for j in N:
        for r in R:
            for m in M:
                solver.Add(X[(j, r, m)] <= a[(j, r, m)], "ct4")


    # 5
    # fIX  P[0] with P[N+1]
    # fix sum constraint
    for i in N:
        for f in R:
            for m in M:
                solver.Add(X[(i, f, m)] == solver.Sum(
                    [Y[(i,f,j,r,m)] for j in N_plus for r in R]), "ct5")


    # 6
    for j in N:
        for r in R:
            for m in M:
                solver.Add(X[(j, r, m)] == solver.Sum(
            [Y[(i,f,j,r,m)] for i in N_minus for f in R]), "ct6")
    
    # 7 Linear max
    # Linearize binary * continuous
    bc7 = {}
    for i in N:
        for f in R:
            for m in M:
                bc7[(i, f, m)] = solver.NumVar(0, bigM, "")

    z7 = {}
    for j in N:
        for r in R_minus:
            for i in range(2):
                z7[(j, r, i)] = solver.BoolVar("")
                
    for j in N:
        for r in R_minus:
            solver.Add(pi[(j, r)] >= gamma[(j,r-1)])
            # solver.Add(pi[(j, r)] <= gamma[(j,r-1)] + bigM*z7[(j, r, 0)])

            for i in N:
                for f in R:
                    for m in M:
                        solver.Add(bc7[(i, f, m)] <= bigM * Y[(i, f, j, r, m)])
                        solver.Add(bc7[(i, f, m)] <= gamma[(i,f)] + bigM*(1-Y[(i, f, j, r, m)]))
                        solver.Add(bc7[(i, f, m)] >= gamma[(i,f)] - bigM*(1-Y[(i, f, j, r, m)]))

    # missing constraint 5

    # 8
    for j in N:
        for r in R:
            solver.Add(gamma[(j, r)] == pi[(j,r)] + solver.Sum([P[(j, r, m)] * X[(j,r,m)] for m in M]), "ct8")

    # 9
    for j in N:
        solver.Add(C[j-1] >= gamma[(j, R[-1])], "ct9")

    # 10
    # for m in M:
    #     solver.Add(solver.Sum(
    #         [Y[(0, f, j, r, m)] for j in N for f in R for r in R]) == 1, "ct10")

    # 11
    # for m in M:
    #     solver.Add(solver.Sum(
    #         [Y[(i, f, n+1, r, m)] for i in N for f in R for r in R]) == 1, "ct11")
        
    # 12
    for j in N:
        solver.Add(solver.Sum(
            [Z[(j, v)] for v in V]) == 1, "ct12")
        
    # 13
    for j in N:
        for v in V:
            solver.Add(Z[(j,v)] == solver.Sum(
                [U[((i, j, v))] for i in N_minus]), "ct13")
    
    # 14 
    for j in N:
        for v in V:
            solver.Add(solver.Sum(
                [U[(i,j,v)] for i in N_minus]) == solver.Sum(
                    [U[(i,j,v)] for i in N_plus]), "ct14a")
            solver.Add(solver.Sum(
                    [U[(i,j,v)] for i in N_plus]) <= 1, "ct14b")
            
    # 15 
    for v in V:
        solver.Add(solver.Sum(
            [U[(0,j,v)] for j in N]) == solver.Sum(
                [U[(i,n+1,v)] for i in N]), "ct15a")
        solver.Add(solver.Sum(
                [U[(i,n+1,v)] for i in N]) <= 1, "ct15b")
        
    # 16
    for v in V:
        solver.Add(solver.Sum(
            [Z[(j, v)]*theta[j-1] for j in N]) <= q[v-1], "ct16")
        
    # 17 Add max ct
    # Linearize binary * continuous
    z17 = {}
    for v in V:
        for j in N:
                z17[(v, j)] = solver.BoolVar("")
    bc17 = {}
    for j in N:
        for v in V:
            bc17[(j,v)] = solver.NumVar(0, bigM, "")

    for v in V:
        solver.Add(T[(0, v)] == S[v-1], "ct15a")
        
        for j in N:
            solver.Add(bc17[(j,v)] <= bigM * Z[(j, v)])
            solver.Add(bc17[(j,v)] <= C[j-1] + bigM *(1 - Z[j, v]))
            solver.Add(bc17[(j,v)] >= C[j-1] - bigM *(1 - Z[j, v]))
            solver.Add(S[v-1] >= bc17[(j,v)])
            solver.Add(S[v-1] <= bc17[(j,v)] + bigM*z17[(v, j)])

        solver.Add(solver.Sum([z17[(v, j)] for j in N]) <= len(N) - 1)
        
    # 18 Linearize binary * continuous
    bc18 = {}
    
    for i in N:
        for j in N:
            for v in V:
                bc18[(i,j,v)] = solver.NumVar(0, bigM, "")

    for j in N:
        for v in V:
            for i in N:
                solver.Add(bc18[(i,j,v)] <= bigM * U[(i,j,v)])
                solver.Add(bc18[(i,j,v)] <= T[(i,v)] + t[(i,j,v)] + bigM*(1-U[(i,j,v)]))
                solver.Add(bc18[(i,j,v)] >= T[(i,v)] + t[(i,j,v)] - bigM*(1-U[(i,j,v)]))
            solver.Add(T[(j,v)] == solver.Sum(
                    [bc18[(i,j,v)] for i in N]), "ct18")

    # 19 Linearize binary * continuous
    bc19 = {}
    for j in N:
        for v in V:
            bc19[(j,v)] = solver.NumVar(0, bigM, "")
    
    for j in N:
        for v in V:
            solver.Add(bc19[(j,v)] <= bigM * Z[j,v])
            solver.Add(bc19[(j,v)] <= T[(j,v)] + bigM*(1-Z[j,v]))
            solver.Add(bc19[(j,v)] >= T[(j,v)] - bigM*(1-Z[j,v]))

        solver.Add(D[j-1] == solver.Sum(
            [bc19[(j,v)] for v in V]), "ct19")
    
    # 20
    for v in V:
        solver.Add(E[v-1] == S[v-1] + solver.Sum(
            [U[(i,j,v)] * t[(i,j,v)] for i in N_minus for j in N_plus])
            , "ct20")
    
    # 21 Add Max constraint
    z21 = {}
    for v in V:
        for j in N:
                z21[(v, j)] = solver.BoolVar("")

    for v in V:        
        for j in N:
            solver.Add(W[v-1] >= Z[(j, v)])
            solver.Add(W[v-1] <= Z[(j, v)] + bigM*z21[(v, j)])

        solver.Add(solver.Sum([z21[(v, j)] for j in N]) <= n - 1)
    
# Objective
    solver.Minimize(solver.Sum(lamda[m-1] * P[(j, r, m)] * X[(j, r, m)] for j in N for r in R for m in M))

    # Solve
    print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()

    # Print solution.
    
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f"Total makespan = {solver.Objective().Value()}\n")

        for j in N:
            print(f"C{j} =", C[j-1].solution_value())
        
        # for f in R:
        #     for j in N_full:
        #         for r in R:
        #             for m in M:
        #                 print(f"C{0, f, j, r, m} =", Y[(0, f, j, r, m)].solution_value())

        # for i in self.N:
        #     for f in self.F:
        #         print("y =", self.Y[(i, f)].solution_value())

        # for i in self.N:
        #     for k in self.lambda_dict[i]:
        #         for j in self.E_dict[k]:
        #             print(f"x{i,j,k} =", self.X[(i,j,k)].solution_value())
        # 
        # print("y =", self.Y.solution_value())
    else:
        print("No solution found.")

    # 11
    total6 = 0
    for i in N:
        for f in R:
            for r in R:
                for m in M:
                    total6 += Y[(i, f, r, r, m)]
    solver.Add(total6 == 1)


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

if __name__ == "__main__":
    MIP_model()
