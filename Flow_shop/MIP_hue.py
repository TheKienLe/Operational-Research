from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np

def arr_to_dict(arr):
    dict = {}
    for key, value in arr:
        if key in dict:
            dict[key].append(value)
        else:
            dict[key] = [value]

    return dict


def MIP_model(data):
    # n, r,  m, v
    # production data
    # If int --> int

    df_params = pd.read_excel(data, "Params")
    numO = df_params.iloc[0, 2]
    numP = df_params.iloc[1, 2]
    numR = df_params.iloc[2, 2]
    numT = df_params.iloc[3, 2]


    O = range(0, numO)
    P = range(0, numP)
    R = range(0, numR)
    T = range(0, numT)

    Pr_arr = np.array(pd.read_excel(data, "Pr"))
    Pr = arr_to_dict(Pr_arr)

    bigM = 1e9

    # Production capacity of each resource r within each time unit t
    U = np.squeeze(
            np.array(pd.read_excel(data, "U", index_col=0), dtype=np.float64)
            )

    # Minimum workload required for each resource r
    H = np.squeeze(
            np.array(pd.read_excel(data, "H", index_col=0), dtype=np.float64)
            )

    # Processing time of each component i
    p = np.squeeze(
            np.array(pd.read_excel(data, "p", index_col=0), dtype=np.float64)
            )

    # Setup time required for producing each component i
    s = np.squeeze(
            np.array(pd.read_excel(data, "s", index_col=0), dtype=np.float64)
            )

    # Batch size of each component i
    B = np.squeeze(
            np.array(pd.read_excel(data, "B", index_col=0), dtype=np.float64)
            )

    # Number of components i' required to produce one unit of final product i
    alpha = np.squeeze(
            np.array(pd.read_excel(data, "alpha", index_col=0), dtype=np.float64)
            )

    # Number of final products requested for each order o
    q = np.squeeze(
            np.array(pd.read_excel(data, "q", index_col=0), dtype=np.int32)
            )

    # Latest production start date for each order o
    ls = np.squeeze(
            np.array(pd.read_excel(data, "ls", index_col=0), dtype=np.int32)
            )

    # Deadline for each order o
    d = np.squeeze(
            np.array(pd.read_excel(data, "d", index_col=0), dtype=np.int32)
            )

    # Customer order delay tolerance
    C = 3

    # Remaining processing time of order o after producing component i
    F = np.squeeze(
            np.array(pd.read_excel(data, "F", index_col=0))
            )
    
    # Production cost
    pc = df_params.iloc[4, 2]

    # Order reject penalty
    pr = df_params.iloc[5, 2]

    # Order delay penalty
    pda = df_params.iloc[6, 2]

    # decision variable
    # production

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return

    # Planning horizon
    tau = solver.IntVar(0, bigM, "")

    # Number of items to produce for each order o
    rho = {}
    for o in O:
            rho[o] = solver.IntVar(0, solver.infinity(), "")

    # Binary indicating that order o is accepted
    x = {}
    for o in O:
        x[o] = solver.BoolVar("")

    # Binary indicating that order o is accepted with complete quantity, not partial
    y = {}
    for o in O:
        y[o] = solver.BoolVar("")

    # Binary indicating that order o is rejected
    r_var = {}
    for o in O:
        r_var[o] = solver.BoolVar("")


    # Binary indicating that target delivery time of order o is t
    theta = {}
    for o in O:
        for t in T:
            theta[(o,t)] = solver.BoolVar("")

    # Required number of components i by time t
    I = {}
    for i in P:
        for t in T:
            I[(i, t)] = solver.IntVar(0, bigM, "")

    # The number of setups necessary for producing component i by time t
    psi = {}
    for i in P:
        for t in T:
            psi[(i, t)] = solver.IntVar(0, bigM, "")


    ## Constraint

    # 2, 3, 4
    for o in O:
        # 2
        solver.Add(tau - ls[o] <= bigM*(x[o] + r_var[o]), "ct2")

        # 3
        solver.Add(tau - d[o] <= bigM*(y[o] + r_var[o]), "ct3")

        # 4
        solver.Add(x[o] + r_var[o] <= 1, "ct4")

        # 5
        solver.Add(tau >= (d[o] + C) - bigM*r_var[o], "ct5")

        # 6
        solver.Add(q[o] * y[o] <= rho[o] , "ct6a")
        solver.Add(q[o] * x[o] >= rho[o], "ct6b")

        # 7
        solver.Add((tau-ls[o]) / (d[o]-ls[o]) - bigM*(1-x[o]) - bigM*y[o] <= rho[o]/q[o], "ct7a")
        solver.Add(rho[o]/q[o] <= (tau -ls[o]*x[o]) / (d[o] - ls[o]), "ct7b")

        # 8
        solver.Add(solver.Sum([theta[(o, t)] for t in range(ls[o], d[o]+C+1)]) == x[o], "ct8")

        # # 9
        solver.Add(solver.Sum([theta[(o, t)] for t in range(d[o], d[o]+C+1)]) >= y[o], "ct9")

        # 10
        solver.Add(bigM*(x[o] - y[o] - 1) + tau <= solver.Sum([t * theta[(o, t)] for t in range(ls[o], d[o]+1)]), "ct10a")
        solver.Add(solver.Sum([t * theta[(o, t)] for t in range(ls[o], d[o])]) <= tau, "ct10b")

 

    # 11
    for t in T:
        for i in P:
            for o in O:
                if (t - F[o][i] - 1 >= 0):
                    min11 = min([1, (t-ls[o]) / (d[o]-ls[o])])

                    solver.Add(I[(i, t - F[o][i])] - I[(i, t - F[o][i] - 1)] >= \
                    alpha[o][i] * solver.Sum([min11 * q[o] * theta[(o, t)] \
                                              for t in range(ls[o], d[o]+C+1)])
                                              , "ct11")

    # 12
    for r in R:
        for t in T:
            solver.Add(solver.Sum([s[i]*psi[(i, t)] + p[i]*I[(i, t)]
                                   for i in Pr[r]])
                       <= t * H[r] * U[r] * 2, "ct12")


    # 13
    for r in R:
        solver.Add(solver.Sum([s[i]*psi[(i, numT-1)] + p[i]*I[(i, numT-1)] for i in Pr[r]])
                   >= H[r] * U[r], "ct13")

    # 14
    for t in T:
        for i in P:
            solver.Add(I[(i, t)] <= B[i] * psi[(i, t)], "ct14")

    # Objective
    solver.Minimize(tau/numT * pc + solver.Sum(
        [q[o]*pr*r_var[o] + pda*solver.Sum(
            [(t-d[o]) * theta[(o, t)] for t in range(d[o]+1, numT)]
            ) for o in O]
        ))

    # Solve
    print(f"Solving with {solver.SolverVersion()}")

    # Sets a time limit of 1 hour.
    solver.SetTimeLimit(10*60*1000)

    # set a minimum gap limit for the integer solution during branch and cut
    gap = 0.2
    solverParams = pywraplp.MPSolverParameters()
    solverParams.SetDoubleParam(solverParams.RELATIVE_MIP_GAP, gap)

    status = solver.Solve()

    # Print solution.

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f"Total cost = {solver.Objective().Value()}\n")

        print(f"tau: {tau.solution_value()}")
        for o in O:
            for t in T:
                if theta[(o, t)].solution_value() == 1:
                    print(f"x{o}: {x[o].solution_value()}, r{o}: {r_var[o].solution_value()}, rho{o}: {rho[o].solution_value()}, theta{o, t}: 1")

    else:
        print("No solution found.")

MIP_model("data_hue.xlsx")
