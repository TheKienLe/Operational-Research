from ortools.linear_solver import pywraplp
import pandas as pd
from utils import *


def cross_dock_model(name):

    # index & subset
    parameter = pd.read_excel(name, "parameter")
    f = parameter.iloc[0, 1]
    s = parameter.iloc[1, 1]
    cl = parameter.iloc[2, 1]
    hs = parameter.iloc[3, 1]
    ls = parameter.iloc[4, 1]

    # set
    F = range(0, f)
    S = range(0, s)
    CL = range(0, cl)
    HS = range(0, hs)
    LS = range(0, ls)

    # parameter
    # floor capacity
    cap = df_to_list(pd.read_excel(
        name, "location_floor_cluster"), "floor", "cap")

    # location each floor in each cluster
    y = df_to_list(pd.read_excel(name, "location_floor_cluster"),
                   "cluster", "floor", "allocation_in_cluster")

    # distance from floor to shipping point
    distS = df_to_list(pd.read_excel(
        name, "distance_floor_shipping"), "floor", "distance")

    # distance picking carton in cluster
    distPcb = df_to_list(pd.read_excel(name, "distance_cluster").query(
        "type == 'carton'"), "cluster", "distance")

    # distance picking plastic in cluster
    distPpb = df_to_list(pd.read_excel(name, "distance_cluster").query(
        "type == 'plastic'"), "cluster", "distance")

    # number of pallets shipped to store s within 24 hrs
    PS = df_to_list(pd.read_excel(name, "demand"), "store", "shipped")

    # requirement number of pallets store s for within 24 hrs
    D = df_to_list(pd.read_excel(name, "demand"), "store", "requirement")

    # space requirement of store s for 24 hrs
    R = df_to_list(pd.read_excel(name, "demand"), "store", "floor_unit")

    # percentage plastic boxes picked compared to total box in 24 hrs
    ppb = 0.5

    # priority floor
    pexp = df_to_list(pd.read_excel(name, "priority"),
                      "store", "floor", "priority")

    # number of picking of store s
    NS = df_to_list(pd.read_excel(name, "demand"),
                    "store", "number_of_picking")

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return

    # variables
    # binary variable = 1 if store s allocated to floor f else = 0
    x = dict()
    for s in S:
        for f in F:
            x[("s" + str(s), "f" + str(f))] = solver.BoolVar("")
            # f 0 --> 39
            # s 0 -> 9
            # ()

    # number of visit to cluster
    NV = dict()
    for cl in CL:
        NV["cl" + str(cl)] = solver.IntVar(0, solver.infinity(), "")
    # constraint
    # 3

    sumry = summary_df(y)
    print(sumry)
    for s in S:
        for cl in CL:
            for f in sumry["cl" + str(cl)]:
                if y[("cl" + str(cl), f)] == 1:
                    solver.Add(NV["cl" + str(cl)] >= NS["s"+str(s)] *
                               x[("s" + str(s), f)], "ct3")

    # 4
    for f in F:
        solver.Add(solver.Sum(x[("s" + str(s), "f" + str(f))] for s in S) <= 1)

    # 5
    for f in F:
        solver.Add(solver.Sum(x[("s" + str(s), "f" + str(f))]
                   for f in F) <= R["s" + str(s)])

    # 6
    for s in S:
        for f in F:
            if ("s" + str(s), "f" + str(f)) not in pexp:
                pass
            else:
                solver.Add(x[("s" + str(s), "f" + str(f))] <=
                           pexp[("s" + str(s), "f" + str(f))])

    # 7
    for s in S:
        solver.Add(solver.Sum(x[("s" + str(s), "f" + str(f))] *
                              cap["f" + str(f)] for f in F) >= D["s" + str(s)])

    # 8
    for f in F:
        for s in S:
            if R["s" + str(s)] >= 2:
                solver.Add(x[("s" + str(s), "f" + str(f))] <=
                           x[("s" + str(s), "f" + str(f+1))] + x[("s" + str(s), "f" + str(f-1))])

    # objective func
    objective = solver.Sum(
        distPcb["cl" + str(cl)] * NV["cl" + str(cl)] * (1-ppb) + distPpb["cl" + str(cl)] * NV["cl" + str(cl)] * ppb for cl in CL) + solver.Sum(distS["f" + str(f)] * x[("s" + str(s), "f" + str(f))] * PS["s" + str(s)] for f in F for s in S)
    solver.Minimize(objective)

    # solve
    print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f"Minimum distance {solver.Objective().value()}")
    else:
        print("No solution")


if __name__ == "__main__":
    cross_dock_model("crossdock_data.xlsx")
