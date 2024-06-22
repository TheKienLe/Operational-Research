from ortools.linear_solver import pywraplp
import pandas as pd
from utils import *
import itertools


def cross_dock_model(name):

    # index & subset
    parameter = pd.read_excel(name, "parameter")
    fl = parameter.iloc[0, 1]
    s = parameter.iloc[1, 1]
    cl = parameter.iloc[2, 1]
    hs = parameter.iloc[3, 1]
    ls = parameter.iloc[4, 1]

    # set
    F = range(0, fl)
    S = range(0, s)
    CL = range(0, cl)
    HS = range(0, hs)
    LS = range(0, ls)

    # parameter
    # floor capacity
    cap = df_to_list(pd.read_excel(
        name, "location_floor_cluster"), "floor", "cap")
    # print("floor cap:", cap)

    # location each floor in each cluster
    y = df_to_list(pd.read_excel(name, "location_floor_cluster"),
                   "cluster", "floor", "allocation_in_cluster")
    # print("cluster location", y)

    # distance from floor to shipping point
    distS = df_to_list(pd.read_excel(
        name, "distance_floor_shipping"), "floor", "distance")
    # print("floor distance", distS)

    # distance picking carton in cluster
    distPcb = df_to_list(pd.read_excel(name, "distance_cluster").query(
        "type == 'carton'"), "cluster", "distance")
    # print("Distance carton", distPcb)

    # distance picking plastic in cluster
    distPpb = df_to_list(pd.read_excel(name, "distance_cluster").query(
        "type == 'plastic'"), "cluster", "distance")
    # print("Distance plastic", distPpb)
    
    # number of pallets shipped to store s within 24 hrs
    PS = df_to_list(pd.read_excel(name, "demand"), "store", "shipped")
    # print("Num pallet shipped", PS)

    # requirement number of pallets store s for within 24 hrs
    D = df_to_list(pd.read_excel(name, "demand"), "store", "demand")
    # print("Store Demand", D)

    # space requirement of store s for 24 hrs
    R = df_to_list(pd.read_excel(name, "demand"), "store", "floor_unit")
    # print("R:", R)
    # print("space requirement", R)

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
        NV["cl" + str(cl)] = solver.NumVar(0, solver.infinity(), "")
    # constraint
    # 3

    sumry = summary_df(y, int(fl/10))
    print("fl", fl)
    print("sumry", sumry)
    print("cl", cl)
    for s in S:
        for cl in CL:
            for f in sumry["cl" + str(cl)]:
                if y[("cl" + str(cl), f)] >= 1:
                    solver.Add(NV["cl" + str(cl)] >=
                               NS["s"+str(s)] * x[("s" + str(s), f)]
                               , "ct3")

    # 4
    for f in F:
        solver.Add(solver.Sum([x[("s" + str(s), "f" + str(f))] for s in S]) <= 1)

    # 5
    for s in S:
        solver.Add(solver.Sum([x[("s" + str(s), "f" + str(f))]
                   for f in F]) <=
                   R["s" + str(s)]
                   , "ct5")

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
        solver.Add(solver.Sum([x[("s" + str(s), "f" + str(f))] *
                              cap["f" + str(f)] for f in F]) >=
                              D["s" + str(s)]
                              , "ct7")

    # 8
    # for f in F:
    #     for s in S:
    #         if R["s" + str(s)] >= 2:
    #             solver.Add(x[("s" + str(s), "f" + str(f))] <=
    #                        x[("s" + str(s), "f" + str(f+1))] + x[("s" + str(s), "f" + str(f-1))])

    # objective func
    PickDist = solver.NumVar(0, solver.infinity(), "")
    ShipDist = solver.NumVar(0, solver.infinity(), "")

    obj_PickDist = solver.Add(PickDist == solver.Sum(
        distPcb["cl" + str(cl)] * NV["cl" + str(cl)] * (1-ppb) + distPpb["cl" + str(cl)] * NV["cl" + str(cl)] * ppb for cl in CL))

    obj_ShipDist = solver.Add(ShipDist == solver.Sum(
        distS["f" + str(f)] * x[("s" + str(s), "f" + str(f))] * PS["s" + str(s)] for f in F for s in S))

    objective = PickDist + ShipDist
  
    solver.Minimize(objective)

    # Sets a time limit of 1 hour.
    solver.SetTimeLimit(10*60*1000)

    # set a minimum gap limit for the integer solution during branch and cut
    gap = 0.05
    solverParams = pywraplp.MPSolverParameters()
    solverParams.SetDoubleParam(solverParams.RELATIVE_MIP_GAP, gap)

    # solve
    print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f"Minimum distance {solver.Objective().Value()}")
        print(f"Picking distance {PickDist.solution_value()}")
        print(f"Shipping distance {ShipDist.solution_value()}")
        for cl in CL: 
            print(f"NV{cl}", NV["cl" + str(cl)].solution_value())
        for s in S:
            for f in F:
                    print("x", s, f, x[("s" + str(s), "f" + str(f))].solution_value())

    else:
        print("No solution")


if __name__ == "__main__":
    cross_dock_model("crossdock_data.xlsx")
