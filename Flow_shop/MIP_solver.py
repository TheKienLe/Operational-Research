from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np
from utils import *

class MIP_model():
    def __init__(self, dataset, mip_solver="SCIP"):
        parameter = pd.read_excel("data.xlsx", "Parameters")
        
        ## Index
        # Total number of job
        self.n = parameter.iloc[0,2]

        # Total number of stage
        self.s = parameter.iloc[1,2]

        # Total number of factory
        self.f = parameter.iloc[2,2]

        # Number of parallel machine at stage k
        self.m = parameter.iloc[3,2]

        ## Range 
        self.N = range(self.n)
        self.S = range(self.s)
        self.F = range(self.f)
        self.M = range(self.m)


        ## Parameter
        # Processing time
        self.p = np.squeeze(
            np.array(pd.read_excel("data.xlsx", "p", index_col=0))
            )

        # Set of stages accessed by job
        lambda_arr = np.array(pd.read_excel("data.xlsx", "lambda"))
        # self.lambda_dict = arr_to_dict(lambda_arr)
        self.lambda_dict = { job:list(self.S) for job in self.N }

        # Set of parallel machine at stage k 
        E_arr = np.array(pd.read_excel("data.xlsx", "E"))
        self.E_dict = { job:list(self.M) for job in self.N }
        
        # Large number
        self.L = 1000000000

        ## Solver
        # Create the mip solver with the SCIP backend.
        self.solver = pywraplp.Solver.CreateSolver(mip_solver)

        if not self.solver:
            return


        ## Variables
        # s_i,k Starting processing time of job i at stage k.
        self.s = {}
        for i in self.N:
            for k in self.lambda_dict[i]:
                    self.s[(i, k)] = self.solver.NumVar(0, self.solver.infinity(), "")

        # Y_ik: if job i is processed in factory f
        self.Y = {}
        for i in self.N:
            for f in self.F:
                self.Y[(i, f)] = self.solver.IntVar(0, 1, "")

        # X_ijk: if job i is processed on machine j at stage k
        self.X = {}
        for i in self.N:
            for k in self.lambda_dict[i]:
                for j in self.E_dict[k]:
                    self.X[(i, j, k)] = self.solver.IntVar(0, 1, "")

        # Z_ilk: if job i precedes job l at stage k
        self.Z = {}  
        for i in self.N:
            for l in self.N:
                for k in intersect(self.lambda_dict[i], self.lambda_dict[l]):
                    self.Z[(i, l, k)] = self.solver.IntVar(0, 1, "")

        # W_ihk if job immediately after its completion at stage j is to be processed at stage h l
        self.W = {}
        for i in self.N:
            for h in self.lambda_dict[i]:
                for k in intersect(self.lambda_dict[i], self.lambda_dict[l]):
                   self.W[(i, h, k)] = self.solver.IntVar(0, 1, "") 

    def solve(self):
        
        # Constraints
        # ct1: Each job should be exactly at one factory
        for i in self.N:
            self.solver.Add(
                self.solver.Sum(
                    [self.Y[(i, f)] for f in self.F]) == 1, 
                    "ct1")

        # ct2: Each job pass all stages and processed by one machine
        for i in self.N:
            for k in self.lambda_dict[i]:
                self.solver.Add(
                    self.solver.Sum(
                        [self.X[(i, j, k)] for j in self.E_dict[k]]) == 1, 
                        "ct2")
        

        # ct3: the next operation can only be started after the previous one is completed
        # Consider eliminate L
        # Add condition h > k
        for i in self.N:
            for h in self.lambda_dict[i]:
                for k in self.lambda_dict[i]:
                    for j in self.E_dict[k]:
                        if h>k:
                            self.solver.Add(
                                self.s[(i,h)] - (self.s[(i,k)] + self.p[i, k]) - self.L*(1 - self.W[(i, k, h)]) >=0,
                                "ct3")
                        
        # ct4: 
        for i in self.N:
            for l in self.N:
                if i != l:
                    for k in intersect(self.lambda_dict[i], self.lambda_dict[l]):
                        self.solver.Add(
                            self.Z[(l,i,k)] + self.Z[(i,l,k)] <= 1,
                        "ct4")
        
        # ct5: 
        # Change from -L to +L
        for i in self.N:
            for l in self.N:
                if i != l:
                    for k in intersect(self.lambda_dict[i], self.lambda_dict[l]):
                        for j in self.E_dict[k]:
                            for f in self.F:
                                self.solver.Add(
                                    self.s[(l,k)] - (self.s[(i,k)] + self.p[i, k]) \
                                        + self.L*(5 - self.Z[(l,i,k)] - self.X[(i,j,k)] 
                                                  - self.X[(l,j,k)] - self.Y[(i,f)] - self.Y[(l,f)]) >=0,
                        "ct5")

        # ct6:
        self.f = self.solver.NumVar(0, self.solver.infinity(), "")
        for i in self.N:
            for k in self.lambda_dict[i]:
                self.solver.Add(self.f >= self.s[(i, k)] + self.p[i, k])


        # Objective
        self.solver.Minimize(self.f)

        # Solve
        print(f"Solving with {self.solver.SolverVersion()}")
        status = self.solver.Solve()

        # Print solution.
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            print(f"Total makespan = {self.solver.Objective().Value()}\n")

            for i in self.N:
                for k in self.lambda_dict[i]:
                    print("s",i,k, self.s[(i, k)].solution_value() + self.p[i,k])

            for i in self.N:
                for f in self.F:
                    print("y",i,f,self.Y[(i, f)].solution_value())

            for i in self.N:
                for k in self.lambda_dict[i]:
                    for j in self.E_dict[k]:
                        print("x",i,j,k, self.X[(i,j,k)].solution_value())

            for i in self.N:
                for l in self.N:
                    for k in intersect(self.lambda_dict[i], self.lambda_dict[l]):
                        print("z",i,l,k,self.Z[(i,l,k)].solution_value())

            for i in self.N:
                for h in self.lambda_dict[i]:
                    for k in intersect(self.lambda_dict[i], self.lambda_dict[l]):
                        print(f"w",i,h,k,self.W[(i,h,k)].solution_value())
        else:
            print("No solution found.")

if __name__ == "__main__":
    dataset = "data.xlsx"
    model = MIP_model(dataset, mip_solver="SCIP")
    model.solve()

