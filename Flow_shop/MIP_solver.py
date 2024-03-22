from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np

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


        ## Range 
        self.N = range(self.n)
        self.S = range(self.s)
        self.F = range(self.f)


        ## Parameter
        # Processing time
        self.p = np.squeeze(
            np.array(pd.read_excel("data.xlsx", "p", index_col=0))
            )

        # Set of stages accessed by job
        lambda_arr = np.array(pd.read_excel("data.xlsx", "lambda"))
        self.lambda_dict = self.arr_to_dict(lambda_arr)

        # Set of parallel machine at stage k 
        E_arr = np.array(pd.read_excel("data.xlsx", "E"))
        self.E_dict = self.arr_to_dict(E_arr)

        # Number of parallel machine at stage k
        self.m = np.squeeze(
            np.array(pd.read_excel("data.xlsx", "m", index_col=0))
            )
        
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
                for k in self.intersect(self.lambda_dict[i], self.lambda_dict[l]):
                    self.Z[(i, l, k)] = self.solver.IntVar(0, 1, "")

        # W_ihk if job immediately after its completion at stage j is to be processed at stage h l
        self.W = {}
        for i in self.N:
            for h in self.lambda_dict[i]:
                for k in self.lambda_dict[i]:
                #    if h!=k:
                       self.W[(i, h, k)] = self.solver.IntVar(0, 1, "") 



    def arr_to_dict(self, arr):
        dict = {}
        for key, value in arr:
            if key in dict:
                dict[key].append(value)
            else:
                dict[key] = [value]

        return dict


    def intersect(self, list1, list2):
        return list(set(list1).intersection(set(list2)))


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
        

        # ct3_sub:
        for i in self.N:
            for index, k in enumerate(self.lambda_dict[i][:-1]):
                for h in self.lambda_dict[i]:
                    if h == self.lambda_dict[i][index+1]:
                        self.solver.Add(
                            self.W[(i, k, h)] == 1,
                            "ct3_sub")
                    else:
                        self.solver.Add(
                            self.W[(i, k, h)] == 0,
                            "ct3_sub")
                        
        
        # ct3: the next operation can only be started after the previous one is completed
        # Consider eliminate L
        # Add condition h > k
        for i in self.N:
            for index, k in enumerate(self.lambda_dict[i][:-1]):
                for h in self.lambda_dict[i]:
                    for j in self.E_dict[k]:
                        if h == self.lambda_dict[i][index+1]:
                            self.solver.Add(
                                self.s[(i,h)] - (self.s[(i,k)] + self.p[i, k]) - self.L*(1 - self.W[(i, k, h)]) >=0,
                                "ct3")
                        
        # ct4: 
        for i in self.N:
            for l in self.N:
                if i != l:
                    for k in self.intersect(self.lambda_dict[i], self.lambda_dict[l]):
                        self.solver.Add(
                            self.Z[(l,i,k)] + self.Z[(i,l,k)] <= 1,
                        "ct4")
                        self.solver.Add(
                            self.Z[(l,i,k)] + self.Z[(i,l,k)] >= 1,
                        "ct4_sub")
        
        # ct5: thời gian bắt đầu của job sau sẽ lớn thời gian kết thúc job trước
        # Change from -L to +L
        for i in self.N:
            for l in self.N:
                if i != l:
                    for k in self.intersect(self.lambda_dict[i], self.lambda_dict[l]):
                        for j in self.E_dict[k]:
                            for f in self.F:
                                self.solver.Add(
                                    self.s[(l,k)] - (self.s[(i,k)] + self.p[i, k]) \
                                        + self.L*(5 - self.Z[(l,i,k)] - self.X[(i,j,k)] 
                                                  - self.X[(l,j,k)] - self.Y[(i,f)] - self.Y[(l,f)]) >=0,
                                        # + self.L*(1 - self.Z[(l,i,k)]) >=0,
                        "ct5")

        # ct6: 
        self.f = self.solver.NumVar(0, self.solver.infinity(), "")
        for i in self.N:
            for k in self.lambda_dict[i]:
                self.solver.Add(self.f >= (self.s[(i, k)] + self.p[i, k]))


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
                        print(f"s{i,k} =", self.s[(i, k)].solution_value() )
                        # print(f"p{i,k} =", self.p[i, k])

            for i in self.N:
                for f in self.F:
                    if f ==1:
                        print(f"y{i,f} =", self.Y[(i, f)].solution_value())

            # for i in self.N:
            #     for k in self.lambda_dict[i]:
            #         for j in self.E_dict[k]:
            #             print(f"x{i,j,k} =", self.X[(i,j,k)].solution_value())
                        
            # for i in self.N:
            #     for k in self.lambda_dict[i]:
            #         for h in self.lambda_dict[i]:
            #             if self.W[(i,k,h)].solution_value() == 1:
            #                 print(f"w{i,k,h} =", self.W[(i,k,h)].solution_value())
                            
            # for i in self.N:
            #     for l in self.N:
            #         for k in self.intersect(self.lambda_dict[i], self.lambda_dict[l]):
            #             print(f"Z{i,l,k} =", self.Z[(i,l,k)].solution_value())

            # 
            # print("y =", self.Y.solution_value())
        else:
            print("No solution found.")

if __name__ == "__main__":
    dataset = "data.xlsx"
    model = MIP_model(dataset, mip_solver="SCIP")
    model.solve()

