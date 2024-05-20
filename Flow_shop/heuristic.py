import pandas as pd
import numpy as np
from copy import deepcopy
from utils import *


# read data
def read_data(file_name):
    # total of job
    N = pd.read_excel(file_name, "Parameters").iloc[0, 2]
    # total of factory
    F = pd.read_excel(file_name, "Parameters").iloc[2, 2]
    # total of stage per factory
    K = pd.read_excel(file_name, "Parameters").iloc[1, 2]

    # number of paralell machine at stage k {stage: qty}
    Mk = summary_df(pd.read_excel(file_name, "E"), "Stage")

    # processing time of job i at stage k {(job, stage): time}
    p = distance_dict(pd.read_excel(file_name, "p"))

    # set of sequences {seq1, seq2, ...}
    # type seq1 = (factory, machine): [sequence of job]
    seq = {(0, 0): [1, 3], (1, 0): [0, 2], (1, 1): [
        4, 5], (0, 1): [], (0, 2): [], (1, 2): []}

    # set of starting time for job i at stage k
    # {(job, stage): starting_time}
    ST = dict()

    # set of Finishing time for job i at stage k
    # {(job, stage): Finishing_time}
    FT = dict()

    # Finishing time at factory
    # {(factory, job): Finishing_time}    {0: 15, 1: 16}
    fin_time_at_stage = dict()

    # job at each factory
    # nested list [[1,2,5], [4,3]]
    job_at_fac = []

    return N, F, K, Mk, p, seq


data = read_data("data.xlsx")


# total make_span func
def total_make_span(data):
    N, F, K, Mk, p, seq = data

    total_seq = []
    new_seq = deepcopy(seq)   # copy function nhu lol --> deepcopy()
    job_at_fac = extract_job_at_fac_from_seq(new_seq)
    print(job_at_fac)
    ST = nested_to_dict(np.zeros([N, K]))
    FT = nested_to_dict(np.zeros([N, K]))

    fac_time = dict()
    # test
    for stage in range(K):
        for f in range(F):
            for m in range(Mk[stage]):
                for j_idx, j in enumerate(seq[(f, m)]):
                    FT[(j, stage)] = ST[(j, stage)]
                    for i in seq[(f, m)][:j_idx+1]:
                        FT[(j, stage)] += p[(i, stage)]
                        ST[(j, stage + 1)] = FT[(j, stage)]
            fac_time[f] = finish_time_at_fac(remove_zero_value(FT))

        # Save the seq of stage k total seq
        total_seq.append(seq)
        # empty sequence --> {(f,m): []}
        seq = empty_dict_with_F_M(F, Mk, stage, K)

        for f in range(F):
            fin_time_at_fac = dict()

            for j in job_at_fac[f]:
                fin_time_at_fac[j] = FT[j, stage]

            print(fin_time_at_fac)
            fin_time_at_fac = sort_by_value(fin_time_at_fac)
            if stage == K - 1:
                break
            # Add job to the next machine alternatively
            for j_idx, j in enumerate(fin_time_at_fac.keys()):
                m = j_idx % Mk[stage]
                seq[(f, m)].append(j)

    makespan = last_finish_time_of_the_last_stage(FT)
    print(len(total_seq))
    return makespan, total_seq


if __name__ == "__main__":
    print(total_make_span(data))
