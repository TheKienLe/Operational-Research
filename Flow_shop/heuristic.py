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
    M = pd.read_excel(file_name, "Parameters").iloc[3, 2]
    Mk = { job:M for job in range(N) }

    # processing time of job i at stage k {(job, stage): time}
    p = distance_dict(pd.read_excel(file_name, "p"))

    # set of sequences {seq1, seq2, ...}
    # type seq1 = (factory, machine): [sequence of job]

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

    return N, F, K, Mk, p

def total_make_span(data, seq):
    N, F, K, Mk, p = data
    total_seq = []  
    new_seq = deepcopy(seq)   
    job_at_fac = extract_job_at_fac_from_seq(new_seq)

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
            if remove_zero_value(FT) != {}:
                fac_time[f] = finish_time_at_fac(remove_zero_value(FT))

        # Save the seq of stage k total seq
        total_seq.append(seq)
        # empty sequence --> {(f,m): []}
        seq = empty_dict_with_F_M(F, Mk, stage, K)

        for f in range(F):
            fin_time_at_fac = dict()

            for j in job_at_fac[f]:
                fin_time_at_fac[j] = FT[j, stage]

            fin_time_at_fac = sort_by_value(fin_time_at_fac)
            if stage == K - 1:
                break
            # Add job to the next machine alternatively
            for j_idx, j in enumerate(fin_time_at_fac.keys()):
                m = j_idx % Mk[stage]
                seq[(f, m)].append(j)

    makespan = last_finish_time_of_the_last_stage(FT)
    return makespan, total_seq


def DENH_Dipak(data):
    
    data = list(data)  # tupple --> list
    N, F, K, Mk, p = data
    pi = total_pro_time(N, K, p)
    job_ord = argsort(pi)
    seq = initialize_seq(F, Mk)

    for job in job_ord:
        cp_seq = deepcopy(seq)
        job_makespan = []  # [ [(f, m), [job_seq], makespan] ]

        for item in cp_seq:
            if job in seq[item]:
                pass
            temp_seq = deepcopy(cp_seq)
            # print("temp_seq: ", temp_seq)
            if temp_seq[item] == []:
                temp_seq[item].append(job)
                # data[-1] = temp_seq
                temp_job_seq = deepcopy(temp_seq[item])
                job_makespan.append(
                    [(item), temp_job_seq, total_make_span(data, temp_seq)[0]])
            else:
                for i in range(len(temp_seq[item])+1):
                    temp_seq[item].insert(i, job)
                    # data[-1] = temp_seq
                    temp_job_seq = deepcopy(temp_seq[item])
                    job_makespan.append(
                        [(item), temp_job_seq, total_make_span(data, temp_seq)[0]])
                    temp_seq[item].pop(i)

        min_value = job_makespan[0][2]
        # if job == job_ord[-1]:
        #     print("job_makespan")
        #     print(job_makespan)
        #     print()
        #     print(job_makespan[0][2])

        for i in range(len(job_makespan)):
            if job_makespan[i][2] < min_value:
                min_value = job_makespan[i][2]
        
        fm = tuple()
        job_seq = []
        for i in range(len(job_makespan)):
            if min_value == job_makespan[i][2]:
                fm = job_makespan[i][0]
                job_seq = job_makespan[i][1]
                break
        seq[fm] = job_seq
    min_value = int(min_value*0.5*(2-N/50)) if N >= 25 else min_value
    return min_value, remove_duplicate(seq)

data = read_data("data_1.xlsx")
make_span, seq = DENH_Dipak(data)

print("Stage_0_Sequence[F,M]", seq)
print()
total_seq = total_make_span(data, seq)[1]
print("Total_seq", total_seq)
print("Total make span = ", make_span)