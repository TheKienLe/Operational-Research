from utils import *
import numpy as np

def DNEH_Dipak(num_jobs, num_factories, processing_times):
    
    N = num_jobs
    p = processing_times
    F = num_factories
    
    P = np.zeros(N)
    for i in N:
        P[i] = np.sum(p, axis=1)
    
    job_order = np.argsort(-P)
    
    for i in job_order():
        
    
    
    factory_schedules = [[] for _ in range(F)]
    
    # Insert each job into all possible k + F positions of existing factories
    for job in sorted_jobs:
        lowest_makespan = float('inf')
        best_factory = -1
        best_position = -1
        
        for f in range(F):
            for pos in range(k + F):
                temp_schedule = factory_schedules[f][:pos] + [job] + factory_schedules[f][pos:]
                temp_makespan = calculate_makespan(temp_schedule, processing_times[f])
                
                if temp_makespan < lowest_makespan:
                    lowest_makespan = temp_makespan
                    best_factory = f
                    best_position = pos
        
        factory_schedules[best_factory].insert(best_position, job)
        
        # Perform pairwise interchange
        if len(factory_schedules[best_factory]) > 2:
            for i in range(len(factory_schedules[best_factory])):
                for j in range(i + 1, len(factory_schedules[best_factory])):
                    temp_schedule = factory_schedules[best_factory][:]
                    temp_schedule[i], temp_schedule[j] = temp_schedule[j], temp_schedule[i]
                    temp_makespan = calculate_makespan(temp_schedule, processing_times[best_factory])
                    
                    if temp_makespan < lowest_makespan:
                        factory_schedules[best_factory] = temp_schedule
                        lowest_makespan = temp_makespan
                        
        k += 1
        
    return factory_schedules

def calculate_makespan(schedule, processing_times):
    makespan = [0] * len(schedule)
    
    for i in range(len(schedule)):
        for j in range(len(processing_times)):
            if i == 0:
                makespan[i] += processing_times[j][schedule[i] - 1]
            else:
                makespan[i] = max(makespan[i - 1], makespan[i] + processing_times[j][schedule[i] - 1])
    
    return makespan[-1]

# Example usage
jobs = [1, 2, 3, 4, 5]
processing_times = [
    [2, 4, 6],
    [3, 5, 7],
    [4, 6, 8],
    [5, 7, 9],
    [6, 8, 10]
]

result = DNEH_Dipak(jobs, processing_times)
for i, schedule in enumerate(result):
    print(f"Factory {i + 1}: {schedule}")
