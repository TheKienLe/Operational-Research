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

def compute_total_processing_time(jobs, processing_times):
    total_processing_time = [0] * len(jobs)
    for i in range(len(jobs)):
        for j in range(len(processing_times[i])):
            total_processing_time[i] += processing_times[i][j]
    return total_processing_time

