def distance_dict(df):
    result = dict()
    rows = df.shape[0]
    columns = df.shape[1]
    for i in range(rows):
        for j in range(1, columns):
            result[(i, j - 1)] = df.iloc[i, j]
    return result


def summary_df(df, *args):
    result = dict()
    for ind in df.index:
        temp = df[args[0]][ind]
        if temp not in result:
            result[temp] = 1
        else:
            result[temp] += 1
    return result


# nested list to dictionary
def nested_to_dict(lst):
    result = dict()
    columns = len(lst)  # represent job order
    rows = len(lst[0])  # represent stage order
    for i in range(columns):
        for j in range(rows):
            result[(i, j)] = lst[i][j]

    return result

# remove zeros value in dictionary


def remove_zero_value(dictionary):
    result = dict()
    for item in dictionary:
        if dictionary[item] != 0.0:
            result[item] = dictionary[item]
    del dictionary

    return result


# finish time at each factory
def finish_time_at_fac(FT):
    maxi = max(FT.values())
    return maxi


# machine availability 1: occupied, 0: vacant
def machine_vacancy(F, K, Mk):
    # generate
    result = dict()
    for f in range(F):
        for k in range(K):
            for m in range(Mk[k]):
                result[(f, k, m)] = 0
    return result


# finishing time of machine at stage
def stage_machine_finish(K, Mk):
    result = dict()
    for s in range(K):
        for m in range(Mk[s]):
            result[(K, m)] = 0
    return result


# extract job for factory from sequence
def extract_job_at_fac_from_seq(seq):
    job_at_fac = dict()
    for item in seq.keys():
        if item[0] not in job_at_fac:
            job_at_fac[item[0]] = seq[item]
        else:
            job_at_fac[item[0]] += seq[item]
    return job_at_fac

# def extract_job_at_fac_from_seq(seq):
#     job_at_fac


def sort_by_value(data):  # data is dictionary

    sorted(data.items(), key=lambda kv:
           (kv[1], kv[0]))

    return data


print(sort_by_value({0: 5.0, 2: 7.0, 4: 7.0, 5: 10.0}))


def empty_dict_with_F_M(F, Mk, stage, K):
    result = dict()
    for f in range(F):
        if stage > K - 2:
            return result
        for m in range(Mk[stage+1]):
            result[(f, m)] = []
    return result


def last_finish_time_of_the_last_stage(FT):
    return max(FT.values())
