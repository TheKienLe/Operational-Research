from ortools.sat.python import cp_model


def flow_shop(all_jobs, all_stages, all_factories):
    # process time data
    p = [
        [5, 2, 2],
        [4, 3, 3],
        [2, 5, 8],
        [5, 5, 6],
        [7, 4, 4],
        [3, 3, 2]
    ]

    # machine at each stage
    E = [3, 2, 2]
    L = 10000

    all_machines = sum(E)

    max_operation_time = sum([sum(job_time) for job_time in p])

    model = cp_model.CpModel()

    s = {}
    for i in range(all_jobs):
        for k in range(all_stages):
            s[(i, k)] = model.new_int_var(
                0, max_operation_time, f"Job {i} in stage {k} start")

    Y = {}
    for i in range(all_jobs):
        for f in range(all_factories):
            Y[(i, f)] = model.new_bool_var(
                f"Job {i} factory {f}")

    X = {}
    for i in range(all_jobs):
        for j in range(all_machines):
            for k in range(all_stages):
                X[(i, j, k)] = model.new_bool_var(
                    f"Job {i} Machine {j} Stage {k}")

    Z = {}
    for i in range(all_jobs):
        for l in range(all_jobs):
            for k in range(all_stages):
                Z[(i, l, k)] = model.new_bool_var(
                    f"Job {i} Job {l} Stage {k}")
    W = {}
    for i in range(all_jobs):
        for l in range(all_stages):
            for k in range(all_stages):
                W[(i, l, k)] = model.new_bool_var(
                    f"Job {i} Stage {l} Stage {k}")

    # add constraint
    model.add_exactly_one(Y[(i, f)] for i in range(all_jobs)
                          for f in range(all_factories))

    model.add_exactly_one(X[(i, j, k)]
                          for i in range(all_jobs) for j in range(all_machines) for k in range(all_stages))

    model.add_linear_constraint(
        s[(i, k)] - (s[(i, k)] + p[i][k]) - L*(1-W[(i, k, k)]), 0, 10000)

    model.add_linear_constraint(s[(i, k)] - (s[(i, k)] + p[i][k]) - L*(
        5-Z[((i, l, k))]-X[(i, j, k)]-X[(l, j, k)]-Y[(i, f)]-Y[(l, f)]), 0, 1000)

    model.add_linear_constraint(Z[(i, l, k)] + Z[(l, i, k)], 0, 1)

    # makespan objective
    obj = model.new_int_var(0, max_operation_time, "make_span")
    model.add_max_equality(obj, [s[(i, k)] + p[i][k]
                           for i in range(all_jobs) for j in range(all_stages)])

    model.minimize(obj)

    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Found")
    else:
        print("No solution")


if __name__ == "__main__":
    flow_shop(6, 3, 2)
