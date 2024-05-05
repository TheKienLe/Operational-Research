def df_to_list(df, *args):
    result = dict()
    for i in df.index:
        if len(args) == 2:
            result[df[args[0]][i]] = df[args[1]][i]
        elif (len(args)) == 3:
            result[(df[args[0]][i], df[args[1]][i])] = df[args[2]][i]
        else:
            print("Missing or too much args")
            return
    return result
