from functools import lru_cache

def o(f1, f2):

    def sequential_composition(*function_arguments):
        return f1(f2(*function_arguments))

    return sequential_composition


def x(*functions, return_type=tuple):
    def parallel_composition(*function_arguments):
        if len(function_arguments) == 1:
            res = return_type([f(function_arguments[0]) for f in functions])
            return res
        elif len(function_arguments) == len(functions):
            res = return_type([f(x) for f, x in zip(functions, function_arguments)])
            return res

    return parallel_composition
