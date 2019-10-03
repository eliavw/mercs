import warnings


def debug_print(msg, level=1, V=0, warn=False):

    if V >= level:
        if warn:
            warnings.warn(msg)
        else:
            print(msg)
    return
