import numpy as np
from .core import ndarray


def define_func(name):
    np_func = getattr(np.random, name)

    def this_func(*args, **kwargs):
        return ndarray(np_func(*args, **kwargs))

    return this_func


seed = define_func('seed')
random = define_func('random')
normal = define_func('normal')
rand = define_func('rand')
randint = define_func('randint')
lognormal = define_func('lognormal')
