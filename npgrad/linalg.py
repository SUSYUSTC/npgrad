import numpy as np
import numpy.linalg as LA
from .core import ndarray, NoGrad, asarray


def norm(x, *args, **kwargs):
    x = asarray(x)
    norm = LA.norm(x.data, *args, **kwargs)
    if x.grad is not NoGrad:
        grad = np.sum(x.grad*x.data, *args, **kwargs) / norm
    else:
        grad = NoGrad
    return ndarray(norm, grad=grad)
