import numpy as np
import numpy.linalg as LA
from .core import ndarray, NoGrad, asarray, GradVector


def norm(x, *args, **kwargs):
    x = asarray(x)
    norm = LA.norm(x.data, *args, **kwargs)
    grads = []
    for grad in x.grad.vec:
        if grad is not NoGrad:
            grads.append(np.sum(grad*x.data, *args, **kwargs) / norm)
        else:
            grads.append(NoGrad)
    return ndarray(norm, grad=GradVector(grads))
