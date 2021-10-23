import numpy as np


class NoGradType(object):
    def __init__(self):
        pass

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other

    def __sub__(self, other):
        return -other

    def __rsub__(self, other):
        return other

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __truediv__(self, other):
        return self

    def __rtruediv__(self, other):
        raise NotImplementedError

    def __neg__(self):
        return self

    def __getitem__(self, index):
        return self

    def __setitem__(self, index, value):
        pass

    def __repr__(self):
        return '<NoGrad>'

    def copy(self):
        return self

    @property
    def T(self):
        return self

    def transpose(self):
        return self

    def dot(self, other):
        return self


NoGrad = NoGradType()


def asarray(x):
    if isinstance(x, ndarray):
        return x
    else:
        return ndarray(x)


def np_nograd_decorator2(func):
    def wrapper(*args, **kwargs):
        if args[1] is NoGrad:
            return NoGrad
        else:
            return func(*args, **kwargs)
    return wrapper


def add_grad(a, b):
    if b is NoGrad:
        return a
    else:
        return a + b


class ndarray(object):
    def __init__(self, data, grad=NoGrad):
        self.data = data
        self.grad = grad

    def set_grad(self, grad):
        self.grad = grad.data

    def __add__(self, other):
        other = asarray(other)
        return ndarray(self.data + other.data, grad=add_grad(self.grad, other.grad))

    def __radd__(self, other):
        other = asarray(other)
        return ndarray(self.data + other.data, grad=add_grad(self.grad, other.grad))

    def __sub__(self, other):
        other = asarray(other)
        return ndarray(self.data - other.data, grad=add_grad(self.grad, -other.grad))

    def __rsub__(self, other):
        other = asarray(other)
        return ndarray(other.data - self.data, grad=add_grad(-self.grad, -other.grad))

    def __mul__(self, other):
        other = asarray(other)
        return ndarray(self.data * other.data, grad=add_grad(self.grad * other.data, other.grad * self.data))

    def __rmul__(self, other):
        other = asarray(other)
        return ndarray(self.data * other.data, grad=add_grad(self.grad * other.data, other.grad * self.data))

    def __truediv__(self, other):
        other = asarray(other)
        return ndarray(self.data / other.data, grad=add_grad(self.grad / other.data, -other.grad * self.data / other.data / other.data))

    def __rtruediv__(self, other):
        other = asarray(other)
        return ndarray(other.data / self.data, grad=add_grad(other.grad / self.data, -self.grad * other.data / self.data / self.data))

    def __neg__(self):
        return self * -1

    def __repr__(self):
        return 'data:\n' + self.data.__repr__() + '\ngrad:\n' + self.grad.__repr__()

    def __getitem__(self, index):
        index = asarray(index)
        return ndarray(self.data[index.data], grad=self.grad[index.data])

    def __setitem__(self, index, value):
        index = asarray(index)
        self.data[index.data] = value.data
        self.grad[index.data] = value.grad

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        return transpose(self)

    def dot(self, other):
        return dot(self, other)

    def copy(self):
        return ndarray(self.data.copy(), self.grad.copy())


def np_nograd_decorator(func):
    def wrapper(*args, **kwargs):
        if args[0] is NoGrad:
            return NoGrad
        else:
            return func(*args, **kwargs)
    return wrapper


np.diag = np_nograd_decorator(np.diag)
np.transpose = np_nograd_decorator(np.transpose)
np.sum = np_nograd_decorator(np.sum)
np.mean = np_nograd_decorator(np.mean)


def array(data, *args, **kwargs):
    npdata = np.array(data, *args, **kwargs)
    return ndarray(npdata)


def diag(x, *args, **kwargs):
    return ndarray(np.diag(x.data, *args, **kwargs), np.diag(x.grad, *args, **kwargs))


def transpose(x, *args, **kwargs):
    return ndarray(np.transpose(x.data, *args, **kwargs), np.transpose(x.grad, *args, **kwargs))


def sum(x, *args, **kwargs):
    return ndarray(np.sum(x.data, *args, **kwargs), np.sum(x.grad, *args, **kwargs))


def mean(x, *args, **kwargs):
    return ndarray(np.mean(x.data, *args, **kwargs), np.mean(x.grad, *args, **kwargs))


def sin(x):
    return ndarray(np.sin(x.data), grad=np.cos(x.data) * x.grad)


def cos(x):
    return ndarray(np.cos(x.data), grad=-np.sin(x.data) * x.grad)


def exp(x):
    return ndarray(np.exp(x.data), grad=np.exp(x.data) * x.grad)


def log(x):
    return ndarray(np.log(x.data), grad=1 / x.data * x.grad)


def square(x):
    return ndarray(np.square(x.data), grad=x.data * 2 * x.grad)


def sqrt(x):
    return ndarray(np.sqrt(x.data), grad=0.5 / np.sqrt(x.data) * x.grad)


def abs(x):
    sign = np.sign(x.data)
    return ndarray(x.data * sign, grad=x.grad * sign)


def max(x, *args, **kwargs):
    arg = np.argmax(x, *args, **kwargs)
    return ndarray(x.data[arg], grad=x.grad[arg])


def min(x, *args, **kwargs):
    arg = np.argmin(x, *args, **kwargs)
    return ndarray(x.data[arg], grad=x.grad[arg])


def argsort(x, *args, **kwargs):
    return ndarray(np.argsort(x.data, *args, **kwargs))


def dot(a, b):
    data = a.data.dot(b.data)
    grad = a.grad.dot(b.data) + a.data.dot(b.grad)
    return ndarray(data, grad)
