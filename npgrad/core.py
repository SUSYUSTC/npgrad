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


def getdata(x):
    if isinstance(x, ndarray):
        return x.data
    for typ in [tuple, list]:
        if isinstance(x, typ):
            return typ([getdata(item) for item in x])
    return x


def getgrad(x):
    if isinstance(x, ndarray):
        return x.grad
    for typ in [tuple, list]:
        if isinstance(x, typ):
            result = [getgrad(item) for item in x]
            for item in result:
                if item is NoGrad:
                    return NoGrad
            return typ(result)
    return NoGrad


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

    def __ge__(self, other):
        other = asarray(other)
        return ndarray(self.data >= other.data)

    def __gt__(self, other):
        other = asarray(other)
        return ndarray(self.data > other.data)

    def __le__(self, other):
        other = asarray(other)
        return ndarray(self.data <= other.data)

    def __lt__(self, other):
        other = asarray(other)
        return ndarray(self.data < other.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return 'data:\n' + self.data.__repr__() + '\ngrad:\n' + self.grad.__repr__()

    def __getitem__(self, index):
        index = getdata(index)
        return ndarray(self.data[index], grad=self.grad[index])

    def __setitem__(self, index, value):
        index = getdata(index)
        value = asarray(value)
        self.data[index] = value.data
        if (self.grad is NoGrad) and (value.grad is not NoGrad):
            self.grad = np.zeros_like(self.data)
        if (self.grad is not NoGrad) and (value.grad is NoGrad):
            self.grad[index] = 0
        else:
            self.grad[index] = value.grad

    @property
    def T(self):
        return self.transpose()

    @property
    def shape(self):
        return self.data.shape

    def __pow__(self, n):
        return power(self, n)

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


def array(array, *args, **kwargs):
    data_array = getdata(array)
    grad_array = getgrad(array)
    data_nparray = np.array(data_array, *args, **kwargs)
    if grad_array is NoGrad:
        return ndarray(data_nparray)
    else:
        grad_nparray = np.array(grad_array, *args, **kwargs)
        return ndarray(data_nparray, grad=grad_nparray)


def diag(x, *args, **kwargs):
    return ndarray(np.diag(x.data, *args, **kwargs), np.diag(x.grad, *args, **kwargs))


def transpose(x, *args, **kwargs):
    return ndarray(np.transpose(x.data, *args, **kwargs), np.transpose(x.grad, *args, **kwargs))


def sum(x, *args, **kwargs):
    if x.grad is NoGrad:
        grad = NoGrad
    else:
        grad = np.sum(x.grad, *args, **kwargs)
    return ndarray(np.sum(x.data, *args, **kwargs), grad=grad)


def mean(x, *args, **kwargs):
    if x.grad is NoGrad:
        grad = NoGrad
    else:
        grad = np.mean(x.grad, *args, **kwargs)
    return ndarray(np.mean(x.data, *args, **kwargs), grad=grad)


def repeat(x, *args, **kwargs):
    if x.grad is NoGrad:
        grad = NoGrad
    else:
        grad = np.repeat(x.grad, *args, **kwargs)
    return ndarray(np.repeat(x.data, *args, **kwargs), grad=grad)


def moveaxis(x, *args, **kwargs):
    if x.grad is NoGrad:
        grad = NoGrad
    else:
        grad = np.moveaxis(x.grad, *args, **kwargs)
    return ndarray(np.moveaxis(x.data, *args, **kwargs), grad=grad)


def sin(x):
    x = asarray(x)
    return ndarray(np.sin(x.data), grad=x.grad * np.cos(x.data))


def cos(x):
    x = asarray(x)
    return ndarray(np.cos(x.data), grad=-x.grad * np.sin(x.data))


def exp(x):
    x = asarray(x)
    return ndarray(np.exp(x.data), grad=x.grad * np.exp(x.data))


def log(x):
    x = asarray(x)
    return ndarray(np.log(x.data), grad=x.grad / x.data)


def square(x):
    x = asarray(x)
    return ndarray(np.square(x.data), grad=x.grad * x.data * 2)


def sqrt(x):
    x = asarray(x)
    return ndarray(np.sqrt(x.data), grad=x.grad * 0.5 / np.sqrt(x.data))


def power(x, n):
    x = asarray(x)
    return ndarray(np.power(x.data, n), grad=x.grad * np.power(x.data, n - 1) * n)


def sign(x):
    x = asarray(x)
    return ndarray(np.sign(x.data))


def abs(x):
    x = asarray(x)
    sign = np.sign(x.data)
    return ndarray(x.data * sign, grad=x.grad * sign)


def max(x, *args, **kwargs):
    x = asarray(x)
    arg = np.argmax(x, *args, **kwargs)
    return ndarray(x.data[arg], grad=x.grad[arg])


def min(x, *args, **kwargs):
    x = asarray(x)
    arg = np.argmin(x, *args, **kwargs)
    return ndarray(x.data[arg], grad=x.grad[arg])


def argsort(x, *args, **kwargs):
    x = asarray(x)
    return ndarray(np.argsort(x.data, *args, **kwargs))


def dot(a, b):
    data = a.data.dot(b.data)
    grad = a.grad.dot(b.data) + a.data.dot(b.grad)
    return ndarray(data, grad)


def triu_indices(*args, **kwargs):
    a, b = np.triu_indices(*args, **kwargs)
    return ndarray(a), ndarray(b)


def tril_indices(*args, **kwargs):
    a, b = np.tril_indices(*args, **kwargs)
    return ndarray(a), ndarray(b)


def arange(*args, **kwargs):
    return ndarray(np.arange(*args, **kwargs))


def zeros(*args, **kwargs):
    return ndarray(np.zeros(*args, **kwargs))


def zeros_like(*args, **kwargs):
    return ndarray(np.zeros_like(*args, **kwargs))


newaxis = np.newaxis
