import numpy as np

#TODO: return copies


class Parameter(object):
    def __init__(self, n):
        self.n = n


n_grad = Parameter(1)


def get_n_grad():
    return n_grad.n


def set_n_grad(n):
    n_grad.n = n


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


def add_grad(a, b):
    if b is NoGrad:
        return a
    else:
        return a + b


def add_vector(vec1, vec2):
    assert isinstance(vec1, GradVector)
    assert isinstance(vec2, GradVector)
    return GradVector([add_grad(x1, x2) for x1, x2 in zip(vec1.vec, vec2.vec)])


class GradVector(object):
    def __init__(self, vec):
        assert isinstance(vec, list)
        self.vec = vec

    def __add__(self, other):
        if not isinstance(other, GradVector):
            return GradVector([x + other for x in self.vec])
        else:
            return add_vector(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return -self.__add__(-other)

    def __mul__(self, other):
        assert not isinstance(other, GradVector)
        return GradVector([x * other for x in self.vec])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def __neg__(self):
        return GradVector([-x for x in self.vec])

    def __repr__(self):
        return 'GradVector\n' + self.vec.__repr__()

    def __len__(self):
        return len(self.vec)

    def __getitem__(self, index):
        index = getdata(index)
        return GradVector([item[index] for item in self.vec])

    def copy(self):
        return GradVector([x.copy() for x in self.vec])

    @property
    def T(self):
        return GradVector([x.T for x in self.vec])

    def transpose(self):
        return GradVector([x.T for x in self.vec])


def get_empty_vector():
    return GradVector([NoGrad for i in range(get_n_grad())])


def getdata(x):
    if isinstance(x, ndarray):
        return x.data
    for typ in [tuple, list]:
        if isinstance(x, typ):
            return typ([getdata(item) for item in x])
    return x


def combine_nograd(x):
    for item in x:
        if item is NoGrad:
            return NoGrad
    return x


def getgrad(x):
    if isinstance(x, GradVector):
        return x
    if isinstance(x, ndarray):
        return x.grad
    for typ in [tuple, list]:
        if isinstance(x, typ):
            result = [getgrad(item) for item in x]
            if len(result) == 0:
                return GradVector([])
            n = len(result[0])
            return GradVector([combine_nograd(typ([item.vec[i] for item in result])) for i in range(n)])
    return get_empty_vector()


def asarray(x):
    if isinstance(x, ndarray):
        return x
    else:
        return ndarray(x)


class ndarray(object):
    def __init__(self, data, grad=None):
        self.data = data
        if grad is None:
            self.grad = GradVector([NoGrad for i in range(get_n_grad())])
        else:
            self.grad = grad

    def set_grad(self, grads):
        self.grad = GradVector([grad.data for grad in grads])

    def __add__(self, other):
        other = asarray(other)
        return ndarray(self.data + other.data, grad=self.grad + other.grad)

    def __radd__(self, other):
        other = asarray(other)
        return ndarray(self.data + other.data, grad=self.grad + other.grad)

    def __sub__(self, other):
        other = asarray(other)
        return ndarray(self.data - other.data, grad=self.grad - other.grad)

    def __rsub__(self, other):
        other = asarray(other)
        return ndarray(other.data - self.data, grad=other.grad - self.grad)

    def __mul__(self, other):
        other = asarray(other)
        return ndarray(self.data * other.data, grad=self.grad * other.data + other.grad * self.data)

    def __rmul__(self, other):
        other = asarray(other)
        return ndarray(self.data * other.data, grad=self.grad * other.data + other.grad * self.data)

    def __truediv__(self, other):
        other = asarray(other)
        return ndarray(self.data / other.data, grad=self.grad / other.data - other.grad * self.data / other.data / other.data)

    def __rtruediv__(self, other):
        other = asarray(other)
        return ndarray(other.data / self.data, grad=other.grad / self.data - self.grad * other.data / self.data / self.data)

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

        for i in range(len(self.grad)):
            x = self.grad.vec[i]
            v = value.grad.vec[i]
            if (x is NoGrad) and (v is not NoGrad):
                self.grad.vec[i] = np.zeros_like(self.data)
            x = self.grad.vec[i]
            if (x is not NoGrad) and (v is NoGrad):
                x[index] = 0
            else:
                x[index] = v

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


def array(array, *args, **kwargs):
    data_array = getdata(array)
    grad_array = getgrad(array)
    data_array = np.array(data_array, *args, **kwargs)
    for i in range(len(grad_array)):
        if grad_array.vec[i] is not NoGrad:
            grad_array.vec[i] = np.array(grad_array.vec[i])
    return ndarray(data_array, grad=grad_array)


def np_nograd_decorator(func):
    def wrapper(*args, **kwargs):
        if args[0] is NoGrad:
            return NoGrad
        else:
            return func(*args, **kwargs)
    return wrapper


#np.diag = np_nograd_decorator(np.diag)
#np.transpose = np_nograd_decorator(np.transpose)
#np.sum = np_nograd_decorator(np.sum)
#np.mean = np_nograd_decorator(np.mean)
#np.repeat = np_nograd_decorator(np.mean)


def vectormap1(func, x, *args, **kwargs):
    assert isinstance(x, GradVector)
    result = []
    for item in x.vec:
        if item is NoGrad:
            result.append(NoGrad)
        else:
            result.append(func(item, *args, **kwargs))
    return GradVector(result)


def apply_type_decorator(func):
    def wrapper(x, *args, **kwargs):
        return ndarray(func(x.data, *args, **kwargs), vectormap1(func, x.grad, *args, **kwargs))
    return wrapper


mean = apply_type_decorator(np.mean)
sum = apply_type_decorator(np.sum)
transpose = apply_type_decorator(np.transpose)
diag = apply_type_decorator(np.diag)
repeat = apply_type_decorator(np.repeat)
moveaxis = apply_type_decorator(np.moveaxis)


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
    grad = GradVector([item.dot(b.data) for item in a.grad.vec]) + GradVector([a.data.dot(item) for item in b.grad.vec])
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
