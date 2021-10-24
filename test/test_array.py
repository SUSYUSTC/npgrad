import npgrad as np
import numpy
import unittest


class Test(unittest.TestCase):
    def test(self):
        np.random.seed(0)
        a = np.random.random((3, 3))
        b = np.random.random((3, 3))
        c = np.random.random((3, 3))
        A = np.random.random((3, 3))
        A.set_grad([np.random.random((3, 3))])
        B = np.random.random((3, 3))
        B.set_grad([np.random.random((3, 3))])
        C = np.random.random((3, 3))
        C.set_grad([np.random.random((3, 3))])

        def func(A, B, C):
            D = np.abs(np.sqrt(A).dot(np.square(B)))[:, None, :]
            E = 0.5 + (-np.cos(B).T * np.exp(D) / a + b.T) * np.sin(c) * 2.0
            return np.linalg.norm(E)

        E = func(A, B, C)
        epsilon = 1e-5
        Ep = func(A + A.grad.vec[0] * epsilon, B + B.grad.vec[0] * epsilon, C + C.grad.vec[0] * epsilon)
        En = func(A - A.grad.vec[0] * epsilon, B - B.grad.vec[0] * epsilon, C - C.grad.vec[0] * epsilon)
        dE = (Ep - En) / (epsilon * 2)
        error = numpy.max(numpy.abs(E.grad.vec[0] - dE.data))
        print(error)
        self.assertTrue(error < 1e-8)
