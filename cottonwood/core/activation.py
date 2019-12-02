import numpy as np

# All of these need to be able to handle 2D numpy arrays as inputs.


class Logistic(object):
    @staticmethod
    def __str__():
        return "logistic"

    @staticmethod
    def calc(v):
        return 1 / (1 + np.exp(-v))

    @staticmethod
    def calc_d(v):
        logistic = 1 / (1 + np.exp(-v))
        return logistic * (1 - logistic)


class ReLU(object):
    @staticmethod
    def __str__():
        return "ReLU"

    @staticmethod
    def calc(v):
        return np.maximum(0, v)

    @staticmethod
    def calc_d(v):
        derivative = np.zeros(v.shape)
        derivative[np.where(v > 0)] = 1
        return derivative


class Sigmoid(Logistic):
    @staticmethod
    def __str__():
        return "sigmoid"


class Tanh(object):
    def __init__(self):
        # Including this class attribute lets Tanh re-use its results.
        # Caching the result in this way speeds up the derivative
        # calculation on the bakward pass.
        self.last_calc_result = None

    @staticmethod
    def __str__():
        return "hyperbolic tangent"

    def calc(self, v):
        self.last_calc_result = np.tanh(v)
        return self.last_calc_result

    def calc_d(self, v):
        return 1 - self.last_calc_result ** 2


