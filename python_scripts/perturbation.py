import numpy as np


class Perturbator:
    def __init__(self, name = "Perturbation"):
       self.name = name

    def __call__(self, input):
        return input




class NormalJittering(Perturbator):

    def __init__(self, bias, standard_deviation, name = "Jittering"):
        super().__init__(name=name)
        self.bias = bias
        self.standard_deviation = standard_deviation

    def __call__(self, input):
        return input + np.random.normal(self.bias, self.standard_deviation , input.shape)
    