from typing import Union

import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.hp.continuous import Float, Integer
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace

from PyEvo.mutation.abstract_mutation import Mutation


class UniformMutation(Mutation):
    """
    Class representing a mutation operation that introduces noise to float and integer hyperparameters.
    The mutation perturbs hyperparameters using random values drawn from a uniform distribution.
    """

    def __init__(self, low: Union[int, float], high: Union[int, float], hp_type: str, prob: float):
        assert hp_type == "int" or hp_type == "float", f"Illegal hp_type {hp_type}. It should be 'int', 'float'!"
        assert low < high, f"Illegal low {low} or high {high}. It should be that low < high!"
        super().__init__(prob)
        self._hp_type = hp_type
        self._low = low
        self._high = high

    def _mutate(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs,
    ) -> list[HyperparameterConfiguration]:
        if self._hp_type == "float":
            hp_type = Float
            data_type = (float, np.float_)
            sampling = random.uniform
            epsilon = 1e-10
        else:
            hp_type = Integer
            data_type = (int, np.int_)
            sampling = random.random_integers
            epsilon = 1

        for ind in pop:
            for key, hp in ind.items():
                if isinstance(cs[key], hp_type) and random.random() <= self._prob:
                    if isinstance(hp, data_type):
                        ind[key] += sampling(low=self._low, high=self._high)
                        if ind[key] < cs[key].lb:
                            ind[key] = cs[key].lb
                        elif ind[key] >= cs[key].ub:
                            ind[key] = cs[key].ub - epsilon
                    elif isinstance(hp, np.ndarray):
                        ind[key] += sampling(low=self._low, high=self._high, size=hp.shape)
                        ind[key][ind[key] < cs[key].lb] = cs[key].lb
                        ind[key][ind[key] >= cs[key].ub] = cs[key].ub - epsilon
        return pop
