from abc import ABC, abstractmethod

import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.hp.continuous import Float
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace

from PyEvo.mutation.abstract_mutation import Mutation


class GaussianMutation(Mutation):
    """
    Class representing a mutation operation that introduces noise to float hyperparameters.
    The mutation perturbs hyperparameters using random values drawn from a normal distribution.

        Args:
            mean (float):
                Mu of the normal distribution N(mu, sigma)

            std (float):
                Sigma of the normal distribution N(mu, sigma)

            prob (float):
                Probability that the mutation occurs
    """

    def __init__(self, mean: float, std: float, prob: float):
        assert std > 0.0, f"Illegal std {std}. The argument should be higher than 0.0!"
        super().__init__(prob)
        self._mean = mean
        self._std = std

    def _mutate(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs,
    ) -> list[HyperparameterConfiguration]:
        for ind in pop:
            for key, value in ind.items():
                hp = cs[key]
                if isinstance(hp, Float) and random.random() <= self._prob:
                    value = hp.adjust_configuration(
                        value + self._mean + self._std * random.normal(loc=0, scale=1, size=hp.get_shape()))
                    ind[key] = value
        return pop


class DecayGaussianMutation(GaussianMutation, ABC):
    """
    TODO: Add Documentation
    """

    def __init__(self, mean: float, min_std: float, max_std: float, prob: float):
        assert max_std > 0.0, f"Illegal max_std {max_std}. The argument should be higher than 0.0!"
        assert min_std > 0.0, f"Illegal min_std {min_std}. The argument should be higher than 0.0!"

        super().__init__(mean, max_std, prob)
        self._min_std = min_std
        self._max_std = max_std
        self._steps = steps

    @abstractmethod
    def _update(self):
        """
        TODO: Add Documentation
        """
        pass

    def _mutate(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs,
    ) -> list[HyperparameterConfiguration]:
        new_pop = super()._mutate(random, cs, pop, fitness, optimizer, **kwargs)
        self._update()
        return new_pop


class LinearDecayGaussianMutation(DecayGaussianMutation):
    """
    # TODO: Add Documentation
    """

    def __init__(self, mean: float, min_std: float, max_std: float, prob: float, steps: float):
        assert steps > 0, f"Illegal steps {steps}. The argument should be higher than 0!"
        super().__init__(mean, min_std, max_std, prob)
        self._steps = steps
        self._momentum = (self._max_std - self._min_std) / self._steps

    def _update(self):
        self._std = max(self._std - self._momentum, self._min_std)


class EpsilonDecayGaussianMutation(DecayGaussianMutation):
    """
    TODO: Add Documentation + Unittests
    """

    def __init__(self, mean: float, min_std: float, max_std: float, prob: float, steps: float):
        assert steps > 0, f"Illegal steps {steps}. The argument should be higher than 0!"
        super().__init__(mean, min_std, max_std, prob)
        self._steps = steps
        self.decay_factor = (self._min_std / self._max_std) ** (1 / self._steps)

    def _update(self):
        self._std = max(self._std * self.decay_factor, self._min_std)
