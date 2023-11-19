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
                Mean of the normal distribution N(mean, std)

            std (float):
                Standard deviation of the normal distribution N(mean, std)

            prob (float):
                Mutation Probability that the mutation occurs
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
    A mutation class that applies Gaussian mutation with a decaying standard deviation.

    This class inherits from GaussianMutation and is designed for mutation operations
    where the standard deviation of the Gaussian distribution decreases over time.

        Args:
            mean (float):
                The mean value of the normal distribution N(mean, std)

            min_std (float):
                The minimum standard deviation after decay is finished

            max_std (float):
                The maximum standard deviation before decay is starting

            prob (float):
                Mutation Probability that the mutation occurs
    """

    def __init__(self, mean: float, min_std: float, max_std: float, prob: float):
        assert max_std > 0.0, f"Illegal max_std {max_std}. The argument should be higher than 0.0!"
        assert min_std > 0.0, f"Illegal min_std {min_std}. The argument should be higher than 0.0!"
        assert min_std < max_std, f"Illegal min_std {min_std} or max_std {max_std}. " \
                                  f"The argument should satisfy the constraint min_std < max_std"

        super().__init__(mean, max_std, prob)
        self._min_std = min_std
        self._max_std = max_std

    @abstractmethod
    def _update(self):
        """
        Updates the current standard deviation (_std) after the decay method.
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
    A mutation class that applies Gaussian mutation with a linearly decaying standard deviation.

    This class inherits from DecayGaussianMutation and is specifically tailored for mutation
    operations where the standard deviation of the Gaussian distribution decreases linearly over time.

    Linear Decaying of the standard deviation is defined as follows:

        momentum := (std_max - std_min) / (#steps)

        std_t+1 := std_t - momentum

        Args:
            mean (float):
                The mean value of the normal distribution N(mean, std)

            min_std (float):
                The minimum standard deviation after decay is finished

            max_std (float):
                The maximum standard deviation before decay is starting

            prob (float):
                Mutation Probability that the mutation occurs

            steps (float):
                Number of times to call the mutation before reaching the minimum standard deviation
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
    A mutation class that applies Gaussian mutation with an epsilon-decaying standard deviation.

    This class inherits from DecayGaussianMutation and is designed for mutation operations
    where the standard deviation of the Gaussian distribution decreases with a minimum value (epsilon) over time.

     Epsilon Decaying of the standard deviation is defined as follows:

        decay_factor := sqrt_#steps(std_min / std_max)

        std_t+1 := std_t * decay_factor

        Args:
            mean (float):
                The mean value of the normal distribution N(mean, std)

            min_std (float):
                The minimum standard deviation after decay is finished

            max_std (float):
                The maximum standard deviation before decay is starting

            prob (float):
                Mutation Probability that the mutation occurs

            steps (float):
                Number of times to call the mutation before reaching the minimum standard deviation

    """

    def __init__(self, mean: float, min_std: float, max_std: float, prob: float, steps: float):
        assert steps > 0, f"Illegal steps {steps}. The argument should be higher than 0!"
        super().__init__(mean, min_std, max_std, prob)
        self._steps = steps
        self._decay_factor = (self._min_std / self._max_std) ** (1 / self._steps)

    def _update(self):
        self._std = max(self._std * self._decay_factor, self._min_std)
