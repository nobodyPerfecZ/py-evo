from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.hp.continuous import Float, Integer
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace

from PyEvo.mutation.abstract_mutation import Mutation


class UniformMutation(Mutation):
    """
    Class representing a mutation operation that introduces noise to float and integer hyperparameters.

    This mutation perturbs hyperparameters using random values drawn from a uniform distribution.

        Args:
            low (Union[int, float]):
                The lower bound (:= a) of the uniform distribution U(a, b)

            high (Union[int, float]):
                The upper bound (:= b) of the uniform distribution U(a, b)

            prob (float):
                Mutation Probability that the mutation occurs
    """

    def __init__(self, low: Union[int, float], high: Union[int, float], prob: float):
        assert low < 0, f"Illegal low {low}. The argument should be less than 0 !"
        assert high > 0, f"Illegal high {high}. The argument should be higher than 0!"
        assert low < high, f"Illegal low {low} or high {high}. The arguments should satisfy the constraint low < high!"
        super().__init__(prob)

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
        for ind in pop:
            for key, value in ind.items():
                hp = cs[key]
                if isinstance(hp, Float) and random.random() <= self._prob:
                    # Case: Hyperparameter is continuous
                    value = hp.adjust_configuration(
                        value + random.uniform(low=self._low, high=self._high)
                    )
                    ind[key] = value
                elif isinstance(hp, Integer) and random.random() <= self._prob:
                    # Case: Hyperparameter is discrete
                    value = hp.adjust_configuration(
                        value + random.random_integers(low=self._low, high=self._high)
                    )
                    ind[key] = value
        return pop


class DecayUniformMutation(UniformMutation, ABC):
    """
    A mutation class that introduces noise to float and integer hyperparameters
    with decayed perturbation values over time.


        Args:
            min_low (Union[int, float]):
                The minimum lower bound (:= a_min) before decaying is starting

            max_low (Union[int, float]):
                The maximum lower bound (:= a_max) after decaying is finished

            min_high (Union[int, float]):
                The minimum upper bound (:= b_min) after decaying is finished

            max_high (Union[int, float]):
                The minimum upper bound (:= b_max) before decaying is starting

            prob (float):
                Mutation Probability that the mutation occurs
    """
    def __init__(
            self,
            min_low: Union[int, float],
            max_low: Union[int, float],
            min_high: Union[int, float],
            max_high: Union[int, float],
            prob: float
    ):
        assert min_low < 0, f"Illegal min_low {min_low}. The argument should be less than 0!"
        assert max_low < 0, f"Illegal max_low {max_low}. The argument should be less than 0!"
        assert min_high > 0, f"Illegal min_high {min_high}. The argument should be higher than 0!"
        assert max_high > 0, f"Illegal max_high {max_high}. The argument should be higher than 0!"
        assert min_low < max_low < min_high < max_high, f"Illegal min_low {min_low}, max_low {max_low}, min_high {min_high} or max_high {max_high}. The arguments should satisfy the constraint min_low < max_low < min_high < max_high!"
        super().__init__(min_low, max_high, prob)

        self._min_low = min_low
        self._max_low = max_low
        self._min_high = min_high
        self._max_high = max_high

    @abstractmethod
    def _update(self):
        """
        Updates the current lower (_low) and upper (_high) bound after the decay method.
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


class LinearDecayUniformMutation(DecayUniformMutation):
    """
    A mutation class that introduces noise to float and integer hyperparameters
    with decayed perturbation values over time.

    Linear Decaying of the lower and upper bound is defined as follows:

        momentum_low := (low_min - low_max) / (#steps)
        momentum_high := (high_max - high_min) / (#steps)

        low_t+1 := low_t - momentum_low
        high_t+1 := high_t - momentum_high

        Args:
            min_low (Union[int, float]):
                The minimum lower bound (:= a_min) before decaying is starting

            max_low (Union[int, float]):
                The maximum lower bound (:= a_max) after decaying is finished

            min_high (Union[int, float]):
                The minimum upper bound (:= b_min) after decaying is finished

            max_high (Union[int, float]):
                The minimum upper bound (:= b_max) before decaying is starting

            prob (float):
                Mutation Probability that the mutation occurs
            
            steps (float):
                Number of times to call the mutation before reaching the minimum standard deviation
    """

    def __init__(
            self,
            min_low: Union[int, float],
            max_low: Union[int, float],
            min_high: Union[int, float],
            max_high: Union[int, float],
            prob: float,
            steps: int,
    ):
        assert steps > 0, f"Illegal steps {steps}. The argument should be higher than 0!"
        super().__init__(min_low, max_low, min_high, max_high, prob)

        self._steps = steps

        self._low_momentum = (self._min_low - self._max_low) / self._steps
        self._high_momentum = (self._max_high - self._min_high) / self._steps

    def _update(self):
        self._low = min(self._low - self._low_momentum, self._max_low)
        self._high = max(self._high - self._high_momentum, self._min_high)


class EpsilonDecayUniformMutation(DecayUniformMutation):
    """
    A mutation class that introduces noise to float and integer hyperparameters
    with decayed perturbation values over time.

    Epsilon Decaying of the lower and upper bound is defined as follows:

        decay_low := sqrt_#steps(low_max / low_min)
        decay_high := sqrt_#steps(high_min / high_max)

        low_t+1 := low_t * decay_low
        high_t+1 := high_t * decay_high

        Args:
            min_low (Union[int, float]):
                The minimum lower bound (:= a_min) before decaying is starting

            max_low (Union[int, float]):
                The maximum lower bound (:= a_max) after decaying is finished

            min_high (Union[int, float]):
                The minimum upper bound (:= b_min) after decaying is finished

            max_high (Union[int, float]):
                The minimum upper bound (:= b_max) before decaying is starting

            prob (float):
                Mutation Probability that the mutation occurs

            steps (float):
                Number of times to call the mutation before reaching the minimum standard deviation
    """
    def __init__(
            self,
            min_low: Union[int, float],
            max_low: Union[int, float],
            min_high: Union[int, float],
            max_high: Union[int, float],
            prob: float,
            steps: int,
    ):
        assert steps > 0, f"Illegal steps {steps}. The argument should be higher than 0!"
        super().__init__(min_low, max_low, min_high, max_high, prob)

        self._steps = steps

        self._low_decay = (self._max_low / self._min_low) ** (1 / self._steps)
        self._high_decay = (self._min_high / self._max_high) ** (1 / self._steps)

    def _update(self):
        self._low = min(self._low * self._low_decay, self._max_low)
        self._high = max(self._high * self._high_decay, self._min_high)
