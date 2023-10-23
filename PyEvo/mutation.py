from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import copy

from PyHyperparameterSpace.hp.continuous import Float, Integer
from PyHyperparameterSpace.hp.categorical import Categorical, Binary
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.configuration import HyperparameterConfiguration


class Mutation(ABC):
    """ Abstract class to model the mutation phase of an evolutionary algorithm. """

    def __init__(self, prob: float):
        assert 0.0 <= prob <= 1.0, f"Invalid prob {prob}. It should be in between 0 <= prob <= 1!"
        self._prob = prob

    def mutate(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
    ) -> list[HyperparameterConfiguration]:
        """
        Creates a new population by mutating each individual from the population.

        Args:
            random (np.random.RandomState):
                The random generator for the sampling procedure

            cs (HyperparameterConfigurationSpace):
                configuration space from where we manage our hyperparameters (individuals)

            pop (list[HyperparameterConfiguration]):
                The population from where we select our individuals

        Returns:
            list[HyperparameterConfiguration]:
                The mutated population
        """
        # Check if each individual in population has all hyperparameters from the configuration space
        for key in cs:
            for p in pop:
                assert key in p, f"Invalid Hyperparameter Configuration. Hyperparameter {key} not found!"

        return self._mutate(random, cs, copy.deepcopy(pop))

    @abstractmethod
    def _mutate(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
    ) -> list[HyperparameterConfiguration]:
        pass


class GaussianMutation(Mutation):
    """
    Class representing a mutation operation that introduces noise to float and integer hyperparameters.
    The mutation perturbs hyperparameters using random values drawn from a normal distribution.
    """

    def __init__(self, loc: float, scale: float, prob: float):
        super().__init__(prob)
        self._loc = loc
        self._scale = scale

    def _mutate(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
    ) -> list[HyperparameterConfiguration]:
        for ind in pop:
            for key, hp in ind.items():
                if isinstance(cs[key], Float) and random.random() <= self._prob:
                    if isinstance(cs, float):
                        ind[key] += random.normal(loc=self._loc, scale=self._scale)

                        # Check for bounds
                        if ind[key] < cs[key].lb:
                            ind[key] = cs[key].lb
                        elif ind[key] >= cs[key].ub:
                            ind[key] = cs[key].ub - 1e-10
                    elif isinstance(hp, np.ndarray):
                        ind[key] += random.normal(loc=self._loc, scale=self._scale, size=hp.shape)

                        # Check for bounds
                        ind[key][ind[key] < cs[key].lb] = cs[key].lb
                        ind[key][ind[key] >= cs[key].ub] = cs[key].ub - 1e-10
        return pop


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
    ) -> list[HyperparameterConfiguration]:
        if self._hp_type == "float":
            hp_type = Float
            data_type = float
            sampling = random.uniform
            epsilon = 1e-10
        else:
            hp_type = Integer
            data_type = int
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


class ChoiceMutation(Mutation):
    """
    Class representing a mutation operation that randomly choice between the values of a categorical hyperparameter,
    drawn from a uniform distribution.
    """

    def _mutate(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
    ) -> list[HyperparameterConfiguration]:
        for ind in pop:
            for key, hp in ind.items():
                if isinstance(cs[key], (Categorical, Binary)) and random.random() <= self._prob:
                    index = random.choice(len(cs[key].get_choices()))
                    ind[key] = np.array([cs[key].get_choices()[index]])
        return pop
