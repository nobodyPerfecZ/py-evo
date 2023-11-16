import copy
from abc import ABC, abstractmethod

import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace


class Mutation(ABC):
    """
    Abstract class to model the mutation phase of an evolutionary algorithm.

        Args:
            prob (float):
                Probability that the mutation occurs
    """

    def __init__(self, prob: float):
        assert 0.0 <= prob <= 1.0, f"Invalid prob {prob}. It should be in between 0 <= prob <= 1!"
        self._prob = prob

    def mutate(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs,
    ) -> list[HyperparameterConfiguration]:
        """
        Creates a new population by mutating each individual from the population.

        Args:
            random (np.random.RandomState):
                The random generator for the sampling procedure

            cs (HyperparameterConfigurationSpace):
                configuration space from where we manage our hyperparameters (individuals)

            pop (list[HyperparameterConfiguration]):
                The population which should be selected from

            fitness (list[float]):
                The fitness values for each individual in the population

            optimizer (str):
                Type of the optimization problem
                    - optimizer="min": problem should be minimized
                    - optimizer="max": problem should be maximized

            **kwargs (dict):
                additional parameters for the function

        Returns:
            list[HyperparameterConfiguration]:
                The mutated population
        """
        # Check if each individual in population has all hyperparameters from the configuration space
        assert all(key in p for p in pop for key in cs), \
            f"Invalid Hyperparameter Configuration. Some Hyperparameters not found!"

        assert 2 <= len(pop), \
            f"Illegal population {pop}. The length of the population should be 2 <= len(pop)!"

        assert optimizer == "min" or optimizer == "max", \
            f"Illegal optimizer {optimizer}. It should be 'min' or 'max'!"

        assert len(pop) == len(fitness), \
            "Illegal population and fitness. Each individual should be assigned to a fitness value!"

        return self._mutate(random, cs, copy.deepcopy(pop), copy.deepcopy(fitness), optimizer, **kwargs)

    @abstractmethod
    def _mutate(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs,
    ) -> list[HyperparameterConfiguration]:
        pass