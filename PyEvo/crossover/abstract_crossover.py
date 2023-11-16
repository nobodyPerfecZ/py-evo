import copy
from abc import ABC, abstractmethod

import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace


class Crossover(ABC):
    """ Abstract class to model the crossover phase of an evolutionary algorithm. """

    def crossover(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            n_childs: int,
            **kwargs,
    ) -> list[HyperparameterConfiguration]:
        """
        Creates new child individuals by selecting repeatedly two parents from the population and crossover them on
        specific hyperparameters.

        Args:
            random (np.random.RandomState):
                The random generator for the sampling procedure

            cs (HyperparameterConfigurationSpace):
                configuration space from where we manage our hyperparameters (individuals)

            pop (list[HyperparameterConfiguration]):
                The population from where we select the parents

            fitness (list[float]):
                The fitness values for each individual in the population

            optimizer (str):
                Type of the optimization problem
                    - optimizer="min": problem should be minimized
                    - optimizer="max": problem should be maximized

            n_childs (int):
                Number of childs to be created from the population

            **kwargs (dict):
                additional parameters for the function

        Returns:
            list[HyperparameterConfiguration]:
                The childs that are crossover from two individuals from the population
        """
        # Check if each individual in population has all hyperparameters from the configuration space
        assert all(key in p for p in pop for key in cs), \
            f"Invalid Hyperparameter Configuration. Some Hyperparameters not found!"

        assert 2 <= len(pop), \
            f"Illegal population {pop}. The length of the population should be 2 <= len(pop)!"

        assert optimizer == "min" or optimizer == "max", \
            f"Illegal optimizer {optimizer}. It should be 'min' or 'max'!"

        assert 1 <= n_childs, f"Illegal n_childs {n_childs}. It should be 1 <= n_childs!"

        return self._crossover(random, cs, copy.deepcopy(pop), copy.deepcopy(fitness), optimizer, n_childs, **kwargs)

    @abstractmethod
    def _crossover(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            n_childs: int,
            **kwargs,
    ) -> list[HyperparameterConfiguration]:
        pass