import copy
from abc import ABC, abstractmethod

import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace


class Selection(ABC):
    """ Abstract class to model the selection phase of an evolutionary algorithm. """

    def select(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            n_select: int,
            **kwargs,
    ) -> tuple[list[HyperparameterConfiguration], list[float], list[HyperparameterConfiguration], list[float]]:
        """
        Choose from the population new individuals to be selected for the next generation.

        Args:
            random (np.random.RandomState):
                The random generator for the sampling procedure

            cs (HyperparameterConfigurationSpace):
                Configuration space from where we manage our hyperparameters (individuals)

            pop (list[HyperparameterConfiguration]):
                The population which should be selected from

            fitness (list[float]):
                The fitness values for each individual in the population

            optimizer (str):
                Type of the optimization problem
                    - optimizer="min": problem should be minimized
                    - optimizer="max": problem should be maximized

            n_select (int):
                Number of individuals to select from the population

            **kwargs (dict):
                Additional parameters for the function

        Returns:
            tuple[list[HyperparameterConfiguration], list[float], list[HyperparameterConfiguration], list[float]]:
                The following information's are returned:
                [0] (list[HyperparameterConfiguration]):
                    The individuals that are selected for the next generation

                [1] (list[float]):
                    The fitness values of the individuals that are selected for the next generation

                [2] (list[HyperparameterConfiguration]):
                    The individuals that are not-selected for the next generation

                [3] (list[float]):
                    The fitness values of the individuals that are not-selected for the next generation
        """
        # Check if each individual in population has all hyperparameters from the configuration space
        assert all(key in p for p in pop for key in cs), \
            f"Invalid Hyperparameter Configuration. Some Hyperparameters not found!"

        assert 2 <= len(pop), \
            f"Illegal population {pop}. The length of the population should be 2 <= len(pop)!"

        assert len(pop) == len(fitness), \
            "Illegal population and fitness. Each individual should be assigned to a fitness value!"

        assert optimizer == "min" or optimizer == "max", \
            f"Illegal optimizer {optimizer}. It should be 'min' or 'max'!"

        assert 1 <= n_select <= len(pop), \
            f"Illegal n_select {n_select}. It should be in between 1 <= n_select <= len(pop)!"

        return self._select(random, cs, copy.deepcopy(pop), copy.deepcopy(fitness), optimizer, n_select, **kwargs)

    @abstractmethod
    def _select(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            n_select: int,
            **kwargs,
    ) -> tuple[list[HyperparameterConfiguration], list[float], list[HyperparameterConfiguration], list[float]]:
        pass
