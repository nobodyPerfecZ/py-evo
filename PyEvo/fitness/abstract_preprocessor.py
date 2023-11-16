import copy
from abc import ABC, abstractmethod

import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace


class FitnessPreprocessor(ABC):
    """
    Class responsible for preprocessing fitness values of a population.

    This class provides methods to transform raw fitness values into a different format.
    """

    def preprocess_fitness(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs,
    ) -> list[float]:
        """
        Preprocesses the raw fitness values and returns the transformed fitness values.

        Args:
            random (np.random.RandomState):
                The random generator for possible sampling procedure

            cs (HyperparameterConfigurationSpace):
                Configuration space from where we manage our hyperparameters (individuals)

            pop (list[HyperparameterConfiguration]):
                The population which is assigned to the given fitness values

            fitness (list[float]):
                The fitness values for each individual in the population

            optimizer (str):
                Type of the optimization problem
                    - optimizer="min": problem should be minimized
                    - optimizer="max": problem should be maximized

            **kwargs (dict):
                Additional parameters for the function

        Returns:
            List[float]:
                Transformed fitness values.
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

        return self._preprocess_fitness(random, cs, copy.deepcopy(pop), copy.deepcopy(fitness), optimizer, **kwargs)

    @abstractmethod
    def _preprocess_fitness(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs,
    ) -> list[float]:
        pass
