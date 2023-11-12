import copy
import numpy as np
from abc import ABC, abstractmethod

from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.hp.continuous import Float, Integer


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


class NoCrossover(Crossover):
    """
    Class representing a crossover operation, that creates the childs by simply copying the parents (no real crossover).
    """

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
        childs = []

        N = len(pop)
        i = 0
        while len(childs) < n_childs:
            # Copy the next individual
            childs += [copy.deepcopy(pop[i])]

            # Update the index
            i = (i + 1) % N
        return childs


class UniformCrossover(Crossover):
    """
    Class representing a crossover operation, that creates the child by choosing the gene i from the 1st parent with
    probability p=0.5 and with probability p=1-0.5=0.5 for the 2nd parent.
    """

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
        assert len(pop) >= 2, f"Invalid pop {pop}. It should be len(pop) >= 2!"

        childs = []
        N = len(pop)
        i = 0
        while len(childs) < n_childs:
            parent1, parent2 = pop[i], pop[i + 1]
            values = {}
            for (key1, value1), (key2, value2) in zip(parent1.items(), parent2.items()):
                if random.random() <= 0.5:
                    # Case: Select hyperparameter from parent1
                    values[key1] = value1
                else:
                    # Case: Select hyperparameter from parent2
                    values[key2] = value2
            childs += [HyperparameterConfiguration(values=values)]

            # Update the index
            i = (i + 1) % (N - 1)
            if i == 0:
                # Case: Shuffle the population randomly to get different combination of parents
                random.shuffle(pop)
        return childs


class IntermediateCrossover(Crossover):
    """
    Class representing a crossover operation, that creates the child by taking the mean value of two parents for each
    float/integer hyperparameter.
    """

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
        childs = []

        N = len(pop)
        i = 0
        while len(childs) < n_childs:
            parent1, parent2 = pop[i], pop[i + 1]
            child = copy.deepcopy(parent1)
            for (key1, value1), (key2, value2) in zip(parent1.items(), parent2.items()):
                if isinstance(cs[key1], Float):
                    # Case: Hyperparameter is a Float Hyperparameter
                    child[key1] = (value1 + value2) / 2
                elif isinstance(cs[key1], Integer):
                    # Case: Hyperparameter is an Integer Hyperparameter
                    child[key1] = np.floor((value1 + value2) / 2).astype(dtype=int)
            childs += [child]

            # Update the index
            i = (i + 1) % (N - 1)
            if i == 0:
                # Case: Shuffle the population randomly to get different combination of parents
                random.shuffle(pop)
        return childs
