import copy
import numpy as np
from abc import ABC, abstractmethod

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


class NoFitnessPreprocessing(FitnessPreprocessor):
    """
    Class responsible for no fitness preprocessing by just returning the given fitness values.
    """

    def _preprocess_fitness(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs
    ) -> list[float]:
        return fitness


class FitnessNormalizer(FitnessPreprocessor):
    """
    Class responsible for normalizing fitness values of a population from [-min_value, +max_value] to [0, 1].
    """

    def _preprocess_fitness(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs,
    ) -> list[float]:
        # Get the max and min value of the current population
        max_fitness = max(fitness)
        min_fitness = min(fitness)

        if (max_fitness - min_fitness) == 0:
            # Case: Prevent dividing with 0
            max_fitness += 1e-10

        # Return normalized fitness values (from [-min_value, +max_value] -> [0, 1])
        normalized_values = [(f - min_fitness) / (max_fitness - min_fitness) for f in fitness]

        return normalized_values


class FitnessZScoreNormalizer(FitnessPreprocessor):
    """
    Class responsible for normalizing fitness values of a population according to the z-score normalization.

    Each fitness value gets normalized by subtracting the mean and dividing the std of the fitness values:
        - fitness_normalized := (fitness - mean(fitness)) / std(fitness)
    """

    def _preprocess_fitness(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs
    ) -> list[float]:
        # Get the mean and std of the fitness values
        mean = np.mean(fitness)
        std = np.std(fitness)

        if std == 0:
            # Case: Prevent dividing with 0
            std = 1e-10

        # Return normalized fitness values according to z-score normalization
        normalized_values = [(f - mean) / std for f in fitness]

        return normalized_values


class FitnessSoftmaxNormalizer(FitnessPreprocessor):
    """
    Class responsible for normalizing fitness values of a population according to the softmax normalization.

    Each fitness value gets normalized by dividing the individual softmax fitness value from the sum of all softmax
    fitness values:
        - fitness_normalized := softmax(fitness / temperature) / sum(softmax(fitness / temperature)
    """

    def __init__(self, temperature: float = 1.0):
        assert temperature > 0.0, f"Illegal temperature {temperature}. The argument should be higher than 0.0!"
        self._temperature = temperature

    def _preprocess_fitness(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs
    ) -> list[float]:
        # Get the softmax values from each individual and its sum
        fitness_softmax = [np.exp(f / self._temperature) for f in fitness]
        sum_fitness_softmax = sum(fitness_softmax)

        # Return normalized softmax fitness values
        normalized_values = [f / sum_fitness_softmax for f in fitness_softmax]
        return normalized_values


class FitnessRanker(FitnessPreprocessor):
    """
    Class responsible for rank-based fitness assignment.

    Each transformed fitness value responds to a rank r_i based on the performance (fitness) f_i of the individual in
    asc order.

    The lowest fitness value f_i gets the lowest rank value r_i := start, where the highest fitness value f_i gets the
    highest rank value r_j := start + j

        Args:
            start (int):
                The rank value of the first place
    """

    def __init__(self, start: int = 1):
        assert start >= 0, f"Illegal start {start}. The argument should be higher or equal to 0!"
        self._start = start

    def _preprocess_fitness(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs,
    ) -> list[float]:
        # Return indices of the sorted list
        indices = sorted(range(len(fitness)), key=lambda idx: fitness[idx], reverse=False)

        # Assign each fitness value the ranks
        ranks = [0 for _ in range(len(fitness))]
        for i, idx in enumerate(indices):
            ranks[idx] = self._start + i
        return ranks
