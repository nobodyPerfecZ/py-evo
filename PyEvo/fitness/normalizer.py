import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace

from PyEvo.fitness.abstract_preprocessor import FitnessPreprocessor


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
