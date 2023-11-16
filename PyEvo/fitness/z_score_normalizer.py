import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace

from PyEvo.fitness.abstract_preprocessor import FitnessPreprocessor


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
