import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace

from PyEvo.fitness.abstract_preprocessor import FitnessPreprocessor


class FitnessSoftmaxNormalizer(FitnessPreprocessor):
    """
    Class responsible for normalizing fitness values of a population according to the softmax normalization.

    Each fitness value gets normalized by dividing the individual softmax fitness value from the sum of all softmax
    fitness values:
        - fitness_normalized := softmax(fitness / temperature) / sum(softmax(fitness / temperature)

        Args:
            temperature (float):
                The temperature parameter controlling the sensitivity to fitness differences
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
