import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace

from PyEvo.fitness.abstract_preprocessor import FitnessPreprocessor


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
