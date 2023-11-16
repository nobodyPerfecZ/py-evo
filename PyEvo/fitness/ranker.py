import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace

from PyEvo.fitness.abstract_preprocessor import FitnessPreprocessor


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
