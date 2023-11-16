import copy

import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace

from PyEvo.crossover.abstract_crossover import Crossover


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
