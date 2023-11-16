import copy

import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.hp.continuous import Float, Integer
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace

from PyEvo.crossover.abstract_crossover import Crossover


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
