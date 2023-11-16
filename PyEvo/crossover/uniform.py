import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace

from PyEvo.crossover.abstract_crossover import Crossover


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
            childs += [HyperparameterConfiguration(cs=cs, values=values)]

            # Update the index
            i = (i + 1) % (N - 1)
            if i == 0:
                # Case: Shuffle the population randomly to get different combination of parents
                random.shuffle(pop)
        return childs
