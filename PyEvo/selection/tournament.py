import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace

from PyEvo.selection.abstract_selection import Selection


class TournamentSelection(Selection):
    """
    Class that represents a tournament selection, where the best individuals from each tournament survives the next
    generation.
    """

    def _select(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            n_select: int,
            **kwargs
    ) -> tuple[list[HyperparameterConfiguration], list[float], list[HyperparameterConfiguration], list[float]]:
        reverse = False if optimizer == "min" else True
        selected = []
        fitness_selected = []

        # Choose the individuals per tournament selection
        for _ in range(n_select):
            # Select the tournament size
            tournament_size = random.randint(0, len(pop) - len(selected)) + 1

            # Select individuals in this tournament (that are not selected before)
            indices = random.choice(len(pop) - len(selected), size=tournament_size, replace=False)

            # Select the best individual in that tournament
            index = sorted(indices, key=lambda key: fitness[key], reverse=reverse)[0]
            selected += [pop[index]]
            fitness_selected += [fitness[index]]

            # Remove individual from population and fitness
            pop.pop(index)
            fitness.pop(index)

        # Safe the non-selected individuals with their fitness
        non_selected = pop
        fitness_non_selected = fitness
        return selected, fitness_selected, non_selected, fitness_non_selected
