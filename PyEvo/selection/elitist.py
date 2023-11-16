import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace

from PyEvo.selection.abstract_selection import Selection


class ElitistSelection(Selection):
    """
    Class that represents an elitist selection, where only the best len(pop)/N individuals survives the next generation.
    """

    def _select(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            n_select: int,
            **kwargs,
    ) -> tuple[list[HyperparameterConfiguration], list[float], list[HyperparameterConfiguration], list[float]]:
        reverse = False if optimizer == "min" else True

        # Return the indices of the sorted lists
        indices = sorted(range(len(fitness)), key=lambda idx: fitness[idx], reverse=reverse)
        fitness_sorted = [fitness[idx] for idx in indices]
        pop_sorted = [pop[idx] for idx in indices]

        # Divide to selected and non-selected
        selected = pop_sorted[:n_select]
        fitness_selected = fitness_sorted[:n_select]
        non_selected = pop_sorted[n_select:]
        fitness_non_selected = fitness_sorted[n_select:]

        return selected, fitness_selected, non_selected, fitness_non_selected
