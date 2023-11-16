import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace

from PyEvo.selection.abstract_selection import Selection


class FitnessProportionalSelection(Selection):
    """
    Class that represents a fitness proportional selection, also known as roulette wheel selection, where each
    individual are sampled based on the probabilities that are proportional to their fitness and the optimizer.
    If the given optimizer minimize the problem, then lower fitness values gets a higher probability to be sampled.
    For maximization higher fitness values gets a higher probability.

    More information to fitness proportional selection can be found here: https://en.wikipedia.org/wiki/Fitness_proportionate_selection

        Args:
            temperature (float):
                The temperature parameter controlling the sensitivity to fitness differences

            replace (bool):
                Argument to control if individuals can be sampled more than one time
    """

    def __init__(self, temperature: float = 1.0, replace: bool = False):
        assert temperature > 0.0, f"Illegal temperature {temperature}. This argument should be higher than 0.0!"
        self._temperature = temperature
        self._replace = replace

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

        if optimizer == "min":
            # Case: Minimization problem, so lower values should get higher probabilities
            # Negative all fitness values
            corrected_fitness = [-f for f in fitness]
        else:
            # Case: Maximization problem
            # No need to change the fitness values
            corrected_fitness = [f for f in fitness]

        # Normalize the fitness values to a probability distribution via softmax
        exp_values = [np.exp(f / self._temperature) for f in corrected_fitness]
        sum_exp_values = sum(exp_values)
        if sum_exp_values == 0:
            # Case: Prevent from dividing with zero
            sum_exp_values = 1e-10
        prob = [f / sum_exp_values for f in exp_values]

        indices = random.choice(range(len(pop)), size=n_select, replace=self._replace, p=prob)

        # Extract the selected, non-selected individuals and fitness values
        selected = [pop[idx] for idx in indices]
        fitness_selected = [fitness[idx] for idx in indices]
        non_selected = [pop[idx] for idx in range(len(pop)) if idx not in indices]
        fitness_non_selected = [fitness[idx] for idx in range(len(pop)) if idx not in indices]

        return selected, fitness_selected, non_selected, fitness_non_selected
