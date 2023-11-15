import copy
from abc import ABC, abstractmethod

import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace


class Selection(ABC):
    """ Abstract class to model the selection phase of an evolutionary algorithm. """

    def select(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            n_select: int,
            **kwargs,
    ) -> tuple[list[HyperparameterConfiguration], list[float], list[HyperparameterConfiguration], list[float]]:
        """
        Choose from the population new individuals to be selected for the next generation.

        Args:
            random (np.random.RandomState):
                The random generator for the sampling procedure

            cs (HyperparameterConfigurationSpace):
                Configuration space from where we manage our hyperparameters (individuals)

            pop (list[HyperparameterConfiguration]):
                The population which should be selected from

            fitness (list[float]):
                The fitness values for each individual in the population

            optimizer (str):
                Type of the optimization problem
                    - optimizer="min": problem should be minimized
                    - optimizer="max": problem should be maximized

            n_select (int):
                Number of individuals to select from the population

            **kwargs (dict):
                Additional parameters for the function

        Returns:
            tuple[list[HyperparameterConfiguration], list[float], list[HyperparameterConfiguration], list[float]]:
                The following information's are returned:
                [0] (list[HyperparameterConfiguration]):
                    The individuals that are selected for the next generation

                [1] (list[float]):
                    The fitness values of the individuals that are selected for the next generation

                [2] (list[HyperparameterConfiguration]):
                    The individuals that are not-selected for the next generation

                [3] (list[float]):
                    The fitness values of the individuals that are not-selected for the next generation
        """
        # Check if each individual in population has all hyperparameters from the configuration space
        assert all(key in p for p in pop for key in cs), \
            f"Invalid Hyperparameter Configuration. Some Hyperparameters not found!"

        assert 2 <= len(pop), \
            f"Illegal population {pop}. The length of the population should be 2 <= len(pop)!"

        assert len(pop) == len(fitness), \
            "Illegal population and fitness. Each individual should be assigned to a fitness value!"

        assert optimizer == "min" or optimizer == "max", \
            f"Illegal optimizer {optimizer}. It should be 'min' or 'max'!"

        assert 1 <= n_select <= len(pop), \
            f"Illegal n_select {n_select}. It should be in between 1 <= n_select <= len(pop)!"

        return self._select(random, cs, copy.deepcopy(pop), copy.deepcopy(fitness), optimizer, n_select, **kwargs)

    @abstractmethod
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
        pass


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
