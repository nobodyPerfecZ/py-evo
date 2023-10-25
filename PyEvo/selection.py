import copy
import numpy as np
from abc import ABC, abstractmethod

from PyHyperparameterSpace.configuration import HyperparameterConfiguration


class Selection(ABC):
    """ Abstract class to model the selection phase of an evolutionary algorithm. """

    def select(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfiguration,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            n_select: int,
            **kwargs,
    ) -> list[HyperparameterConfiguration]:
        """
        Choose from the population new individuals to be selected for the next generation.

        Args:
            random (np.random.RandomState):
                The random generator for the sampling procedure

            cs (HyperparameterConfigurationSpace):
                configuration space from where we manage our hyperparameters (individuals)

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
                additional parameters for the function

        Returns:
            list[HyperparameterConfiguration]:
                The individuals that are selected for the next generation
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
            cs: HyperparameterConfiguration,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            n_select: int,
            **kwargs,
    ):
        pass


class ElitistSelection(Selection):
    """
    Class that represents an elitist selection, where only the best len(pop)/N individuals survives the next generation.
    """
    def _select(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfiguration,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            n_select: int,
            **kwargs,
    ) -> list[HyperparameterConfiguration]:
        reverse = False if optimizer == "min" else True
        new_pop = sorted(pop, key=lambda key: fitness[pop.index(key)], reverse=reverse)[:n_select]
        return new_pop


class TournamentSelection(Selection):
    """
    Class that represents a tournament selection, where the best individuals from each tournament survives the next
    generation.
    """

    def _select(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfiguration,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            n_select: int,
            **kwargs
    ) -> list[HyperparameterConfiguration]:
        reverse = False if optimizer == "min" else True
        new_pop = []

        # Choose the individuals per tournament selection
        for _ in range(n_select):
            # Select the tournament size
            tournament_size = random.randint(0, len(pop) - len(new_pop)) + 1

            # Select individuals in this tournament (that are not selected before)

            indices = random.choice(len(pop) - len(new_pop), size=tournament_size, replace=False)
            tournament = [pop[idx] for idx in indices]
            tournament_fitness = [fitness[idx] for idx in indices]

            # Select the best individual in that tournament
            ind = sorted(tournament, key=lambda key: tournament_fitness[tournament.index(key)], reverse=reverse)[0]
            new_pop += [ind]

            # Remove individual from population and fitness
            index = pop.index(ind)
            pop.remove(ind)
            fitness.remove(fitness[index])
        return new_pop

# TODO: Implement fitness proportionate selection (https://en.wikipedia.org/wiki/Fitness_proportionate_selection)
