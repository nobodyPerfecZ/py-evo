from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import copy

from PyHyperparameterSpace.hp.continuous import Float, Integer
from PyHyperparameterSpace.hp.categorical import Categorical, Binary
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.configuration import HyperparameterConfiguration


class Mutation(ABC):
    """ Abstract class to model the mutation phase of an evolutionary algorithm. """

    def __init__(self, prob: float):
        assert 0.0 <= prob <= 1.0, f"Invalid prob {prob}. It should be in between 0 <= prob <= 1!"
        self._prob = prob

    def mutate(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs,
    ) -> list[HyperparameterConfiguration]:
        """
        Creates a new population by mutating each individual from the population.

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

            **kwargs (dict):
                additional parameters for the function

        Returns:
            list[HyperparameterConfiguration]:
                The mutated population
        """
        # Check if each individual in population has all hyperparameters from the configuration space
        assert all(key in p for p in pop for key in cs), \
            f"Invalid Hyperparameter Configuration. Some Hyperparameters not found!"

        assert 2 <= len(pop), \
            f"Illegal population {pop}. The length of the population should be 2 <= len(pop)!"

        assert optimizer == "min" or optimizer == "max", \
            f"Illegal optimizer {optimizer}. It should be 'min' or 'max'!"

        return self._mutate(random, cs, copy.deepcopy(pop), copy.deepcopy(fitness), optimizer, **kwargs)

    @abstractmethod
    def _mutate(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs,
    ) -> list[HyperparameterConfiguration]:
        pass


class GaussianMutation(Mutation):
    """
    Class representing a mutation operation that introduces noise to float hyperparameters.
    The mutation perturbs hyperparameters using random values drawn from a normal distribution.
    """

    def __init__(self, loc: float, scale: float, prob: float):
        super().__init__(prob)
        self._loc = loc
        self._scale = scale

    def _mutate(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs,
    ) -> list[HyperparameterConfiguration]:
        for ind in pop:
            for key, hp in ind.items():
                if isinstance(cs[key], Float) and random.random() <= self._prob:
                    if isinstance(cs, float):
                        ind[key] += random.normal(loc=self._loc, scale=self._scale)

                        # Check for bounds
                        if ind[key] < cs[key].lb:
                            ind[key] = cs[key].lb
                        elif ind[key] >= cs[key].ub:
                            ind[key] = cs[key].ub - 1e-10
                    elif isinstance(hp, np.ndarray):
                        ind[key] += random.normal(loc=self._loc, scale=self._scale, size=hp.shape)

                        # Check for bounds
                        ind[key][ind[key] < cs[key].lb] = cs[key].lb
                        ind[key][ind[key] >= cs[key].ub] = cs[key].ub - 1e-10
        return pop


class AdaptiveGaussianMutation(GaussianMutation):
    """
    Class representing a mutation operation that introduces noise to float hyperparameters.
    The mutation perturbs hyperparameters using random values drawn from a normal distribution.
    Its self-adaptive method, where the standard deviation (scale) gets changed during the evolutionary algorithm.
    """

    def __init__(self, threshold: float, alpha: float, n_generation: int, initial_loc: float, initial_scale: float,
                 initial_prob: float):
        super().__init__(initial_loc, initial_scale, initial_prob)
        assert 0.0 < threshold < 1.0, \
            f"Illegal threshold {threshold}. It should be in between of 0.0 < threshold < 1.0!"
        assert 0.0 < alpha < 1.0, \
            f"Illegal alpha {alpha}. It should be in between 0.0 < alpha < 1.0!"
        assert 1 <= n_generation, \
            f"Illegal n_generation {n_generation}. It should be 1 <= n_generation!"

        self._threshold = threshold
        self._alpha = alpha
        self._n_generation = n_generation

        # Necessary for the self-adaptation
        self._fitness = []  # stores the mean fitness over the last n generations
        self._improvements = 0
        self._generations = 0

    @property
    def _success_rate(self) -> float:
        """
        Returns:
            float:
                The success rate := #improvements over last n-th generation / #generations
        """
        return self._improvements / self._generations

    def _update_fitness(self, fitness: list[float]):
        """
        Updates the (mean) fitness queue of the last n-th generations.

        If the fitness queue has less than n elements, then it appends the element to the queue.
        If the fitness queue has more than n elements, then it pops the oldest element from the queue and appends the
        newest element to the queue.

        Args:
            fitness (list[float]):
                The fitness values of the current population
        """
        mean_fitness = np.mean(fitness)

        if len(self._fitness) < self._n_generation:
            # Case: Fill fitness before doing self-adaptation
            self._fitness.append(mean_fitness)
        else:
            # Case: Remove the oldest fitness and fill the newest fitness
            self._fitness.pop(0)
            self._fitness.append(mean_fitness)

    def _check_improvement(self, fitness: list[float], optimizer: str) -> bool:
        """
        Returns True if the mean fitness of the current population has a better value than the mean fitness over the
        last n-th generation.

        Args:
            fitness (list[float]):
                The fitness values of the current population

            optimizer (str):
                Type of the optimization problem
                    - optimizer="min": problem should be minimized
                    - optimizer="max": problem should be maximized

        Returns:
            bool:
                True, if the current population has an improvement over the last n-th generation
        """
        mean_fitness = np.mean(fitness)
        last_fitness = np.mean(self._fitness)
        if (optimizer == "min" and mean_fitness < last_fitness) or (optimizer == "max" and mean_fitness > last_fitness):
            return True
        return False

    def _adaptive_adjustment(self, fitness: list[float], optimizer: str):
        """
        Do an adjustment step of the standard deviation after the 1/n-th rule, where we (...)
            - decrease by a factor alpha if the success rate > threshold
            - increase by a factor alpha if the success rate < threshold

        Args:
            fitness (list[float]):
                The fitness values of the current population

            optimizer (str):
                Type of the optimization problem
                    - optimizer="min": problem should be minimized
                    - optimizer="max": problem should be maximized
        """
        if len(self._fitness) < self._n_generation:
            self._update_fitness(fitness)
        else:
            if self._check_improvement(fitness, optimizer):
                self._improvements += 1
            self._generations += 1

            # Self-adaptation of standard deviation
            if self._success_rate > self._threshold:
                # Case: Do less exploration, more exploitation
                self._scale *= self._alpha
            elif self._success_rate < self._threshold:
                # Case: Do more exploration, less exploitation
                self._scale /= self._alpha

            self._update_fitness(fitness)
            print(f"Fitness: {fitness}")
            print(f"N-th mean fitness: {self._fitness}")
            print(f"Current mean fitness: {np.mean(fitness)}")
            print(f"Success Rate={self._success_rate}")
            print(f"Updated std={self._scale}")

    def _mutate(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs,
    ) -> list[HyperparameterConfiguration]:
        # Make an adaptive adjustment
        self._adaptive_adjustment(fitness, optimizer)
        return super()._mutate(random, cs, pop, fitness, optimizer, **kwargs)


class UniformMutation(Mutation):
    """
    Class representing a mutation operation that introduces noise to float and integer hyperparameters.
    The mutation perturbs hyperparameters using random values drawn from a uniform distribution.
    """

    def __init__(self, low: Union[int, float], high: Union[int, float], hp_type: str, prob: float):
        assert hp_type == "int" or hp_type == "float", f"Illegal hp_type {hp_type}. It should be 'int', 'float'!"
        assert low < high, f"Illegal low {low} or high {high}. It should be that low < high!"
        super().__init__(prob)
        self._hp_type = hp_type
        self._low = low
        self._high = high

    def _mutate(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs,
    ) -> list[HyperparameterConfiguration]:
        if self._hp_type == "float":
            hp_type = Float
            data_type = float
            sampling = random.uniform
            epsilon = 1e-10
        else:
            hp_type = Integer
            data_type = int
            sampling = random.random_integers
            epsilon = 1

        for ind in pop:
            for key, hp in ind.items():
                if isinstance(cs[key], hp_type) and random.random() <= self._prob:
                    if isinstance(hp, data_type):
                        ind[key] += sampling(low=self._low, high=self._high)
                        if ind[key] < cs[key].lb:
                            ind[key] = cs[key].lb
                        elif ind[key] >= cs[key].ub:
                            ind[key] = cs[key].ub - epsilon
                    elif isinstance(hp, np.ndarray):
                        ind[key] += sampling(low=self._low, high=self._high, size=hp.shape)
                        ind[key][ind[key] < cs[key].lb] = cs[key].lb
                        ind[key][ind[key] >= cs[key].ub] = cs[key].ub - epsilon
        return pop


class AdaptiveUniformMutation(UniformMutation):
    """
    Class representing a mutation operation that introduces noise to float and integer hyperparameters.
    The mutation perturbs hyperparameters using random values drawn from a uniform distribution.
    Its self-adaptive method, where the lower and upper bounds gets changed during the evolutionary algorithm.
    """

    def __init__(
            self,
            threshold: float,
            alpha: float,
            n_generation: int,
            initial_low: Union[int, float],
            inital_high: Union[int, float],
            initial_prob: float,
    ):
        super().__init__(initial_low, inital_high, "float", initial_prob)

        assert 0.0 < threshold < 1.0, \
            f"Illegal threshold {threshold}. It should be in between of 0.0 < threshold < 1.0!"
        assert 0.0 < alpha < 1.0, \
            f"Illegal alpha {alpha}. It should be in between 0.0 < alpha < 1.0!"
        assert 1 <= n_generation, \
            f"Illegal n_generation {n_generation}. It should be 1 <= n_generation!"

        self._threshold = threshold
        self._alpha = alpha
        self._n_generation = n_generation

        # Necessary for the self-adaptation
        self._fitness = []  # stores the mean fitness over the last n generations
        self._improvements = 0
        self._generations = 0

    @property
    def _success_rate(self) -> float:
        """
        Returns:
            float:
                The success rate := #improvements over last n-th generation / #generations
        """
        return self._improvements / self._generations

    def _update_fitness(self, fitness: list[float]):
        """
        Updates the (mean) fitness queue of the last n-th generations.

        If the fitness queue has less than n elements, then it appends the element to the queue.
        If the fitness queue has more than n elements, then it pops the oldest element from the queue and appends the
        newest element to the queue.

        Args:
            fitness (list[float]):
                The fitness values of the current population
        """
        mean_fitness = np.mean(fitness)

        if len(self._fitness) < self._n_generation:
            # Case: Fill fitness before doing self-adaptation
            self._fitness.append(mean_fitness)
        else:
            # Case: Remove the oldest fitness and fill the newest fitness
            self._fitness.pop(0)
            self._fitness.append(mean_fitness)

    def _check_improvement(self, fitness: list[float], optimizer: str) -> bool:
        """
        Returns True if the mean fitness of the current population has a better value than the mean fitness over the
        last n-th generation.

        Args:
            fitness (list[float]):
                The fitness values of the current population

            optimizer (str):
                Type of the optimization problem
                    - optimizer="min": problem should be minimized
                    - optimizer="max": problem should be maximized

        Returns:
            bool:
                True, if the current population has an improvement over the last n-th generation
        """
        mean_fitness = np.mean(fitness)
        last_fitness = np.mean(self._fitness)
        if (optimizer == "min" and mean_fitness < last_fitness) or (optimizer == "max" and mean_fitness > last_fitness):
            return True
        return False

    def _adaptive_adjustment(self, fitness: list[float], optimizer: str):
        """
        Do an adjustment step of the lower and upper bound after the 1/n-th rule, where we (...)
            - decrease by a factor alpha if the success rate > threshold
            - increase by a factor alpha if the success rate < threshold

        Args:
            fitness (list[float]):
                The fitness values of the current population

            optimizer (str):
                Type of the optimization problem
                    - optimizer="min": problem should be minimized
                    - optimizer="max": problem should be maximized
        """
        if len(self._fitness) < self._n_generation:
            self._update_fitness(fitness)
        else:
            if self._check_improvement(fitness, optimizer):
                self._improvements += 1
            self._generations += 1

            # Self-adaptation of lower and upper bound
            if self._success_rate > self._threshold:
                # Case: Do less exploration, more exploitation
                self._low *= self._alpha
                self._high *= self._alpha
            elif self._success_rate < self._threshold:
                # Case: Do more exploration, less exploitation
                self._low /= self._alpha
                self._high /= self._alpha

            self._update_fitness(fitness)
            print(f"Fitness: {fitness}")
            print(f"N-th mean fitness: {self._fitness}")
            print(f"Current mean fitness: {np.mean(fitness)}")
            print(f"Success Rate={self._success_rate}")
            print(f"Updated lower bound={self._low}")
            print(f"Updated upper bound={self._high}")

    def _mutate(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs,
    ) -> list[HyperparameterConfiguration]:
        # Make an adaptive adjustment
        self._adaptive_adjustment(fitness, optimizer)
        return super()._mutate(random, cs, pop, fitness, optimizer, **kwargs)


class ChoiceMutation(Mutation):
    """
    Class representing a mutation operation that randomly choice between the values of a categorical hyperparameter,
    drawn from a uniform distribution.
    """

    def _mutate(
            self,
            random: np.random.RandomState,
            cs: HyperparameterConfigurationSpace,
            pop: list[HyperparameterConfiguration],
            fitness: list[float],
            optimizer: str,
            **kwargs,
    ) -> list[HyperparameterConfiguration]:
        for ind in pop:
            for key, hp in ind.items():
                if isinstance(cs[key], (Categorical, Binary)) and random.random() <= self._prob:
                    index = random.choice(len(cs[key].get_choices()))
                    ind[key] = np.array([cs[key].get_choices()[index]])
        return pop
