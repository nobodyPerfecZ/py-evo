from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import copy

from PyHyperparameterSpace.hp.continuous import Float, Integer
from PyHyperparameterSpace.hp.categorical import Categorical, Binary
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.configuration import HyperparameterConfiguration


class Mutation(ABC):
    """
    Abstract class to model the mutation phase of an evolutionary algorithm.

        Args:
            prob (float):
                Probability that the mutation occurs
    """

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

        assert len(pop) == len(fitness), \
            "Illegal population and fitness. Each individual should be assigned to a fitness value!"

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

        Args:
            mean (float):
                Mu of the normal distribution N(mu, sigma)

            std (float):
                Sigma of the normal distribution N(mu, sigma)

            prob (float):
                Probability that the mutation occurs
    """

    def __init__(self, mean: float, std: float, prob: float):
        super().__init__(prob)
        self._mean = mean
        self._std = std

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
                        ind[key] = ind[key] + self._mean + self._std * random.normal(loc=0, scale=1)
                        if ind[key] < cs[key].lb:
                            ind[key] = cs[key].lb
                        elif ind[key] >= cs[key].ub:
                            ind[key] = cs[key].ub - 1e-10
                    elif isinstance(hp, np.ndarray):
                        ind[key] = ind[key] + self._mean + self._std * random.normal(loc=0, scale=1, size=hp.shape)
                        ind[key][ind[key] < cs[key].lb] = cs[key].lb
                        ind[key][ind[key] >= cs[key].ub] = cs[key].ub - 1e-10
        return pop


class AdaptiveGaussianMutation(GaussianMutation):
    """
    Class representing a mutation operation that introduces noise to float hyperparameters.
    The mutation perturbs hyperparameters using random values drawn from a normal distribution.
    Its self-adaptive method, where the standard deviation (scale) gets changed during the evolutionary algorithm.
    """

    def __init__(
            self,
            threshold: float,
            alpha: float,
            n_generations: int,
            initial_loc: float,
            initial_scale: float,
            initial_prob: float
    ):
        super().__init__(initial_loc, initial_scale, initial_prob)
        assert 0.0 < threshold < 1.0, \
            f"Illegal threshold {threshold}. It should be in between of 0.0 < threshold < 1.0!"
        assert 0.0 < alpha < 1.0, \
            f"Illegal alpha {alpha}. It should be in between 0.0 < alpha < 1.0!"
        assert 1 <= n_generations, \
            f"Illegal n_generation {n_generations}. It should be 1 <= n_generation!"

        self._threshold = threshold
        self._alpha = alpha
        self._n_generations = n_generations

        # Necessary for the self-adaptation
        self._counter = 0
        self._last_fitness = None  # stores the fitness of the last generation
        self._success_rates = []  # stores the success rates over the last n-th generations

    def _sma_success_rate(self) -> float:
        """
        Returns:
            float:
                Simple moving average (SMA) of the success rates over the last n-th generations
        """
        return np.mean(self._success_rates)

    def _current_success_rate(self, fitness: list[float], optimizer: str) -> float:
        """
        Returns:
            float:
                The success rate of the current generation. It is calculated by the following formula:
                    success rate := #successful_mutations / population_size
        """
        population_size = len(fitness)
        successful_mutations = 0
        for last_f, f in zip(self._last_fitness, fitness):
            if (optimizer == "min" and f < last_f) or (optimizer == "max" and f > last_f):
                successful_mutations += 1
        return successful_mutations / population_size

    def _update(self, fitness: list[float], optimizer: str) -> bool:
        """
        Updates the fitness of the last generation and the success rates over the last n-th generations.

        Args:
            fitness (list[float]):
                 The fitness values of the current population

            optimizer (str):
                Type of the optimization problem
                    - optimizer="min": problem should be minimized
                    - optimizer="max": problem should be maximized
        """
        if self._last_fitness is None:
            self._last_fitness = fitness
            return False
        else:
            if len(self._success_rates) >= self._n_generations:
                self._success_rates.pop(0)
            self._success_rates += [self._current_success_rate(fitness, optimizer)]
            self._last_fitness = fitness
            self._counter = (self._counter + 1) % self._n_generations
            return self._counter == 0

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
        adjustment = self._update(fitness, optimizer)

        if adjustment:
            # Do Self-adaptation of standard deviation
            if self._sma_success_rate() > self._threshold:
                # Case: Do less exploration, more exploitation
                self._std *= self._alpha
                print(f"Updated std: {self._std}")
            elif self._sma_success_rate() < self._threshold:
                # Case: Do more exploration, less exploitation
                self._std /= self._alpha
                print(f"Updated std: {self._std}")

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
            n_generations: int,
            initial_low: Union[int, float],
            inital_high: Union[int, float],
            initial_prob: float,
    ):
        super().__init__(initial_low, inital_high, "float", initial_prob)

        assert 0.0 < threshold < 1.0, \
            f"Illegal threshold {threshold}. It should be in between of 0.0 < threshold < 1.0!"
        assert 0.0 < alpha < 1.0, \
            f"Illegal alpha {alpha}. It should be in between 0.0 < alpha < 1.0!"
        assert 1 <= n_generations, \
            f"Illegal n_generation {n_generations}. It should be 1 <= n_generation!"

        self._threshold = threshold
        self._alpha = alpha
        self._n_generations = n_generations

        # Necessary for the self-adaptation
        self._counter = 0
        self._last_fitness = None  # stores the fitness of the last generation
        self._success_rates = []  # stores the success rates over the last n-th generations

    def _sma_success_rate(self) -> float:
        """
        Returns:
            float:
                Simple moving average (SMA) of the success rates over the last n-th generations
        """
        return np.mean(self._success_rates)

    def _current_success_rate(self, fitness: list[float], optimizer: str) -> float:
        """
        Returns:
            float:
                The success rate of the current generation. It is calculated by the following formula:
                    success rate := #successful_mutations / population_size
        """
        population_size = len(fitness)
        successful_mutations = 0
        for last_f, f in zip(self._last_fitness, fitness):
            if (optimizer == "min" and f < last_f) or (optimizer == "max" and f > last_f):
                successful_mutations += 1
        return successful_mutations / population_size

    def _update(self, fitness: list[float], optimizer: str) -> bool:
        """
        Updates the fitness of the last generation and the success rates over the last n-th generations.

        Args:
            fitness (list[float]):
                 The fitness values of the current population

            optimizer (str):
                Type of the optimization problem
                    - optimizer="min": problem should be minimized
                    - optimizer="max": problem should be maximized
        """
        if self._last_fitness is None:
            self._last_fitness = fitness
            return False
        else:
            if len(self._success_rates) >= self._n_generations:
                self._success_rates.pop(0)
            self._success_rates += [self._current_success_rate(fitness, optimizer)]
            self._last_fitness = fitness
            self._counter = (self._counter + 1) % self._n_generations
            return self._counter == 0

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
        adjustment = self._update(fitness, optimizer)

        if adjustment:
            # Do Self-adaptation of standard deviation
            if self._sma_success_rate() > self._threshold:
                # Case: Do less exploration, more exploitation
                self._low *= self._alpha
                self._high *= self._alpha
                print(f"Updated lower bound: {self._low}")
                print(f"Updated upper bound: {self._high}")
            elif self._sma_success_rate() < self._threshold:
                # Case: Do more exploration, less exploitation
                self._low /= self._alpha
                self._high /= self._alpha
                print(f"Updated lower bound: {self._low}")
                print(f"Updated upper bound: {self._high}")

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
