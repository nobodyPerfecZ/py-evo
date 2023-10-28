from typing import Callable, Union
import multiprocessing
import numpy as np
import time
import copy

from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.configuration import HyperparameterConfiguration

from PyEvo.selection import Selection
from PyEvo.crossover import Crossover
from PyEvo.mutation import Mutation


class EA:
    """ Classical Evolutionary Algorithm. """

    def __init__(
            self,
            problem: Callable[[HyperparameterConfiguration], float],
            cs: HyperparameterConfigurationSpace,
            pop_size: int,
            selection_factor: int,
            n_iter: Union[int, None],
            walltime_limit: Union[float, None],
            n_cores: int,
            seed: int,
            optimizer: str,
            selections: Selection,
            crossovers: Union[Crossover, list[Crossover]],
            mutations: Union[Mutation, list[Mutation]],
    ):
        assert 2 <= pop_size, f"Illegal pop_size {pop_size}. It should be 2 <= pop_size!"
        assert 2 <= selection_factor <= pop_size, \
            f"Illegal selection_factor {selection_factor}. It should be 2 <= selection_factor <= pop_size!"
        assert (n_iter is not None and walltime_limit is None) or (n_iter is None and walltime_limit is not None), \
            "Either n_iter or walltime_limit should be assigned to a value!"
        if n_iter:
            assert 1 <= n_iter, f"Illegal n_iter {n_iter}. It should be 1 <= n_iter!"
        if walltime_limit:
            assert 1 <= walltime_limit, f"Illegal walltime_limit {walltime_limit}. It should be 1 <= walltime_limit!"
        assert 1 <= n_cores <= multiprocessing.cpu_count(), \
            f"Illegal n_cores {n_cores}. It should be in 1 <= n_cores <= {multiprocessing.cpu_count()}!"
        assert optimizer == "min" or optimizer == "max", f"Illegal optimizer {optimizer}. It should be 'min' or 'max'!"

        self._problem = problem
        self._cs = cs
        self._pop_size = pop_size
        self._selection_factor = selection_factor
        self._n_cores = n_cores
        self._n_iter = n_iter
        self._walltime_limit = walltime_limit
        self._random = np.random.RandomState(seed)
        self._optimizer = optimizer
        self._selections = selections
        self._crossovers = crossovers if isinstance(crossovers, list) else [crossovers]
        self._mutations = mutations if isinstance(mutations, list) else [mutations]

        # Necessary for the EA Loop
        self._pop = None
        self._start_time = None
        self._cur_iter = None
        self._incumbent = None

    def _initialize_population(self, cfg: Union[HyperparameterConfiguration, None] = None) \
            -> list[HyperparameterConfiguration]:
        """
        Returns the population of the first generation.

        If cfg is given, then the population contains _pop_size of the same individual.
        It represents the continuation of the local search.

        If cfg is not given, then population contains _pop_size of randomly sampled individual
        from the configuration space.

        Args:
            cfg (Union[HyperparameterConfiguration, None]):
                Individual from where we start the local search

        Returns:
            list[HyperparameterConfiguration]:
                Population of the first generation
        """
        if cfg is not None:
            return [copy.deepcopy(cfg) for _ in range(self._pop_size)]
        else:
            return self._cs.sample_configuration(self._pop_size)

    def _check_n_iter(self) -> bool:
        """
        Checks if the current iteration (_cur_iter) extends the maximum allowed iterations (_n_iter) for the EA.
        If _n_iter is not given, then it will always return False.

        Returns:
            bool:
                True, if _cur_iter extends _n_iter, otherwise False
        """
        if self._n_iter is not None:
            return self._cur_iter >= self._n_iter
        return False

    def _check_walltime_limit(self) -> bool:
        """
        Checks if the used time extends the maximum allowed time budget (_walltime_limit) for the EA.
        If _walltime_limit is not given, then it will always return False.

        Returns:
            bool:
                True, if current used time extends _walltime_limit, otherwise False
        """
        if self._walltime_limit is not None and self._start_time is not None:
            used_time = time.time() - self._start_time
            return used_time >= self._walltime_limit
        return False

    def _update_incumbent(self, fitness: list[float]):
        """
        Updates the incumbent by assigning it to the best individual from the current population.

        Args:
            fitness (list[float]):
                Fitness values for each individual in population
        """
        if self._optimizer == "max":
            select = max
        elif self._optimizer == "min":
            select = min
        self._incumbent = self._pop[select(range(len(fitness)), key=fitness.__getitem__)]

    def _evaluate(self) -> list[float]:
        """
        Evaluates the current population in parallel and returns the fitness values of each individual.
        If an individual was already evaluated before, then it will just return the fitness stored in the cache.

        Returns:
            list[float]:
                Fitness values for each individual in population
        """
        if self._n_cores == 1:
            # Case: Evaluate with single core
            fitness = [self._problem(ind) for ind in self._pop]
        else:
            # Case: Evaluate the individual in parallel
            pool = multiprocessing.Pool(processes=self._n_cores)
            fitness = pool.map(self._problem, self._pop)

        # Update the cache with the new rewards
        self._update_incumbent(fitness)
        return fitness

    @property
    def incumbent(self) -> Union[HyperparameterConfiguration, None]:
        """
        Returns:
            Union[HyperparameterConfiguration, None]:
                Individual with the best fitness value. If no individuals are evaluated before, then it will return
                None.
        """
        return self._incumbent

    def fit(self, cfg: Union[HyperparameterConfiguration, None] = None):
        """
        Main loop for the Evolutionary Algorithm (EA), employing the following strategy:

        1. Initialization: Generate an initial population of individuals, according to the given hyperparameter
        configuration space or given configuration.

            1. If configuration is given then we continue the local search. The first population contains (_pop_size)
            of the same configuration.
            2. If configuration is not given then we sample randomly individuals from the configuration space.

        2. Evaluation: Assess the fitness of each individual in the population based on the optimization objective
        (min or max).

        3. Evolution Cycle: Repeat the following steps until number of iterations (_n_iter) or time budget
        (_walltime_budget) is met:
            1. Evaluation: Assess the fitness of each individual in the population based on the optimization objective
            (min or max).
            2. Selection: Choose individuals from the current population as parents for reproduction.
            3. Crossover: Apply the crossover operation to create new childs from pairs of parents.
            4. Mutation: Introduce random changes to the genetic information of some childs.
            5. Replacement: Form the next generation with the selected parents and childs to create the next population

        4. After the .fit() method is called, the best solution can be returned by calling .incumbent

        Args:
            cfg (Union[HyperparameterConfiguration, None]):
                Individual from where we start the local search
        """
        self._start_time = time.time()
        self._cur_iter = 0  # current number of generation
        self._pop = self._initialize_population(cfg)

        while True:
            print(f"#################################################################")
            print(f"###################### Generation: {self._cur_iter + 1:6} #######################")
            print(f"#################################################################")
            print(f"Remaining Walltime: {(self._walltime_limit - (time.time() - self._start_time)):.4f}")

            if self._check_walltime_limit():
                return

            # Evaluate all genomes in the population
            fitness = self._evaluate()
            print(f"Fitness: {fitness}")
            if self._check_walltime_limit():
                return

            if not self._check_n_iter():
                # Perform selection
                pop = self._selections.select(self._random, self._cs, self._pop, fitness, self._optimizer,
                                              self._pop_size // self._selection_factor)

                if self._check_walltime_limit():
                    return

                # Perform crossover
                childs_size = self._pop_size - len(pop)
                childs = pop
                for crossover in self._crossovers:
                    childs = crossover.crossover(self._random, self._cs, childs, fitness, self._optimizer, childs_size)

                if self._check_walltime_limit():
                    return

                # Perform mutation
                for mutation in self._mutations:
                    childs = mutation.mutate(self._random, self._cs, childs, fitness, self._optimizer)

                if self._check_walltime_limit():
                    return

                # Combine childs and selected population to the new population
                self._pop = pop + childs
            else:
                # Case: n_iter is reached!
                # Stop the algorithm
                return

            # Update the generation counter
            self._cur_iter += 1
