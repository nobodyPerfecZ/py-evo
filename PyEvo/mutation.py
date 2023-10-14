import copy
from abc import ABC
from typing import Any
import numpy as np


class Mutation(ABC):
    """
    Abstract class for performing a mutation over a given population.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def mutate(self, population: list[dict[str, Any]], inplace: bool = True) -> list[dict[str, Any]]:
        pass


class GaussianMutation(Mutation):

    def __init__(self, mean: float, std: float, mut_prob: float, **kwargs):
        super().__init__(**kwargs)
        self._mean = mean
        self._std = std
        self._mut_prob = mut_prob

    def mutate(self, population: list[dict[str, Any]], inplace: bool = True) -> list[dict[str, Any]]:
        # TODO: Add Max-/Min- Constraints
        if not inplace:
            pop = copy.deepcopy(population)
        else:
            pop = population

        if np.random() > self._mut_prob:
            # Case: Don't do the mutation
            return pop
        # Case: Do the mutation
        for ind in pop:
            noise = np.random.normal(loc=self._mean, scale=self._std)
            ind += noise
        return pop
