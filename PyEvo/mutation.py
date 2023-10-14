import copy
from abc import ABC
from typing import Any
from decimal import Decimal
import numpy as np
import torch


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

        if np.random.random() > self._mut_prob:
            # Case: Don't do the mutation
            return pop

        # Case: Do the mutation
        for ind in pop:
            for key in ind:
                if isinstance(ind[key], Decimal):
                    ind[key] += np.random.normal(loc=self._mean, scale=self._std)
                elif isinstance(ind[key], np.ndarray) and np.issubdtype(ind[key].dtype, np.float):
                    ind[key] += np.random.normal(loc=self._mean, scale=self._std, size=ind.shape)
                elif isinstance(ind[key], torch.Tensor) and torch.is_floating_point(ind[key]):
                    ind[key] += torch.normal(mean=self._mean, std=self._std, size=ind.size())
        return pop
