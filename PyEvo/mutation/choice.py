import numpy as np
from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.hp.categorical import Categorical
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace

from PyEvo.mutation.abstract_mutation import Mutation


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
                if isinstance(cs[key], Categorical) and random.random() <= self._prob:
                    index = random.choice(len(cs[key].get_choices()))
                    ind[key] = np.array([cs[key].get_choices()[index]])
        return pop
