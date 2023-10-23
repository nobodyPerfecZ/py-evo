import unittest

from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.hp.continuous import Float
from PyHyperparameterSpace.hp.constant import Constant

from PyEvo.ea import EA
from PyEvo.selection import TournamentSelection
from PyEvo.crossover import UniformCrossover
from PyEvo.mutation import UniformMutation


class TestEA(unittest.TestCase):
    """
    Tests the class EA.
    """

    def optimization_function(self, cfg: HyperparameterConfiguration) -> float:
        """
        Example optimization function, where we want to find the minimum value of the Rosenbrock function: https://en.wikipedia.org/wiki/Rosenbrock_function
        """
        # Extract the important hps
        a = cfg["a"]
        b = cfg["b"]
        x = cfg["x"]
        y = cfg["y"]
        return (a - x) ** 2 + b * (y - x ** 2) ** 2

    def setUp(self):
        self.cs = HyperparameterConfigurationSpace(
            values={
                "a": Constant("a", default=1),
                "b": Constant("b", default=100),
                "x": Float("x", bounds=(-2.0, 2.0), shape=(1,)),
                "y": Float("y", bounds=(-1.0, 3.0), shape=(1,)),
            },
            seed=0,
        )

        self.EA = EA(
            problem=self.optimization_function,
            cs=self.cs,
            pop_size=10,
            selection_factor=3,
            n_iter=None,
            walltime_limit=100,
            n_cores=1,
            seed=0,
            optimizer="min",
            selections=TournamentSelection(),
            crossovers=UniformCrossover(),
            mutations=UniformMutation(low=-0.5, high=0.5, hp_type="float", prob=1.0),
        )

    def test_initialize_population(self):
        """
        Tests the method initialize_population().
        """
        pop = self.EA._initialize_population()

        self.assertEqual(self.EA._pop_size, len(pop))

    def test_check_n_iter(self):
        """
        Tests the method check_n_iter()
        """
        self.assertFalse(self.EA._check_n_iter())

    def test_check_walltime_limit(self):
        """
        Tests the method check_walltime_limit().
        """
        self.assertFalse(self.EA._check_walltime_limit())

    def test_incumbent(self):
        """
        Tests the property incumbent.
        """
        self.EA.incumbent

        self.assertIsNone(self.EA.incumbent)

    def test_fit(self):
        """
        Tests the method fit().
        """
        self.EA.fit()
        incumbent = self.EA.incumbent
        value = self.optimization_function(incumbent)

        self.assertIsNotNone(self.EA.incumbent)
        self.assertLessEqual(value, 0.01)


if __name__ == '__main__':
    unittest.main()
