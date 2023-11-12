import unittest

from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.hp.continuous import Float
from PyHyperparameterSpace.hp.constant import Constant

from PyEvo.ea import EA
from PyEvo.fitness import NoFitnessPreprocessing
from PyEvo.selection import TournamentSelection
from PyEvo.crossover import UniformCrossover
from PyEvo.mutation import UniformMutation


class TestEA(unittest.TestCase):
    """
    Tests the class EA.
    """

    def optimization_function(self, cfg: HyperparameterConfiguration) -> float:
        """
        Example optimization function, where we want to find the minimum value of the Rosenbrock function:
        https://en.wikipedia.org/wiki/Rosenbrock_function
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
            pop_size=20,
            selection_factor=2,
            n_iter=None,
            walltime_limit=10,
            n_cores=1,
            seed=0,
            optimizer="min",
            fitness_preprocessors=NoFitnessPreprocessing(),
            selections=TournamentSelection(),
            crossovers=UniformCrossover(),
            mutations=UniformMutation(low=-0.5, high=0.5, hp_type="float", prob=1.0),
        )

        self.cfg = HyperparameterConfiguration(
            values={
                "a": 1,
                "b": 100,
                "x": 1.0122048232965808,
                "y": 1.0238352137771527,
            }
        )

    def test_initialize_population_randomly(self):
        """
        Tests the method initialize_population() with no given configuration.
        """
        pop = self.EA._initialize_population()

        self.assertEqual(self.EA._pop_size, len(pop))

    def test_initialize_population_local_search(self):
        """
        Tests the method initialize_population() with given configuration (local search).
        """
        pop = self.EA._initialize_population(self.cfg)

        self.assertEqual(self.EA._pop_size, len(pop))
        # Check if all individuals in population are equal to the given configuration
        self.assertTrue(all(ind == self.cfg for ind in pop))

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
        incumbent = self.EA.incumbent

        self.assertIsNone(incumbent)

    def test_fit_randomly(self):
        """
        Tests the method fit() with no given configuration.
        """
        self.EA.fit()
        incumbent = self.EA.incumbent
        value = self.optimization_function(incumbent)

        self.assertIsNotNone(self.EA.incumbent)
        self.assertLessEqual(value, 0.01)

    def test_fit_local_search(self):
        """
        Tests the method fit() with given configuration (local search).
        """
        self.EA.fit(self.cfg)
        incumbent = self.EA.incumbent
        value = self.optimization_function(incumbent)

        self.assertIsNotNone(self.EA.incumbent)
        self.assertLessEqual(value, 0.01)


if __name__ == '__main__':
    unittest.main()
