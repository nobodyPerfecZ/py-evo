import unittest
import numpy as np

from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.hp.continuous import Float, Integer
from PyHyperparameterSpace.hp.categorical import Categorical
from PyHyperparameterSpace.hp.constant import Constant
from PyHyperparameterSpace.dist.continuous import Normal

from PyEvo.selection import ElitistSelection, TournamentSelection


class TestElitistSelection(unittest.TestCase):
    """
    Tests the class ElitistSelection.
    """

    def setUp(self):
        # Construct the hyperparameter configuration space
        self.random = np.random.RandomState(0)
        self.cs = HyperparameterConfigurationSpace(
            values={
                "max_episodes": Constant("max_episodes", default=3),
                "max_episode_length": Constant("max_episode_length", default=1000),
                "hidden1_shape": Constant("hidden1_shape", default=64),
                "hidden2_shape": Constant("hidden2_shape", default=32),
                "fc1.weight": Float("fc1.weight", bounds=(-1.0, 1.0), shape=(64, 8)),
                "fc1.bias": Float("fc1.bias", bounds=(-1.0, 1.0), shape=(64,)),
                "fc2.weight": Float("fc2.weight", bounds=(-1.0, 1.0), shape=(32, 64)),
                "fc2.bias": Float("fc2.bias", bounds=(-1.0, 1.0), shape=(32,)),
                "fc3.weight": Float("fc3.weight", bounds=(-1.0, 1.0), shape=(4, 32)),
                "fc3.bias": Float("fc3.bias", bounds=(-1.0, 1.0), shape=(4,)),
            },
            seed=0,
        )
        self.pop = self.cs.sample_configuration(10)
        self.fitness = [-92, -17, 22, 56, -96, -20, 76, 29, -48, -56]
        self.optimizer_min = "min"
        self.optimizer_max = "max"
        self.n_select = 5
        self.selector = ElitistSelection()

    def test_select_minimizer(self):
        """
        Tests the method select() with minimizer.
        """
        new_pop = self.selector.select(self.random, self.cs, self.pop, self.fitness, self.optimizer_min, self.n_select)

        self.assertEqual(self.n_select, len(new_pop))
        self.assertEqual(new_pop[0], self.pop[4])
        self.assertEqual(new_pop[1], self.pop[0])
        self.assertEqual(new_pop[2], self.pop[9])
        self.assertEqual(new_pop[3], self.pop[8])
        self.assertEqual(new_pop[4], self.pop[5])

    def test_select_maximize(self):
        """
        Tests the method select() with maximizer.
        """
        new_pop = self.selector.select(self.random, self.cs, self.pop, self.fitness, self.optimizer_max, self.n_select)

        self.assertEqual(self.n_select, len(new_pop))
        self.assertEqual(new_pop[0], self.pop[6])
        self.assertEqual(new_pop[1], self.pop[3])
        self.assertEqual(new_pop[2], self.pop[7])
        self.assertEqual(new_pop[3], self.pop[2])
        self.assertEqual(new_pop[4], self.pop[1])


class TestTournamentSelection(unittest.TestCase):
    """
    Tests the class TournamentSelection.
    """

    def setUp(self):
        # Construct the hyperparameter configuration space
        self.random = np.random.RandomState(0)
        self.cs = HyperparameterConfigurationSpace(
            values={
                "max_episodes": Constant("max_episodes", default=3),
                "max_episode_length": Constant("max_episode_length", default=1000),
                "hidden1_shape": Constant("hidden1_shape", default=64),
                "hidden2_shape": Constant("hidden2_shape", default=32),
                "fc1.weight": Float("fc1.weight", bounds=(-1.0, 1.0), shape=(64, 8)),
                "fc1.bias": Float("fc1.bias", bounds=(-1.0, 1.0), shape=(64,)),
                "fc2.weight": Float("fc2.weight", bounds=(-1.0, 1.0), shape=(32, 64)),
                "fc2.bias": Float("fc2.bias", bounds=(-1.0, 1.0), shape=(32,)),
                "fc3.weight": Float("fc3.weight", bounds=(-1.0, 1.0), shape=(4, 32)),
                "fc3.bias": Float("fc3.bias", bounds=(-1.0, 1.0), shape=(4,)),
            },
            seed=0,
        )
        self.pop = self.cs.sample_configuration(10)
        self.fitness = [-92, -17, 22, 56, -96, -20, 76, 29, -48, -56]
        self.optimizer_min = "min"
        self.optimizer_max = "max"
        self.n_select = 5
        self.selector = TournamentSelection()

    def test_select_minimizer(self):
        """
        Tests the method select() with minimizer.
        """

        new_pop = self.selector.select(self.random, self.cs, self.pop, self.fitness, self.optimizer_min, self.n_select)

        self.assertEqual(self.n_select, len(new_pop))
        self.assertIn(new_pop[0], self.pop)
        self.assertIn(new_pop[1], self.pop)
        self.assertIn(new_pop[2], self.pop)
        self.assertIn(new_pop[3], self.pop)
        self.assertIn(new_pop[4], self.pop)

    def test_select_maximizer(self):
        """
        Tests the method select() with maximizer.
        """
        new_pop = self.selector.select(self.random, self.cs, self.pop, self.fitness, self.optimizer_max, self.n_select)

        self.assertEqual(self.n_select, len(new_pop))
        self.assertIn(new_pop[0], self.pop)
        self.assertIn(new_pop[1], self.pop)
        self.assertIn(new_pop[2], self.pop)
        self.assertIn(new_pop[3], self.pop)
        self.assertIn(new_pop[4], self.pop)


if __name__ == '__main__':
    unittest.main()
