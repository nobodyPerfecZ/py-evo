import unittest
import numpy as np

from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.hp.continuous import Float, Integer
from PyHyperparameterSpace.hp.categorical import Categorical
from PyHyperparameterSpace.hp.constant import Constant
from PyHyperparameterSpace.dist.continuous import Normal

from PyEvo.selection import ElitistSelection, TournamentSelection, FitnessProportionalSelection


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
        selected, fitness_selected, non_selected, fitness_non_selected = self.selector.select(self.random, self.cs,
                                                                                              self.pop, self.fitness,
                                                                                              self.optimizer_min,
                                                                                              self.n_select)

        self.assertEqual(self.n_select, len(selected))
        self.assertEqual(selected[0], self.pop[4])
        self.assertEqual(selected[1], self.pop[0])
        self.assertEqual(selected[2], self.pop[9])
        self.assertEqual(selected[3], self.pop[8])
        self.assertEqual(selected[4], self.pop[5])

        self.assertEqual(self.n_select, len(fitness_selected))
        self.assertEqual(fitness_selected[0], self.fitness[4])
        self.assertEqual(fitness_selected[1], self.fitness[0])
        self.assertEqual(fitness_selected[2], self.fitness[9])
        self.assertEqual(fitness_selected[3], self.fitness[8])
        self.assertEqual(fitness_selected[4], self.fitness[5])

        self.assertEqual(len(self.pop) - self.n_select, len(non_selected))
        self.assertEqual(non_selected[0], self.pop[1])
        self.assertEqual(non_selected[1], self.pop[2])
        self.assertEqual(non_selected[2], self.pop[7])
        self.assertEqual(non_selected[3], self.pop[3])
        self.assertEqual(non_selected[4], self.pop[6])

        self.assertEqual(len(self.fitness) - self.n_select, len(fitness_non_selected))
        self.assertEqual(fitness_non_selected[0], self.fitness[1])
        self.assertEqual(fitness_non_selected[1], self.fitness[2])
        self.assertEqual(fitness_non_selected[2], self.fitness[7])
        self.assertEqual(fitness_non_selected[3], self.fitness[3])
        self.assertEqual(fitness_non_selected[4], self.fitness[6])

    def test_select_maximize(self):
        """
        Tests the method select() with maximizer.
        """
        selected, fitness_selected, non_selected, fitness_non_selected = self.selector.select(self.random, self.cs,
                                                                                              self.pop, self.fitness,
                                                                                              self.optimizer_max,
                                                                                              self.n_select)

        self.assertEqual(self.n_select, len(selected))
        self.assertEqual(selected[0], self.pop[6])
        self.assertEqual(selected[1], self.pop[3])
        self.assertEqual(selected[2], self.pop[7])
        self.assertEqual(selected[3], self.pop[2])
        self.assertEqual(selected[4], self.pop[1])

        self.assertEqual(self.n_select, len(fitness_selected))
        self.assertEqual(fitness_selected[0], self.fitness[6])
        self.assertEqual(fitness_selected[1], self.fitness[3])
        self.assertEqual(fitness_selected[2], self.fitness[7])
        self.assertEqual(fitness_selected[3], self.fitness[2])
        self.assertEqual(fitness_selected[4], self.fitness[1])

        self.assertEqual(len(self.pop) - self.n_select, len(non_selected))
        self.assertEqual(non_selected[0], self.pop[5])
        self.assertEqual(non_selected[1], self.pop[8])
        self.assertEqual(non_selected[2], self.pop[9])
        self.assertEqual(non_selected[3], self.pop[0])
        self.assertEqual(non_selected[4], self.pop[4])

        self.assertEqual(len(self.fitness) - self.n_select, len(fitness_non_selected))
        self.assertEqual(fitness_non_selected[0], self.fitness[5])
        self.assertEqual(fitness_non_selected[1], self.fitness[8])
        self.assertEqual(fitness_non_selected[2], self.fitness[9])
        self.assertEqual(fitness_non_selected[3], self.fitness[0])
        self.assertEqual(fitness_non_selected[4], self.fitness[4])


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

        selected, fitness_selected, non_selected, fitness_non_selected = self.selector.select(self.random, self.cs,
                                                                                              self.pop, self.fitness,
                                                                                              self.optimizer_min,
                                                                                              self.n_select)
        self.assertEqual(self.n_select, len(selected))
        self.assertEqual(selected[0], self.pop[4])
        self.assertEqual(selected[1], self.pop[0])
        self.assertEqual(selected[2], self.pop[3])
        self.assertEqual(selected[3], self.pop[5])
        self.assertEqual(selected[4], self.pop[1])

        self.assertEqual(self.n_select, len(fitness_selected))
        self.assertEqual(fitness_selected[0], self.fitness[4])
        self.assertEqual(fitness_selected[1], self.fitness[0])
        self.assertEqual(fitness_selected[2], self.fitness[3])
        self.assertEqual(fitness_selected[3], self.fitness[5])
        self.assertEqual(fitness_selected[4], self.fitness[1])

        self.assertEqual(len(self.pop) - self.n_select, len(non_selected))
        self.assertEqual(non_selected[0], self.pop[2])
        self.assertEqual(non_selected[1], self.pop[6])
        self.assertEqual(non_selected[2], self.pop[7])
        self.assertEqual(non_selected[3], self.pop[8])
        self.assertEqual(non_selected[4], self.pop[9])

        self.assertEqual(len(self.fitness) - self.n_select, len(fitness_non_selected))
        self.assertEqual(fitness_non_selected[0], self.fitness[2])
        self.assertEqual(fitness_non_selected[1], self.fitness[6])
        self.assertEqual(fitness_non_selected[2], self.fitness[7])
        self.assertEqual(fitness_non_selected[3], self.fitness[8])
        self.assertEqual(fitness_non_selected[4], self.fitness[9])

    def test_select_maximizer(self):
        """
        Tests the method select() with maximizer.
        """
        selected, fitness_selected, non_selected, fitness_non_selected = self.selector.select(self.random, self.cs,
                                                                                              self.pop, self.fitness,
                                                                                              self.optimizer_max,
                                                                                              self.n_select)

        self.assertEqual(self.n_select, len(selected))
        self.assertEqual(selected[0], self.pop[6])
        self.assertEqual(selected[1], self.pop[3])
        self.assertEqual(selected[2], self.pop[2])
        self.assertEqual(selected[3], self.pop[4])
        self.assertEqual(selected[4], self.pop[1])

        self.assertEqual(self.n_select, len(fitness_selected))
        self.assertEqual(fitness_selected[0], self.fitness[6])
        self.assertEqual(fitness_selected[1], self.fitness[3])
        self.assertEqual(fitness_selected[2], self.fitness[2])
        self.assertEqual(fitness_selected[3], self.fitness[4])
        self.assertEqual(fitness_selected[4], self.fitness[1])

        self.assertEqual(len(self.pop) - self.n_select, len(non_selected))
        self.assertEqual(non_selected[0], self.pop[0])
        self.assertEqual(non_selected[1], self.pop[5])
        self.assertEqual(non_selected[2], self.pop[7])
        self.assertEqual(non_selected[3], self.pop[8])
        self.assertEqual(non_selected[4], self.pop[9])

        self.assertEqual(len(self.fitness) - self.n_select, len(fitness_non_selected))
        self.assertEqual(fitness_non_selected[0], self.fitness[0])
        self.assertEqual(fitness_non_selected[1], self.fitness[5])
        self.assertEqual(fitness_non_selected[2], self.fitness[7])
        self.assertEqual(fitness_non_selected[3], self.fitness[8])
        self.assertEqual(fitness_non_selected[4], self.fitness[9])


class TestFitnessProportionalSelection(unittest.TestCase):
    """
    Tests the class FitnessProportionalSelection.
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
        self.fitness = [0.004914004914004915, 0.09705159705159706, 0.14496314496314497, 0.18673218673218675, 0.0, 0.09336609336609338, 0.21130221130221133, 0.15356265356265356, 0.05896805896805897, 0.049140049140049144]
        self.optimizer_min = "min"
        self.optimizer_max = "max"
        self.n_select = 5
        self.selector = FitnessProportionalSelection()

    def test_select_minimizer(self):
        """
        Tests the method select() with minimizer.
        """
        selected, fitness_selected, non_selected, fitness_non_selected = self.selector.select(self.random, self.cs,
                                                                                              self.pop, self.fitness,
                                                                                              self.optimizer_min,
                                                                                              self.n_select)
        self.assertEqual(self.n_select, len(selected))
        self.assertEqual(selected[0], self.pop[6])
        self.assertEqual(selected[1], self.pop[3])
        self.assertEqual(selected[2], self.pop[7])
        self.assertEqual(selected[3], self.pop[5])
        self.assertEqual(selected[4], self.pop[8])

        self.assertEqual(self.n_select, len(fitness_selected))
        self.assertEqual(fitness_selected[0], self.fitness[6])
        self.assertEqual(fitness_selected[1], self.fitness[3])
        self.assertEqual(fitness_selected[2], self.fitness[7])
        self.assertEqual(fitness_selected[3], self.fitness[5])
        self.assertEqual(fitness_selected[4], self.fitness[8])

        self.assertEqual(len(self.pop) - self.n_select, len(non_selected))
        self.assertEqual(non_selected[0], self.pop[0])
        self.assertEqual(non_selected[1], self.pop[1])
        self.assertEqual(non_selected[2], self.pop[2])
        self.assertEqual(non_selected[3], self.pop[4])
        self.assertEqual(non_selected[4], self.pop[9])

        self.assertEqual(len(self.fitness) - self.n_select, len(fitness_non_selected))
        self.assertEqual(fitness_non_selected[0], self.fitness[0])
        self.assertEqual(fitness_non_selected[1], self.fitness[1])
        self.assertEqual(fitness_non_selected[2], self.fitness[2])
        self.assertEqual(fitness_non_selected[3], self.fitness[4])
        self.assertEqual(fitness_non_selected[4], self.fitness[9])

    def test_select_maximizer(self):
        """
        Tests the method select() with maximizer.
        """
        selected, fitness_selected, non_selected, fitness_non_selected = self.selector.select(self.random, self.cs,
                                                                                              self.pop, self.fitness,
                                                                                              self.optimizer_max,
                                                                                              self.n_select)

        self.assertEqual(self.n_select, len(selected))
        self.assertEqual(selected[0], self.pop[6])
        self.assertEqual(selected[1], self.pop[3])
        self.assertEqual(selected[2], self.pop[7])
        self.assertEqual(selected[3], self.pop[5])
        self.assertEqual(selected[4], self.pop[8])

        self.assertEqual(self.n_select, len(fitness_selected))
        self.assertEqual(fitness_selected[0], self.fitness[6])
        self.assertEqual(fitness_selected[1], self.fitness[3])
        self.assertEqual(fitness_selected[2], self.fitness[7])
        self.assertEqual(fitness_selected[3], self.fitness[5])
        self.assertEqual(fitness_selected[4], self.fitness[8])

        self.assertEqual(len(self.pop) - self.n_select, len(non_selected))
        self.assertEqual(non_selected[0], self.pop[0])
        self.assertEqual(non_selected[1], self.pop[1])
        self.assertEqual(non_selected[2], self.pop[2])
        self.assertEqual(non_selected[3], self.pop[4])
        self.assertEqual(non_selected[4], self.pop[9])

        self.assertEqual(len(self.fitness) - self.n_select, len(fitness_non_selected))
        self.assertEqual(fitness_non_selected[0], self.fitness[0])
        self.assertEqual(fitness_non_selected[1], self.fitness[1])
        self.assertEqual(fitness_non_selected[2], self.fitness[2])
        self.assertEqual(fitness_non_selected[3], self.fitness[4])
        self.assertEqual(fitness_non_selected[4], self.fitness[9])


if __name__ == '__main__':
    unittest.main()
