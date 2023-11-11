import unittest
import numpy as np

from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.hp.continuous import Float
from PyHyperparameterSpace.hp.categorical import Categorical
from PyHyperparameterSpace.hp.constant import Constant

from PyEvo.mutation import GaussianMutation, AdaptiveGaussianMutation, UniformMutation, AdaptiveUniformMutation, \
    ChoiceMutation


class TestGaussianMutation(unittest.TestCase):
    """
    Tests the class GaussianMutation.
    """

    def setUp(self):
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
        self.mutator = GaussianMutation(mean=0.0, std=1.0, prob=1.0)

    def test_mutate(self):
        """
        Tests the method mutate().
        """
        new_pop = self.mutator.mutate(self.random, self.cs, self.pop, self.fitness, self.optimizer_min)

        self.assertEqual(len(self.pop), len(new_pop))
        self.assertNotEqual(self.pop, new_pop)


class TestAdaptiveGaussianMutation(unittest.TestCase):
    """
    Tests the class AdaptiveGaussianMutation.
    """

    def setUp(self):
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
        self.fitness1 = [-92, -17, 22, 56, -96, -20, 76, 29, -48, -56]
        self.fitness2 = [-90, -15, 24, 58, -94, -18, 78, 31, -46, -54]
        self.fitness3 = [-88, -13, 26, 60, -92, -16, 80, 33, -44, -52]
        self.fitness4 = [-86, -11, 28, 62, -90, -14, 82, 35, -42, -50]
        self.optimizer_min = "min"
        self.optimizer_max = "max"
        self.initial_scale = 1.0
        self.mutator = AdaptiveGaussianMutation(
            threshold=0.2,
            alpha=0.95,
            n_generations=2,
            initial_loc=0.0,
            initial_scale=self.initial_scale,
            initial_prob=1.0
        )

    def test_mutate_minimizer(self):
        """
        Tests the method mutate() for minimization problem.
        """
        self.mutator.mutate(self.random, self.cs, self.pop, self.fitness1, self.optimizer_min)
        self.mutator.mutate(self.random, self.cs, self.pop, self.fitness2, self.optimizer_min)
        self.mutator.mutate(self.random, self.cs, self.pop, self.fitness3, self.optimizer_min)
        new_pop = self.mutator.mutate(self.random, self.cs, self.pop, self.fitness4, self.optimizer_min)

        self.assertEqual(len(self.pop), len(new_pop))
        self.assertNotEqual(self.pop, new_pop)

        # Check if the updates are done right
        self.assertGreater(self.mutator._std, self.initial_scale)

    def test_mutate_maximizer(self):
        """
        Tests the method mutate() for maximization problem.
        """
        self.mutator.mutate(self.random, self.cs, self.pop, self.fitness1, self.optimizer_max)
        self.mutator.mutate(self.random, self.cs, self.pop, self.fitness2, self.optimizer_max)
        self.mutator.mutate(self.random, self.cs, self.pop, self.fitness3, self.optimizer_max)
        new_pop = self.mutator.mutate(self.random, self.cs, self.pop, self.fitness4, self.optimizer_max)

        self.assertEqual(len(self.pop), len(new_pop))
        self.assertNotEqual(self.pop, new_pop)

        # Check if the updates are done right
        self.assertLess(self.mutator._std, self.initial_scale)


class TestUniformMutation(unittest.TestCase):
    """
    Tests the class UniformMutation.
    """

    def setUp(self):
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
                "fc4.bias": Float("fc4.bias", bounds=(-1.0, 1.0), shape=(1,))
            },
            seed=0,
        )
        self.pop = self.cs.sample_configuration(10)
        self.fitness = [-92, -17, 22, 56, -96, -20, 76, 29, -48, -56]
        self.optimizer_min = "min"
        self.optimizer_max = "max"
        self.mutator_float = UniformMutation(low=-0.5, high=0.5, hp_type="float", prob=1.0)
        self.mutator_int = UniformMutation(low=-1, high=2, hp_type="int", prob=1.0)

    def test_mutate_float(self):
        """
        Tests the method mutate() for float hyperparameters.
        """
        new_pop = self.mutator_float.mutate(self.random, self.cs, self.pop, self.fitness, self.optimizer_min)

        self.assertEqual(len(self.pop), len(new_pop))
        self.assertNotEqual(self.pop, new_pop)

    def test_mutate_integer(self):
        """
        Tests the method mutate() for integer hyperparameters.
        """
        new_pop = self.mutator_int.mutate(self.random, self.cs, self.pop, self.fitness, self.optimizer_min)

        self.assertEqual(len(self.pop), len(new_pop))
        self.assertEqual(self.pop, new_pop)


class TestAdaptiveUniformMutation(unittest.TestCase):
    """
    Tests the class AdaptiveUniformMutation.
    """

    def setUp(self):
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
        self.fitness1 = [-92, -17, 22, 56, -96, -20, 76, 29, -48, -56]
        self.fitness2 = [-90, -15, 24, 58, -94, -18, 78, 31, -46, -54]
        self.fitness3 = [-88, -13, 26, 60, -92, -16, 80, 33, -44, -52]
        self.fitness4 = [-86, -11, 28, 62, -90, -14, 82, 35, -42, -50]
        self.optimizer_min = "min"
        self.optimizer_max = "max"
        self.initial_low = -1.0
        self.initial_high = 1.0
        self.mutator = AdaptiveUniformMutation(
            threshold=0.2,
            alpha=0.95,
            n_generations=2,
            initial_low=self.initial_low,
            inital_high=self.initial_high,
            initial_prob=1.0,
        )

    def test_mutate_minimizer(self):
        """
        Tests the method mutate() for minimization problem.
        """
        self.mutator.mutate(self.random, self.cs, self.pop, self.fitness1, self.optimizer_min)
        self.mutator.mutate(self.random, self.cs, self.pop, self.fitness2, self.optimizer_min)
        self.mutator.mutate(self.random, self.cs, self.pop, self.fitness3, self.optimizer_min)
        new_pop = self.mutator.mutate(self.random, self.cs, self.pop, self.fitness4, self.optimizer_min)

        self.assertEqual(len(self.pop), len(new_pop))
        self.assertNotEqual(self.pop, new_pop)

        # Check if the updates are done right
        self.assertLess(self.mutator._low, self.initial_low)
        self.assertGreater(self.mutator._high, self.initial_high)

    def test_mutate_maximizer(self):
        """
        Tests the method mutate() for maximization problem.
        """
        self.mutator.mutate(self.random, self.cs, self.pop, self.fitness1, self.optimizer_max)
        self.mutator.mutate(self.random, self.cs, self.pop, self.fitness2, self.optimizer_max)
        self.mutator.mutate(self.random, self.cs, self.pop, self.fitness3, self.optimizer_max)
        new_pop = self.mutator.mutate(self.random, self.cs, self.pop, self.fitness4, self.optimizer_max)

        self.assertEqual(len(self.pop), len(new_pop))
        self.assertNotEqual(self.pop, new_pop)

        # Check if the updates are done right
        self.assertGreater(self.mutator._low, self.initial_low)
        self.assertLess(self.mutator._high, self.initial_high)


class TestChoiceMutation(unittest.TestCase):
    """
    Tests the class ChoiceMutation.
    """

    def setUp(self):
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
                "optimizer_type": Categorical("optimizer_type", choices=["adam", "adamw", "sgd"], shape=(1,))
            },
            seed=0,
        )
        self.pop = self.cs.sample_configuration(10)
        self.fitness = [-92, -17, 22, 56, -96, -20, 76, 29, -48, -56]
        self.optimizer_min = "min"
        self.optimizer_max = "max"
        self.mutator = ChoiceMutation(prob=1.0)

    def test_mutate(self):
        """
        Tests the method mutate().
        """
        new_pop = self.mutator.mutate(self.random, self.cs, self.pop, self.fitness, self.optimizer_min)

        self.assertEqual(len(self.pop), len(new_pop))
        self.assertNotEqual(self.pop, new_pop)


if __name__ == '__main__':
    unittest.main()
