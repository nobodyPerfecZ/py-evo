import unittest
import numpy as np

from PyHyperparameterSpace.configuration import HyperparameterConfiguration
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace
from PyHyperparameterSpace.hp.continuous import Float
from PyHyperparameterSpace.hp.categorical import Categorical
from PyHyperparameterSpace.hp.constant import Constant

from PyEvo.mutation import GaussianMutation, UniformMutation, ChoiceMutation


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
        self.mutator = GaussianMutation(loc=0.0, scale=1.0, prob=1.0)

    def test_mutate(self):
        """
        Tests the method mutate().
        """
        new_pop = self.mutator.mutate(self.random, self.cs, self.pop)

        self.assertEqual(len(self.pop), len(new_pop))
        self.assertNotEqual(self.pop, new_pop)


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
        self.mutator_float = UniformMutation(low=-0.5, high=0.5, hp_type="float", prob=1.0)
        self.mutator_int = UniformMutation(low=-1, high=2, hp_type="int", prob=1.0)

    def test_mutate_float(self):
        """
        Tests the method mutate() for float hyperparameters.
        """
        new_pop = self.mutator_float.mutate(self.random, self.cs, self.pop)

        self.assertEqual(len(self.pop), len(new_pop))
        self.assertNotEqual(self.pop, new_pop)

    def test_mutate_integer(self):
        """
        Tests the method mutate() for integer hyperparameters.
        """
        new_pop = self.mutator_int.mutate(self.random, self.cs, self.pop)

        self.assertEqual(len(self.pop), len(new_pop))
        self.assertEqual(self.pop, new_pop)


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
        self.mutator = ChoiceMutation(prob=1.0)

    def test_mutate(self):
        """
        Tests the method mutate().
        """
        new_pop = self.mutator.mutate(self.random, self.cs, self.pop)

        self.assertEqual(len(self.pop), len(new_pop))
        self.assertNotEqual(self.pop, new_pop)


if __name__ == '__main__':
    unittest.main()
