import unittest

import numpy as np
from PyHyperparameterSpace.hp.constant import Constant
from PyHyperparameterSpace.hp.continuous import Float
from PyHyperparameterSpace.space import HyperparameterConfigurationSpace

from PyEvo.fitness.z_score_normalizer import FitnessZScoreNormalizer


class TestFitnessZScoreNormalizer(unittest.TestCase):
    """
    Tests the class FitnessZScoreNormalizer.
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
        self.fitness_preprocessor = FitnessZScoreNormalizer()

    def test_preprocess_fitness_minimizer(self):
        """
        Tests the method preprocess_fitness() with minimizer.
        """
        preprocessed_fitness = self.fitness_preprocessor.preprocess_fitness(self.random, self.cs, self.pop,
                                                                            self.fitness, self.optimizer_min)
        normalized = [(f - np.mean(self.fitness)) / np.std(self.fitness) for f in self.fitness]

        self.assertEqual(len(self.fitness), len(preprocessed_fitness))
        self.assertEqual(normalized, preprocessed_fitness)

    def test_preprocess_fitness_maximizer(self):
        """
        Tests the method preprocess_fitness() with maximizer.
        """
        preprocessed_fitness = self.fitness_preprocessor.preprocess_fitness(self.random, self.cs, self.pop,
                                                                            self.fitness, self.optimizer_max)
        normalized = [(f - np.mean(self.fitness)) / np.std(self.fitness) for f in self.fitness]

        self.assertEqual(len(self.fitness), len(preprocessed_fitness))
        self.assertEqual(normalized, preprocessed_fitness)


if __name__ == '__main__':
    unittest.main()
