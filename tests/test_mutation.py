import unittest

from PyEvo.mutation import GaussianMutation


class TestGaussianMutation(unittest.TestCase):
    """
    Tests the class GaussianMutation.
    """

    def setUp(self):
        self.gaussian = GaussianMutation(mean=0.0, std=1.0, mut_prob=1.0)

    def test_mutate(self):
        """
        Tests the method mutate().
        """
        pass


if __name__ == '__main__':
    unittest.main()
