##Note - this code is ai generated cause I was too lazy to write test cases for simple functions cheers


## gemini wanted to make a very elaborate testing framework so I let it go crazy and learnt something new run this and see something new

import unittest
import math
import functions as fn 

class TestMathFunctions(unittest.TestCase):

    def assertListAlmostEqual(self, list1, list2, places=7):
        """Helper function to compare two lists of floats."""
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, places=places)

    ## Activation Function Tests

    def test_sigmoid(self):
        # Test single float
        self.assertAlmostEqual(fn.sigmoid(0), 0.5)
        self.assertAlmostEqual(fn.sigmoid(1), 1 / (1 + math.exp(-1)))
        self.assertAlmostEqual(fn.sigmoid(-1), 1 / (1 + math.exp(1)))
        
        # Test list of floats
        x_list = [0, 1, -1]
        expected = [0.5, 1 / (1 + math.exp(-1)), 1 / (1 + math.exp(1))]
        self.assertListAlmostEqual(fn.sigmoid(x_list), expected)

    def test_tanh(self):
        # Test single float
        self.assertAlmostEqual(fn.tanh(0), 0.0)
        self.assertAlmostEqual(fn.tanh(1), (math.exp(1) - math.exp(-1)) / (math.exp(1) + math.exp(-1)))
        self.assertAlmostEqual(fn.tanh(-1), (math.exp(-1) - math.exp(1)) / (math.exp(-1) + math.exp(1)))

        # Test list of floats
        x_list = [0, 1, -1]
        expected = [fn.tanh(0), fn.tanh(1), fn.tanh(-1)]
        self.assertListAlmostEqual(fn.tanh(x_list), expected)

    def test_relu(self):
        # Test single float
        self.assertEqual(fn.relu(10), 10)
        self.assertEqual(fn.relu(-10), 0)
        self.assertEqual(fn.relu(0), 0)
        self.assertEqual(fn.relu(3.5), 3.5)
        self.assertEqual(fn.relu(-3.5), 0)

        # Test list of floats
        x_list = [1, -2, 0, 3.5, -1.2]
        expected = [1, 0, 0, 3.5, 0]
        self.assertEqual(fn.relu(x_list), expected)

    def test_softmax(self):
        x_list = [1.0, 2.0, 3.0]
        exp_x = [math.exp(1), math.exp(2), math.exp(3)]
        sum_exp_x = sum(exp_x)
        expected = [exp_x[0]/sum_exp_x, exp_x[1]/sum_exp_x, exp_x[2]/sum_exp_x]
        
        result = fn.softmax(x_list)
        self.assertListAlmostEqual(result, expected)
        # Test that probabilities sum to 1
        self.assertAlmostEqual(sum(result), 1.0)

    ## Vector Operation Tests

    def test_dot_product(self):
        self.assertEqual(fn.dot_product([1, 2, 3], [4, 5, 6]), 32)
        self.assertEqual(fn.dot_product([1, -2, 3], [4, 5, -6]), -24)
        self.assertEqual(fn.dot_product([1, 2, 3], [0, 0, 0]), 0)
        self.assertEqual(fn.dot_product([], []), 0)
    
    def test_mag(self):
        # Test based on the corrected mag function
        self.assertAlmostEqual(fn.mag([3, 4]), 5.0)
        self.assertAlmostEqual(fn.mag([0, 0]), 0.0)
        self.assertAlmostEqual(fn.mag([1, 1, 1]), math.sqrt(3))
        self.assertAlmostEqual(fn.mag([5]), 5.0)

    def test_cosine_similarity(self):
        # Parallel vectors
        self.assertAlmostEqual(fn.cosine_similarity([1, 2], [2, 4]), 1.0)
        # Orthogonal vectors
        self.assertAlmostEqual(fn.cosine_similarity([1, 0], [0, 1]), 0.0)
        # Opposite vectors
        self.assertAlmostEqual(fn.cosine_similarity([1, 2], [-1, -2]), -1.0)
        # Zero vector (based on corrected function)
        self.assertAlmostEqual(fn.cosine_similarity([0, 0], [1, 2]), 0.0)

    ## Normalisation Tests

    def test_L1_norm(self):
        self.assertEqual(fn.L1_norm([1, 2, 3]), 6)
        self.assertEqual(fn.L1_norm([1, -2, 3]), 6)
        self.assertEqual(fn.L1_norm([0, 0]), 0)
        self.assertEqual(fn.L1_norm([-1, -2, -3]), 6)

    def test_min_max_norm(self):
        x_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.assertListAlmostEqual(fn.min_max_norm(x_list), expected)
        
        # Test with all same values
        x_list_same = [5.0, 5.0, 5.0]
        expected_same = [0.0, 0.0, 0.0]
        self.assertListAlmostEqual(fn.min_max_norm(x_list_same), expected_same)

        # Test with empty list
        self.assertEqual(fn.min_max_norm([]), [])

    ## Heuristic Test

    def test_hill_climb(self):
        # Function max is at x=3
        # Test climbing from 0, should stop at 3.0
        # (0.1 * 30 steps = 3.0. Next step 3.1 is lower)
        self.assertAlmostEqual(fn.hill_climb(0.0, 0.1, 100), 3.0)
        
        # Test starting past the max (only climbs in positive direction)
        # Should take 0 steps and return the start
        self.assertAlmostEqual(fn.hill_climb(4.0, 0.1, 100), 4.0)

        # Test max_iter
        # Should only take 10 steps from 0.0
        self.assertAlmostEqual(fn.hill_climb(0.0, 0.1, 10), 1.0)


if __name__ == '__main__':
    unittest.main()
