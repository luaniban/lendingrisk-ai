import unittest
import numpy as np
from src.config import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, CLASS_MAP
from src.training import choose_threshold, scores

class Basics(unittest.TestCase):
    def test_features_are_unique(self):
        self.assertEqual(len(FEATURES), len(set(FEATURES)))
        self.assertEqual(set(FEATURES), set(NUMERIC_FEATURES + CATEGORICAL_FEATURES))

    def test_labels(self):
        self.assertEqual(CLASS_MAP["Fully Paid"], 0)
        self.assertEqual(CLASS_MAP["Charged Off"], 1)

    def test_metrics(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(scores(y, p, 0.5)["pr_auc"], 1.0)
        self.assertBetween(choose_threshold(y, p), 0.1, 0.9)

    def assertBetween(self, number, minimum, maximum):
        self.assertTrue(minimum <= number <= maximum)

if __name__ == "__main__":
    unittest.main()
