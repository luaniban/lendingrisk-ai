import unittest

import numpy as np

from src.config import CATEGORICAL_FEATURES, CLASS_MAP, FEATURES, NUMERIC_FEATURES
from src.evaluation import (
    calibration_table, expected_calibration_error, f_beta, metrics_by_group,
    reference_scores, threshold_curve,
)
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
        result = scores(y, p, 0.5)
        self.assertAlmostEqual(result["average_precision"], 1.0)
        self.assertNotIn("pr_auc", result)  # nomenclatura correta: Average Precision
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["acuracia"], 1.0)
        self.assertBetween(choose_threshold(y, p), 0.1, 0.9)

    def assertBetween(self, number, minimum, maximum):
        self.assertTrue(minimum <= number <= maximum)


class EvaluationTests(unittest.TestCase):
    def test_f_beta(self):
        self.assertAlmostEqual(f_beta(0.5, 0.5, 2), 0.5)
        self.assertAlmostEqual(f_beta(1.0, 0.5, 1), 2 / 3)
        self.assertEqual(f_beta(0, 0, 2), 0.0)

    def test_scores_uma_classe_nao_quebra(self):
        result = scores(np.zeros(10), np.linspace(0, 0.9, 10), 0.5)
        self.assertIsNone(result["average_precision"])
        self.assertIsNone(result["roc_auc"])
        self.assertIsNotNone(result["brier"])

    def test_limiar_na_grade_e_no_maximo_f2(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 500)
        p = np.clip(0.3 * y + rng.uniform(0, 0.7, 500), 0, 1)
        threshold = choose_threshold(y, p)
        curve = threshold_curve(y, p)
        self.assertAlmostEqual(curve.loc[curve["limiar"] == threshold, "f2"].iloc[0], curve["f2"].max())
        self.assertTrue(0.01 <= threshold <= 0.99)

    def test_calibracao_perfeita_tem_ece_baixo(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0, 1, 20000)
        y = (rng.uniform(0, 1, 20000) < p).astype(int)
        self.assertLess(expected_calibration_error(y, p), 0.02)
        self.assertGreater(expected_calibration_error(y, np.clip(p + 0.3, 0, 1)), 0.2)
        self.assertEqual(calibration_table(y, p)["linhas"].sum(), 20000)

    def test_referencias(self):
        ref = reference_scores([0, 0, 0, 1])
        self.assertEqual(ref["average_precision_aleatorio"], 0.25)
        self.assertAlmostEqual(ref["brier_prevalencia_constante"], 0.1875)

    def test_metricas_por_grupo_ignora_grupos_pequenos(self):
        import pandas as pd
        frame = pd.DataFrame({"g": ["a"] * 150 + ["b"] * 20, "target": [0, 1] * 85,
                              "score": np.linspace(0, 1, 170)})
        table = metrics_by_group(frame, "g", 0.5)
        self.assertEqual(table["valor"].tolist(), ["a"])


if __name__ == "__main__":
    unittest.main()
