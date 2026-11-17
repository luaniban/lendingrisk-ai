"""Testes das análises pós-teste e da variante 'ausente como categoria' (dados SINTÉTICOS)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import pos_teste
from src.config import FEATURES, MISSING_CATEGORY
from src.evaluation import f_beta, threshold_curve
from src.training import build_pipeline, train
from tests.sintetico import write_synthetic_csv


def _scores(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.15
    p = np.clip(0.12 + 0.18 * y + rng.normal(0, 0.08, n), 0.001, 0.999)
    return y.astype(int), p


class ThresholdOptionsTests(unittest.TestCase):
    def test_tres_criterios_definidos_na_validacao(self):
        y, p = _scores()
        options = pos_teste.threshold_options(y, p, official_threshold=0.09)
        self.assertEqual([o["criterio"] for o in options][0], "Máximo F2 (oficial)")
        self.assertEqual(options[0]["limiar"], 0.09)
        curve = threshold_curve(y, p)
        f1 = [f_beta(pr, rc, beta=1.0) for pr, rc in zip(curve["precision"], curve["recall"])]
        self.assertAlmostEqual(options[1]["limiar"], float(curve["limiar"][int(np.argmax(f1))]))
        self.assertAlmostEqual(float((p >= options[2]["limiar"]).mean()), pos_teste.FLAG_SHARE, delta=0.01)

    def test_tabela_tem_validacao_e_teste_e_matriz_completa(self):
        y_val, p_val = _scores(seed=1)
        y_test, p_test = _scores(seed=2)
        options = pos_teste.threshold_options(y_val, p_val, 0.09)
        table = pos_teste.sensitivity_table(y_val, p_val, y_test, p_test, options)
        self.assertEqual(len(table), 6)
        self.assertEqual(set(table["particao"]), {"validação", "teste"})
        totals = table[["verdadeiros_positivos", "falsos_positivos", "falsos_negativos",
                        "verdadeiros_negativos"]].sum(axis=1)
        self.assertTrue((totals == len(y_val)).all())
        # limiar maior => sinaliza menos
        test_rows = table[table["particao"] == "teste"].sort_values("limiar")
        self.assertTrue(test_rows["taxa_sinalizada"].is_monotonic_decreasing)


class MissingCategoryTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "loan_amnt": [1e4, 2e4, 5e3, 8e3], "annual_inc": [5e4, 6e4, 4e4, 7e4], "dti": [10, 20, 30, 15],
            "fico_range_low": [700, 680, 720, 690], "inq_last_6mths": [0, 1, 2, 0], "revol_util": [40, 50, 60, 30],
            "term": ["36 months"] * 4, "purpose": ["credit_card"] * 4, "home_ownership": ["RENT"] * 4,
            "emp_length": ["10+ years", np.nan, "10+ years", "2 years"],
        })[FEATURES]
        self.y = [0, 1, 0, 1]

    def _categories(self, missing_as_category):
        pipeline = build_pipeline(LogisticRegression(), missing_as_category).fit(self.frame, self.y)
        encoder = pipeline.named_steps["prep"].named_transformers_["cat"].named_steps["one_hot"]
        return [str(c) for c in encoder.categories_[-1]]

    def test_oficial_imputa_a_moda(self):
        self.assertNotIn(MISSING_CATEGORY, self._categories(False))

    def test_variante_cria_categoria_propria(self):
        self.assertIn(MISSING_CATEGORY, self._categories(True))


class MissingSubgroupTests(unittest.TestCase):
    def test_le_o_grupo_ausente_da_tabela_de_subgrupos(self):
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame([
                {"atributo": "emp_length", "valor": "10+ years", "linhas": 500, "taxa_charged_off": 0.13,
                 "score_medio": 0.12, "recall": 0.8, "precision": 0.2},
                {"atributo": "emp_length", "valor": "ausente", "linhas": 300, "taxa_charged_off": 0.22,
                 "score_medio": 0.16, "recall": 0.9, "precision": 0.25},
            ]).to_csv(Path(tmp) / "analise_subgrupos_teste.csv", index=False)
            group = pos_teste._missing_subgroup(Path(tmp))
        self.assertEqual(group["linhas"], 300)
        self.assertAlmostEqual(group["taxa_charged_off_real"], 0.22)
        self.assertAlmostEqual(group["score_medio_previsto"], 0.16)


class EndToEndTests(unittest.TestCase):
    """Treina oficial e variante em dados sintéticos e roda as duas análises."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.csv = write_synthetic_csv(root / "sintetico.csv", n=3000, seed=21)
        common = dict(fraction=1.0, max_rows=None, maturity_filter=False, make_figures=False,
                      importance_rows=200, importance_repeats=1, verbose=False)
        cls.official_dir, cls.variant_dir = root / "reports", root / "reports" / "v2"
        cls.model_path = root / "modelo.joblib"
        cls.official = train(cls.csv, model_path=cls.model_path, report_dir=cls.official_dir, **common)
        cls.variant = train(cls.csv, model_path=root / "modelo_v2.joblib", report_dir=cls.variant_dir,
                            missing_as_category=True, **common)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_variante_registrada_nos_metadados(self):
        self.assertFalse(self.official["variante"]["ausente_como_categoria"])
        self.assertTrue(self.variant["variante"]["ausente_como_categoria"])
        self.assertIn(MISSING_CATEGORY, self.variant["criterios"]["categoricos_ausentes"])

    def test_sensibilidade_reproduz_o_oficial(self):
        out = Path(self.tmp.name) / "analises"
        payload = pos_teste.run_threshold_analysis(self.csv, out_dir=out, metrics_path=self.official_dir / "metricas.json",
                                                   model_path=self.model_path, verbose=False)
        self.assertTrue(all(payload["conferencias"].values()))
        self.assertTrue((out / "sensibilidade_limiar.csv").is_file())
        self.assertEqual(payload["criterios"][0]["limiar"], self.official["limiar"])

    def test_dados_diferentes_sao_recusados(self):
        other = write_synthetic_csv(Path(self.tmp.name) / "outro.csv", n=3000, seed=99)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            pos_teste.run_threshold_analysis(other, out_dir=Path(self.tmp.name) / "x",
                                             metrics_path=self.official_dir / "metricas.json",
                                             model_path=self.model_path, verbose=False)

    def test_comparacao_v1_v2(self):
        out = Path(self.tmp.name) / "comparacao"
        payload = pos_teste.compare_variants(self.official_dir, self.variant_dir, out)
        versions = [row["versao"] for row in payload["comparacao"]]
        self.assertEqual(versions, ["v1_oficial_moda", "v2_ausente_como_categoria"])
        self.assertIn("teste_average_precision", payload["comparacao"][0])
        saved = json.loads((out / "comparacao_v1_v2.json").read_text(encoding="utf-8"))
        self.assertIn("DEPOIS", saved["aviso"])


if __name__ == "__main__":
    unittest.main()
