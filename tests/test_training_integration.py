"""Teste de integração ponta a ponta do treinamento com dados SINTÉTICOS.

Os números produzidos aqui NÃO têm valor científico: o objetivo é verificar
que o pipeline roda, respeita as regras metodológicas e grava artefatos
coerentes. Tudo é escrito em diretório temporário.
"""
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURES
from src.data import load_dataset, temporal_split
from src.training import main as training_main
from src.training import train
from tests.sintetico import write_synthetic_csv


class TrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.csv = write_synthetic_csv(root / "sintetico.csv", n=4000, seed=7)
        cls.model_path = root / "models" / "modelo.joblib"
        cls.report_dir = root / "reports"
        cls.payload = train(cls.csv, fraction=1.0, max_rows=None, maturity_filter=False,
                            model_path=cls.model_path, report_dir=cls.report_dir,
                            importance_rows=400, importance_repeats=2, verbose=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_artefatos_gerados(self):
        expected = [
            "metricas.json", "comparacao_modelos.csv", "importancia_atributos.csv",
            "matriz_confusao_teste.csv", "curva_limiar_validacao.csv", "calibracao_validacao.csv",
            "calibracao_teste.csv", "estabilidade_temporal.csv", "analise_subgrupos_teste.csv",
            "historico_execucoes.csv",
        ]
        for name in expected:
            self.assertTrue((self.report_dir / name).is_file(), name)
        self.assertTrue(self.model_path.is_file())
        for figure in self.payload["figuras"]:
            self.assertTrue((self.report_dir / figure).is_file(), figure)
        self.assertGreaterEqual(len(self.payload["figuras"]), 9)

    def test_tres_modelos_e_selecao_pela_validacao(self):
        comparison = self.payload["comparacao_validacao"]
        self.assertEqual([r["modelo"] for r in comparison], ["Regressão Logística", "Random Forest", "XGBoost"])
        best = max(comparison, key=lambda r: r["average_precision"])
        self.assertEqual(self.payload["modelo_escolhido"], best["modelo"])
        self.assertEqual(self.payload["limiar"], best["limiar"])

    def test_metricas_json_coerente(self):
        metrics = json.loads((self.report_dir / "metricas.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["atributos"], FEATURES)
        self.assertIn("average_precision", metrics["metricas_teste"])
        self.assertNotIn("pr_auc", metrics["metricas_teste"])
        matrix = np.array(metrics["matriz_confusao_teste"])
        self.assertEqual(matrix.sum(), metrics["linhas"]["teste"])
        self.assertEqual(sum(metrics["linhas"][k] for k in ("treino", "validacao", "teste")), metrics["linhas"]["total"])
        self.assertIn("sha256", metrics["dataset"])
        self.assertIn("scikit_learn", metrics["ambiente"])
        self.assertEqual(set(metrics["esquema_entrada"]["numericos"]), set(metrics["atributos_numericos"]))
        # recall do teste é coerente com a matriz de confusão
        tp, fn = matrix[1, 1], matrix[1, 0]
        self.assertAlmostEqual(metrics["metricas_teste"]["recall"], round(tp / (tp + fn), 5), places=4)

    def test_preprocessamento_ajustado_somente_no_treino(self):
        data, _ = load_dataset(self.csv, sample_fraction=1.0, max_rows=None, maturity_filter=False)
        train_set, val_set, _ = temporal_split(data)
        model = joblib.load(self.model_path)
        imputer = model.named_steps["prep"].named_transformers_["num"].named_steps["imputer"]
        columns = list(imputer.feature_names_in_)
        np.testing.assert_allclose(imputer.statistics_, train_set[columns].median().to_numpy())
        scaler = model.named_steps["prep"].named_transformers_["num"].named_steps["scaler"]
        train_imputed = train_set[columns].fillna(train_set[columns].median())
        np.testing.assert_allclose(scaler.mean_, train_imputed.mean().to_numpy())
        # a média incluindo validação+teste é diferente: prova de que eles não entraram no ajuste
        all_imputed = data[columns].fillna(train_set[columns].median())
        self.assertFalse(np.allclose(scaler.mean_, all_imputed.mean().to_numpy()))

    def test_modelo_salvo_preve_probabilidades(self):
        model = joblib.load(self.model_path)
        data, _ = load_dataset(self.csv, sample_fraction=1.0, max_rows=None, maturity_filter=False)
        p = model.predict_proba(data[FEATURES].head(20))[:, 1]
        self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_historico_acumula_execucoes(self):
        history = pd.read_csv(self.report_dir / "historico_execucoes.csv")
        self.assertGreaterEqual(len(history), 1)
        self.assertIn("ap_teste", history.columns)


class TrainingCliTests(unittest.TestCase):
    def test_argumentos_invalidos(self):
        with self.assertRaises(SystemExit):
            training_main(["--data", "x.csv", "--fraction", "1.5"])
        with self.assertRaises(SystemExit):
            training_main(["--data", "x.csv", "--max-rows", "-1"])

    def test_arquivo_inexistente(self):
        with self.assertRaises(FileNotFoundError):
            training_main(["--data", "nao_existe.csv"])


if __name__ == "__main__":
    unittest.main()
