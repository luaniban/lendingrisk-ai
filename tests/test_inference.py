"""Testes de carregamento de artefatos e inferência (modelo treinado com dados SINTÉTICOS)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config import FEATURE_LABELS, FEATURES
from src.inference import (
    MISSING_LABEL, ArtefatosInvalidos, format_record_pt, load_metrics, load_model,
    percent_pt, predict_probability, validate_record, version_warnings,
)
from src.training import train
from tests.sintetico import write_synthetic_csv

EXAMPLE = {
    "loan_amnt": 10_000.0, "annual_inc": 60_000.0, "dti": 18.0, "fico_range_low": 700,
    "inq_last_6mths": 1, "revol_util": 45.0, "term": "36 months",
    "purpose": "debt_consolidation", "home_ownership": "RENT", "emp_length": "5 years",
}


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        csv = write_synthetic_csv(root / "sintetico.csv", n=2500, seed=11)
        cls.model_path = root / "modelo.joblib"
        cls.report_dir = root / "reports"
        train(csv, fraction=1.0, max_rows=None, maturity_filter=False, model_path=cls.model_path,
              report_dir=cls.report_dir, make_figures=False, importance_rows=200,
              importance_repeats=1, verbose=False)
        cls.metrics_path = cls.report_dir / "metricas.json"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_carrega_artefatos_validos(self):
        metrics = load_metrics(self.metrics_path)
        model = load_model(self.model_path)
        self.assertEqual(version_warnings(metrics), [])
        record, warnings = validate_record(EXAMPLE, metrics["esquema_entrada"])
        probability = predict_probability(model, record)
        self.assertTrue(0 <= probability <= 1)

    def test_categoria_desconhecida_gera_aviso_sem_quebrar(self):
        metrics = load_metrics(self.metrics_path)
        model = load_model(self.model_path)
        record, warnings = validate_record({**EXAMPLE, "purpose": "wedding"}, metrics["esquema_entrada"])
        self.assertTrue(any("não apareceu no treino" in w for w in warnings))
        self.assertTrue(0 <= predict_probability(model, record) <= 1)

    def test_valor_fora_da_faixa_gera_aviso(self):
        metrics = load_metrics(self.metrics_path)
        _, warnings = validate_record({**EXAMPLE, "annual_inc": 9_000_000}, metrics["esquema_entrada"])
        self.assertTrue(any("fora da faixa" in w and "9.000.000" in w for w in warnings))

    def test_valores_invalidos(self):
        for field, value in [("dti", -1), ("loan_amnt", "abc"), ("annual_inc", float("nan"))]:
            with self.assertRaises(ValueError, msg=field):
                validate_record({**EXAMPLE, field: value})
        with self.assertRaises(ValueError):
            validate_record({k: v for k, v in EXAMPLE.items() if k != "term"})

    def test_ausente_vira_nan_e_modelo_imputa(self):
        model = load_model(self.model_path)
        record, _ = validate_record({**EXAMPLE, "emp_length": MISSING_LABEL})
        self.assertTrue(np.isnan(record["emp_length"]))
        self.assertTrue(0 <= predict_probability(model, record) <= 1)

    def test_metricas_ausentes_ou_corrompidas(self):
        root = Path(self.tmp.name)
        with self.assertRaises(ArtefatosInvalidos):
            load_metrics(root / "nao_existe.json")
        broken = root / "quebrado.json"
        broken.write_text("{ isso não é json", encoding="utf-8")
        with self.assertRaisesRegex(ArtefatosInvalidos, "corrompido"):
            load_metrics(broken)
        metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        wrong = root / "atributos_errados.json"
        wrong.write_text(json.dumps({**metrics, "atributos": FEATURES[:-1]}), encoding="utf-8")
        with self.assertRaisesRegex(ArtefatosInvalidos, "atributos diferentes"):
            load_metrics(wrong)
        bad_threshold = root / "limiar.json"
        bad_threshold.write_text(json.dumps({**metrics, "limiar": 1.5}), encoding="utf-8")
        with self.assertRaisesRegex(ArtefatosInvalidos, "Limiar"):
            load_metrics(bad_threshold)

    def test_modelo_ausente_ou_invalido(self):
        root = Path(self.tmp.name)
        with self.assertRaises(ArtefatosInvalidos):
            load_model(root / "nao_existe.joblib")
        broken = root / "quebrado.joblib"
        broken.write_bytes(b"lixo")
        with self.assertRaises(ArtefatosInvalidos):
            load_model(broken)

    def test_aviso_de_versao(self):
        metrics = load_metrics(self.metrics_path)
        metrics["ambiente"]["scikit_learn"] = "0.0.1"
        self.assertTrue(any("scikit_learn" in w for w in version_warnings(metrics)))

    def test_formatacao_pt(self):
        text = format_record_pt({**EXAMPLE, "emp_length": np.nan})
        self.assertEqual(text[FEATURE_LABELS["loan_amnt"]], "US$ 10.000")
        self.assertEqual(text[FEATURE_LABELS["revol_util"]], "45,0%")
        self.assertEqual(text[FEATURE_LABELS["term"]], "36 meses")
        self.assertEqual(text[FEATURE_LABELS["emp_length"]], MISSING_LABEL)
        self.assertEqual(percent_pt(0.2345), "23,4%")


if __name__ == "__main__":
    unittest.main()
