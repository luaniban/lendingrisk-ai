"""Teste da interface Streamlit (AppTest) com modelo treinado em dados SINTÉTICOS.

O Ollama é simulado (mock) para o teste não depender de rede nem de hardware.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import streamlit as st
from streamlit.testing.v1 import AppTest

from src import config
from src.training import train
from tests.sintetico import write_synthetic_csv

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _fake_tags(*_, **__):
    response = mock.Mock(status_code=200)
    response.json.return_value = {"models": [{"name": config.LLM_MODEL}]}
    response.raise_for_status.return_value = None
    return response


class AppWithoutModelTests(unittest.TestCase):
    def test_mostra_instrucao_quando_nao_ha_modelo(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(config, "MODEL_PATH", Path(tmp) / "x.joblib"), \
                mock.patch.object(config, "METRICS_PATH", Path(tmp) / "x.json"):
            st.cache_data.clear()
            st.cache_resource.clear()
            at = AppTest.from_file(APP, default_timeout=60).run()
            self.assertFalse(at.exception)
            self.assertTrue(any("Modelo indisponível" in e.value for e in at.error))
            self.assertTrue(any("src.training" in c.value for c in at.code))


class AppWithModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        csv = write_synthetic_csv(root / "sintetico.csv", n=2500, seed=5)
        cls.report_dir = root / "reports"
        cls.model_path = root / "models" / "modelo.joblib"
        train(csv, fraction=1.0, max_rows=None, maturity_filter=False, model_path=cls.model_path,
              report_dir=cls.report_dir, make_figures=True, importance_rows=200, importance_repeats=1,
              verbose=False)
        cls.patches = [
            mock.patch.object(config, "MODEL_PATH", cls.model_path),
            mock.patch.object(config, "METRICS_PATH", cls.report_dir / "metricas.json"),
            mock.patch.object(config, "REPORT_DIR", cls.report_dir),
            mock.patch("src.llm.requests.get", side_effect=_fake_tags),
        ]
        for patch in cls.patches:
            patch.start()

    @classmethod
    def tearDownClass(cls):
        for patch in cls.patches:
            patch.stop()
        cls.tmp.cleanup()

    def setUp(self):
        st.cache_data.clear()
        st.cache_resource.clear()

    def _submit(self, at, **changes):
        for field, value in changes.items():
            widget = at.number_input(key=f"in_{field}") if field in config.NUMERIC_FEATURES \
                else at.selectbox(key=f"in_{field}")
            widget.set_value(value)
        submit = next(b for b in at.button if b.label == "Estimar risco")
        return submit.click().run()

    def test_carrega_e_mostra_abas(self):
        at = AppTest.from_file(APP, default_timeout=60).run()
        self.assertFalse(at.exception)
        self.assertEqual([t.label for t in at.tabs], ["Simular empréstimo", "Desempenho dos modelos", "Dados e metodologia"])
        self.assertTrue(any("Ollama ativo" in s.value for s in at.success))

    def test_previsao_pelo_formulario(self):
        at = AppTest.from_file(APP, default_timeout=60).run()
        at = self._submit(at, loan_amnt=12000.0, dti=25.0)
        self.assertFalse(at.exception)
        labels = [m.label for m in at.metric]
        self.assertIn("Probabilidade estimada de Charged Off", labels)
        value = next(m.value for m in at.metric if m.label == "Probabilidade estimada de Charged Off")
        self.assertTrue(value.endswith("%"))
        self.assertTrue(any("Classe prevista" in x.value for x in list(at.error) + list(at.success)))

    def test_explicacao_llm_e_falha_do_ollama(self):
        at = AppTest.from_file(APP, default_timeout=60).run()
        at = self._submit(at)
        probability = next(m.value for m in at.metric if m.label == "Probabilidade estimada de Charged Off")
        reply = mock.Mock(status_code=200)
        reply.json.return_value = {"response": f"A probabilidade estimada é {probability}. {'x' * 10}"}
        with mock.patch("src.llm.requests.post", return_value=reply):
            button = next(b for b in at.button if "Explicar com LLM" in b.label)
            at = button.click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any(probability in i.value for i in at.info))

        import requests
        with mock.patch("src.llm.requests.post", side_effect=requests.exceptions.ConnectionError()):
            button = next(b for b in at.button if "Explicar com LLM" in b.label)
            at = button.click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any("LLM indisponível" in w.value for w in at.warning))
        # a previsão continua na tela mesmo com a LLM fora do ar
        self.assertIn("Probabilidade estimada de Charged Off", [m.label for m in at.metric])


if __name__ == "__main__":
    unittest.main()
