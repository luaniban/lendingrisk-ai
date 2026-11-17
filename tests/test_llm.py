"""Testes da integração com o Ollama usando respostas simuladas (sem rede)."""
import unittest
from unittest import mock

import requests

from src import llm

INPUTS = {"Valor solicitado (USD)": "US$ 10.000", "Uso do crédito rotativo (%)": "45,0%"}


def _response(status=200, payload=None, text=""):
    response = mock.Mock()
    response.status_code = status
    response.text = text
    response.json.return_value = payload if payload is not None else {}
    return response


class PromptTests(unittest.TestCase):
    def test_prompt_traz_valores_do_classificador(self):
        prompt = llm.build_prompt(0.234, 0.18, INPUTS, ["Score FICO"], "XGBoost")
        self.assertIn("23,4%", prompt)
        self.assertIn("18,0%", prompt)
        self.assertIn("ACIMA do limiar", prompt)
        self.assertIn("US$ 10.000", prompt)
        self.assertIn("NÃO deste caso", prompt)
        self.assertIn("XGBoost", prompt)

    def test_classe_abaixo(self):
        self.assertIn("ABAIXO", llm.build_prompt(0.10, 0.18, INPUTS, []))

    def test_system_prompt_restritivo(self):
        self.assertIn("Não calcule", llm.SYSTEM_PROMPT)
        self.assertIn("Não recomende aprovar", llm.SYSTEM_PROMPT)


class OllamaCallTests(unittest.TestCase):
    @mock.patch("src.llm.requests.post")
    def test_sucesso_envia_opcoes_seguras(self, post):
        post.return_value = _response(payload={"response": "  Texto gerado.  "})
        text = llm.explain_with_ollama(0.2, 0.18, INPUTS, ["Prazo"])
        self.assertEqual(text, "Texto gerado.")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["options"]["temperature"], 0.1)
        self.assertEqual(body["options"]["num_predict"], llm.MAX_TOKENS)
        self.assertFalse(body["stream"])
        self.assertIn("system", body)
        self.assertEqual(post.call_args.kwargs["timeout"], (llm.CONNECT_TIMEOUT_S, llm.READ_TIMEOUT_S))
        self.assertTrue(post.call_args.args[0].endswith("/api/generate"))

    @mock.patch("src.llm.requests.post", side_effect=requests.exceptions.ConnectionError("recusada"))
    def test_ollama_fora_do_ar(self, _):
        with self.assertRaisesRegex(llm.LLMIndisponivel, "Não foi possível conectar"):
            llm.explain_with_ollama(0.2, 0.18, INPUTS, [])

    @mock.patch("src.llm.requests.post", side_effect=requests.exceptions.ReadTimeout("lento"))
    def test_timeout(self, _):
        with self.assertRaisesRegex(llm.LLMIndisponivel, "não respondeu"):
            llm.explain_with_ollama(0.2, 0.18, INPUTS, [])

    @mock.patch("src.llm.requests.post")
    def test_modelo_nao_baixado(self, post):
        post.return_value = _response(status=404, text='{"error":"model not found"}')
        with self.assertRaisesRegex(llm.LLMIndisponivel, "ollama pull"):
            llm.explain_with_ollama(0.2, 0.18, INPUTS, [])

    @mock.patch("src.llm.requests.post")
    def test_erro_http_e_resposta_vazia_e_json_invalido(self, post):
        post.return_value = _response(status=500, text="falha interna")
        with self.assertRaisesRegex(llm.LLMIndisponivel, "HTTP 500"):
            llm.explain_with_ollama(0.2, 0.18, INPUTS, [])
        post.return_value = _response(payload={"response": "   "})
        with self.assertRaisesRegex(llm.LLMIndisponivel, "vazia"):
            llm.explain_with_ollama(0.2, 0.18, INPUTS, [])
        bad = _response()
        bad.json.side_effect = ValueError("não é json")
        post.return_value = bad
        with self.assertRaisesRegex(llm.LLMIndisponivel, "JSON"):
            llm.explain_with_ollama(0.2, 0.18, INPUTS, [])

    @mock.patch("src.llm.requests.get")
    def test_check_ollama(self, get):
        get.return_value = _response(payload={"models": [{"name": "llama3.2:3b"}]})
        self.assertTrue(llm.check_ollama(model="llama3.2:3b")["ok"])
        get.return_value = _response(payload={"models": [{"name": "outro:1b"}]})
        status = llm.check_ollama(model="llama3.2:3b")
        self.assertFalse(status["ok"])
        self.assertIn("ollama pull", status["mensagem"])
        get.side_effect = requests.exceptions.ConnectionError()
        self.assertFalse(llm.check_ollama()["ok"])


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.prompt = llm.build_prompt(0.234, 0.18, INPUTS, [])

    def test_texto_coerente_sem_avisos(self):
        text = "A probabilidade estimada é de 23,4%, acima do limiar de 18,0%. " + llm.CLOSING_SENTENCE
        self.assertEqual(llm.check_consistency(text, 0.234, 0.18, self.prompt), [])

    def test_percentual_inventado(self):
        text = "A probabilidade é 23,4%, e 70% dos clientes assim pagam."
        warnings = llm.check_consistency(text, 0.234, 0.18, self.prompt)
        self.assertTrue(any("70%" in w for w in warnings))

    def test_probabilidade_omitida_ou_alterada(self):
        warnings = llm.check_consistency("O risco é de 30%.", 0.234, 0.18, self.prompt)
        self.assertTrue(any("não cita a probabilidade" in w for w in warnings))

    def test_classe_contraditoria(self):
        text = "Com 23,4%, o caso ficou abaixo do limiar."
        warnings = llm.check_consistency(text, 0.234, 0.18, self.prompt)
        self.assertTrue(any("ACIMA" in w for w in warnings))
        text = "Com 10,0%, o caso ficou acima do limiar."
        warnings = llm.check_consistency(text, 0.10, 0.18, llm.build_prompt(0.10, 0.18, INPUTS, []))
        self.assertTrue(any("ABAIXO" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
