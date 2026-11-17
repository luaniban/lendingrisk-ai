"""Testes de leitura, limpeza, amostragem, maturidade e divisão temporal (dados SINTÉTICOS)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATE_COLUMN, FEATURES, LEAKAGE_COLUMNS
from src.data import (
    check_columns, load_dataset, maturity_mask, parse_issue_date, parse_numeric,
    temporal_split, term_months,
)
from tests.sintetico import make_synthetic_frame, write_synthetic_csv


class ParsingTests(unittest.TestCase):
    def test_issue_date_independe_de_locale_e_aceita_iso(self):
        values = pd.Series(["Dec-2015", "jan-2012", "2016-03-01", "2017-07", "Dez-2015", "", None, "13-2015"])
        parsed = parse_issue_date(values)
        self.assertEqual(parsed[0], pd.Timestamp("2015-12-01"))
        self.assertEqual(parsed[1], pd.Timestamp("2012-01-01"))
        self.assertEqual(parsed[2], pd.Timestamp("2016-03-01"))
        self.assertEqual(parsed[3], pd.Timestamp("2017-07-01"))
        self.assertTrue(parsed[4:].isna().all())  # "Dez" (pt) e inválidos viram NaT

    def test_numericos_com_virgula_e_porcentagem(self):
        parsed = parse_numeric(pd.Series(["1,000.5", "45.3%", " 12 ", "abc", None]))
        self.assertEqual(parsed[0], 1000.5)
        self.assertEqual(parsed[1], 45.3)
        self.assertEqual(parsed[2], 12.0)
        self.assertTrue(np.isnan(parsed[3]) and np.isnan(parsed[4]))

    def test_term_months(self):
        self.assertEqual(list(term_months(pd.Series([" 36 months", "60 months"]))), [36.0, 60.0])
        self.assertTrue(np.isnan(term_months(pd.Series(["sem prazo"]))[0]))

    def test_maturity_mask(self):
        dates = pd.Series(pd.to_datetime(["2015-12-01", "2016-01-01", "2013-12-01", "2014-01-01"]))
        terms = pd.Series(["36 months", "36 months", "60 months", "60 months"])
        mask = maturity_mask(dates, terms, pd.Timestamp("2018-12-01"))
        self.assertEqual(mask.tolist(), [True, False, True, False])


class LeakageTests(unittest.TestCase):
    def test_features_sem_vazamento(self):
        self.assertFalse(set(FEATURES) & LEAKAGE_COLUMNS)
        self.assertNotIn(DATE_COLUMN, FEATURES)


class LoadDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_arquivo_inexistente(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset(self.dir / "nao_existe.csv")

    def test_colunas_faltando(self):
        path = self.dir / "incompleto.csv"
        make_synthetic_frame(50).drop(columns=["dti"]).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "dti"):
            check_columns(path)

    def test_filtra_status_e_converte_tipos(self):
        path = write_synthetic_csv(self.dir / "s.csv", n=2000)
        data, info = load_dataset(path, sample_fraction=1.0, max_rows=None, maturity_filter=False)
        self.assertEqual(set(data["loan_status"]), {"Fully Paid", "Charged Off"})
        self.assertEqual(set(data["target"]), {0, 1})
        self.assertNotIn("total_pymnt", data.columns)
        self.assertNotIn("last_fico_range_high", data.columns)
        self.assertTrue(pd.api.types.is_float_dtype(data["revol_util"]))
        self.assertEqual(set(data["term"].dropna()), {"36 months", "60 months"})
        self.assertTrue(data[DATE_COLUMN].is_monotonic_increasing)
        self.assertEqual(info["linhas_lidas"], 2002)  # inclui 2 linhas de rodapé
        self.assertEqual(info["linhas_utilizadas"], info["linhas_com_desfecho_final"])
        self.assertIn("Current", info["contagem_status"])

    def test_datas_invalidas_sao_contadas(self):
        frame = make_synthetic_frame(400)
        frame["loan_status"] = "Fully Paid"
        frame.loc[:199, "loan_status"] = "Charged Off"
        frame.loc[:9, "issue_d"] = "data-ruim"
        path = self.dir / "datas.csv"
        frame.to_csv(path, index=False)
        data, info = load_dataset(path, sample_fraction=1.0, max_rows=None, maturity_filter=False)
        self.assertEqual(info["datas_invalidas_entre_finalizados"], 10)
        self.assertEqual(len(data), 390)

    def test_gz_igual_ao_csv(self):
        csv = write_synthetic_csv(self.dir / "s.csv", n=1500, seed=3)
        gz = write_synthetic_csv(self.dir / "s.csv.gz", n=1500, seed=3)
        a, _ = load_dataset(csv, sample_fraction=0.5, max_rows=None, maturity_filter=False)
        b, _ = load_dataset(gz, sample_fraction=0.5, max_rows=None, maturity_filter=False)
        pd.testing.assert_frame_equal(a, b)

    def test_amostragem_uniforme_reprodutivel_e_independente_do_bloco(self):
        path = write_synthetic_csv(self.dir / "s.csv", n=6000, seed=1)
        full, _ = load_dataset(path, sample_fraction=1.0, max_rows=None, maturity_filter=False)
        a, _ = load_dataset(path, sample_fraction=0.3, max_rows=None, maturity_filter=False, chunksize=500)
        b, _ = load_dataset(path, sample_fraction=0.3, max_rows=None, maturity_filter=False, chunksize=500)
        pd.testing.assert_frame_equal(a, b)
        self.assertAlmostEqual(len(a) / len(full), 0.3, delta=0.03)
        # sem piso por bloco: blocos pequenos não são super-representados
        small_chunks, _ = load_dataset(path, sample_fraction=0.3, max_rows=None, maturity_filter=False, chunksize=50)
        self.assertAlmostEqual(len(small_chunks) / len(full), 0.3, delta=0.03)

    def test_teto_de_linhas(self):
        path = write_synthetic_csv(self.dir / "s.csv", n=3000)
        data, info = load_dataset(path, sample_fraction=1.0, max_rows=500, maturity_filter=False)
        self.assertEqual(len(data), 500)
        self.assertTrue(info["amostragem"]["teto_aplicado"])

    def test_filtro_de_maturidade(self):
        path = write_synthetic_csv(self.dir / "s.csv", n=3000, start_year=2010, months=72)
        data, info = load_dataset(path, sample_fraction=1.0, max_rows=None, maturity_filter=True)
        reference = pd.Timestamp(info["filtro_maturidade"]["data_referencia"])
        self.assertEqual(reference, pd.Timestamp("2015-12-01"))
        end_month = data[DATE_COLUMN].dt.year * 12 + data[DATE_COLUMN].dt.month - 1 + term_months(data["term"])
        self.assertTrue((end_month <= reference.year * 12 + reference.month - 1).all())
        self.assertGreater(info["filtro_maturidade"]["linhas_removidas"], 0)

    def test_fracao_invalida(self):
        path = write_synthetic_csv(self.dir / "s.csv", n=300)
        with self.assertRaises(ValueError):
            load_dataset(path, sample_fraction=0)


class TemporalSplitTests(unittest.TestCase):
    def _frame(self, n=1000, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.to_datetime("2012-01-01") + pd.to_timedelta(rng.integers(0, 48, n) * 31, unit="D")
        dates = dates.to_period("M").to_timestamp()
        return pd.DataFrame({DATE_COLUMN: dates, "target": rng.integers(0, 2, n),
                             "term": "36 months"}).sort_values(DATE_COLUMN).reset_index(drop=True)

    def test_particoes_disjuntas_e_ordenadas(self):
        train, val, test = temporal_split(self._frame())
        self.assertLess(train[DATE_COLUMN].max(), val[DATE_COLUMN].min())
        self.assertLess(val[DATE_COLUMN].max(), test[DATE_COLUMN].min())
        self.assertEqual(len(train) + len(val) + len(test), 1000)

    def test_ordena_se_necessario(self):
        frame = self._frame().sample(frac=1, random_state=1)
        train, val, test = temporal_split(frame)
        self.assertLess(train[DATE_COLUMN].max(), test[DATE_COLUMN].min())

    def test_erro_quando_falta_classe(self):
        frame = self._frame()
        frame.loc[frame.index[-200:], "target"] = 0
        with self.assertRaisesRegex(ValueError, "teste"):
            temporal_split(frame)

    def test_erro_quando_um_unico_mes(self):
        frame = self._frame()
        frame[DATE_COLUMN] = pd.Timestamp("2015-01-01")
        with self.assertRaises(ValueError):
            temporal_split(frame)


if __name__ == "__main__":
    unittest.main()
