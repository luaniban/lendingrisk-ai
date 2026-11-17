"""Baixa o Lending Club (empréstimos ACEITOS, 2007–2018Q4) e confere o SHA-256.

Uso (da raiz do projeto):  python -m src.download

Fonte primária citada no trabalho: Kaggle — wordsforthewise/lending-club
(https://www.kaggle.com/datasets/wordsforthewise/lending-club, licença CC0).
O Kaggle exige login para baixar. Por isso, o script usa um espelho público
do MESMO arquivo no Hugging Face (licença CC0), fixado numa revisão, e confere
o SHA-256. O mesmo hash aparece em outros espelhos independentes do arquivo.
Quem preferir o Kaggle pode baixar manualmente o ZIP, extrair o
accepted_2007_to_2018Q4.csv.gz para data/ e treinar direto com o .csv.gz.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests

from src.config import DATA_DIR, DEFAULT_DATA_FILE

MIRROR_URL = (
    "https://huggingface.co/datasets/codesignal/lending-club-loan-accepted/resolve/"
    "c435da047e81f2875a23827c1548f5083a33d178/accepted_2007_to_2018Q4.csv"
)
EXPECTED_SHA256 = "3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a"
EXPECTED_SIZE = 1_675_133_810


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(destination: Path, url: str = MIRROR_URL, attempts: int = 10) -> Path:
    """Download com retomada (HTTP Range); só renomeia para o nome final após conferir o hash."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == EXPECTED_SIZE:
        print(f"Arquivo já existe: {destination}. Conferindo SHA-256...")
        if sha256_of(destination) == EXPECTED_SHA256:
            print("SHA-256 confere.")
            return destination
        raise SystemExit("O arquivo existente tem SHA-256 diferente do esperado. Apague-o e rode de novo.")
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset >= EXPECTED_SIZE:
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(10, 60)) as response:
                if response.status_code not in (200, 206):
                    raise requests.HTTPError(f"HTTP {response.status_code}")
                mode = "ab" if response.status_code == 206 else "wb"
                with open(partial, mode) as handle:
                    written = offset if mode == "ab" else 0
                    for block in response.iter_content(chunk_size=1 << 20):
                        handle.write(block)
                        written += len(block)
                        print(f"\r  {written / 1e6:,.0f} / {EXPECTED_SIZE / 1e6:,.0f} MB", end="", flush=True)
            print()
        except (requests.RequestException, OSError) as exc:
            print(f"\n  Tentativa {attempt} interrompida ({exc}); retomando...", file=sys.stderr)
    if not partial.exists() or partial.stat().st_size != EXPECTED_SIZE:
        raise SystemExit("Download incompleto. Rode o comando novamente para retomar.")
    print("Conferindo SHA-256...")
    digest = sha256_of(partial)
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"SHA-256 inesperado ({digest}). O arquivo NÃO será usado.")
    partial.rename(destination)
    print(f"OK: {destination} (SHA-256 {digest})")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description="Baixa o Lending Club (aceitos) com verificação de hash")
    parser.add_argument("--dest", type=Path, default=DATA_DIR / DEFAULT_DATA_FILE)
    args = parser.parse_args(argv)
    download(args.dest)


if __name__ == "__main__":
    main()
