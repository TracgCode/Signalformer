"""Download the benchmark datasets used by the demo into demo/data/.

The datasets belong to their original authors and have their own licenses (see demo/README.md). They are
downloaded locally and ignored by git; do not commit them.

    python demo/download_data.py
"""

import io
import urllib.request
import zipfile
from pathlib import Path

from datasets import LOB_TEST_FILES, LOB_TRAIN_FILES

DATA_DIR = Path(__file__).resolve().parent / "data"

# ETT: original repository by the Informer authors
ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
# Weather and Traffic: Autoformer benchmark files, mirrored by Nixtla (datasetsforecast)
SIX_DATASETS_URL = "https://nixtla-public.s3.amazonaws.com/datasetsforecast/all_six_datasets.zip"
FROM_ARCHIVE = {
    "all_six_datasets/Weather/Y_df.csv": "Weather/weather.csv",
    "all_six_datasets/TrafficL/Y_df.csv": "Traffic/traffic.csv",
}
# FI-2010 (CC BY 4.0, Ntakaris et al.): the DecPre-normalized Setup 2 files distributed with the DeepLOB code
FI2010_URL = (
    "https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/"
    "master/data/data.zip"
)


def download(url: str) -> bytes:
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response:
        return response.read()


def main() -> None:
    for name in ("ETTm1", "ETTm2"):
        target = DATA_DIR / "ETT" / f"{name}.csv"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(download(ETT_URL.format(name=name)))

    missing = {src: DATA_DIR / dst for src, dst in FROM_ARCHIVE.items() if not (DATA_DIR / dst).exists()}
    if missing:
        archive = zipfile.ZipFile(io.BytesIO(download(SIX_DATASETS_URL)))
        for src, target in missing.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(src))

    lob_dir = DATA_DIR / "FI2010"
    lob_files = (*LOB_TRAIN_FILES, *LOB_TEST_FILES)
    if not all((lob_dir / name).exists() for name in lob_files):
        archive = zipfile.ZipFile(io.BytesIO(download(FI2010_URL)))
        lob_dir.mkdir(parents=True, exist_ok=True)
        for name in lob_files:
            (lob_dir / name).write_bytes(archive.read(name))

    for path in sorted([*DATA_DIR.rglob("*.csv"), *lob_dir.glob("*.txt")]):
        print(f"ready: {path.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
