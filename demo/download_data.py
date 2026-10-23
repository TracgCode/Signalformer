"""Download the benchmark datasets used by the demo into demo/data/.

The datasets belong to their original authors and have their own licenses (see demo/README.md). They are
downloaded locally and ignored by git; do not commit them.

    python demo/download_data.py
"""

import io
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# ETT: original repository by the Informer authors
ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
# Weather and Traffic: Autoformer benchmark files, mirrored by Nixtla (datasetsforecast)
SIX_DATASETS_URL = "https://nixtla-public.s3.amazonaws.com/datasetsforecast/all_six_datasets.zip"
FROM_ARCHIVE = {
    "all_six_datasets/Weather/Y_df.csv": "Weather/weather.csv",
    "all_six_datasets/TrafficL/Y_df.csv": "Traffic/traffic.csv",
}


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

    for path in sorted(DATA_DIR.rglob("*.csv")):
        print(f"ready: {path.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
