"""Benchmark loading with the standard long-term forecasting protocol (Informer / Autoformer / PatchTST).

- ETT (15-minute): 12 / 4 / 4 months for train / validation / test.
- Weather, Traffic: 70% / 10% / 20%, split in time order.
- Validation and test windows start seq_len steps before their split, so their inputs may overlap the previous
  split while their targets never do.
- Values are z-scored with training-split statistics only.
- Calendar features default to hour of day and day of week. Day of year is opt-in: on datasets spanning one
  or two years it acts as a date ID that the model memorizes, and test months may never appear in training.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    ett_minutely: bool = False  # use the fixed ETT month-based borders


SPECS = {
    "ETTm1": DatasetSpec("ETT/ETTm1.csv", ett_minutely=True),
    "ETTm2": DatasetSpec("ETT/ETTm2.csv", ett_minutely=True),
    "Weather": DatasetSpec("Weather/weather.csv"),
    "Traffic": DatasetSpec("Traffic/traffic.csv"),
}


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (timestamps as datetime64[s], values as float32 (T, C)). The first column must be the date.

    Parsing is slow for large files (Traffic is 136 MB), so the result is cached as <name>.npz next to the CSV
    and reused while it is newer than the CSV.
    """
    cache = path.with_suffix(".npz")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        with np.load(cache) as data:
            return data["dates"].astype("datetime64[s]"), data["values"]
    dates, values = parse_csv(path)
    np.savez(cache, dates=dates.astype(np.int64), values=values)
    return dates, values


def num_signals(name: str) -> int:
    """Number of signal columns, read from the CSV header only."""
    with open(DATA_DIR / SPECS[name].path, encoding="latin-1") as f:
        return len(f.readline().split(",")) - 1


def parse_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, encoding="latin-1") as f:  # Weather headers contain non-UTF-8 unit symbols
        f.readline()
        dates, rows = [], []
        for line in f:
            date, *values = line.rstrip("\n").split(",")
            dates.append(date)
            rows.append(values)
    return np.array(dates, dtype="datetime64[s]"), np.array(rows, dtype=np.float32)


TIME_FIELDS = ("hour", "dow", "doy")  # hour of day, day of week, day of year


def time_features(dates: np.ndarray, fields: tuple[str, ...] = ("hour", "dow")) -> np.ndarray:
    """Cyclical calendar encodings, one sin/cos pair per field: (T, 2 * len(fields)) float32."""
    day = dates.astype("datetime64[D]")
    values = {
        "hour": ((dates - day).astype(np.int64) / 3600.0, 24.0),
        "dow": ((day.astype(np.int64) + 3) % 7, 7.0),  # 1970-01-01 was a Thursday; 0 = Monday
        "doy": ((day - dates.astype("datetime64[Y]")).astype(np.int64), 365.25),
    }
    parts = []
    for field in fields:
        value, period = values[field]
        angle = 2 * np.pi * value / period
        parts += [np.sin(angle), np.cos(angle)]
    if not parts:
        return np.zeros((len(dates), 0), dtype=np.float32)
    return np.stack(parts, axis=1).astype(np.float32)


def split_borders(length: int, seq_len: int, spec: DatasetSpec) -> list[tuple[int, int]]:
    if spec.ett_minutely:
        month = 30 * 24 * 4
        ends = [12 * month, 16 * month, 20 * month]
    else:
        n_train, n_test = int(length * 0.7), int(length * 0.2)
        ends = [n_train, length - n_test, length]
    starts = [0, ends[0] - seq_len, ends[1] - seq_len]
    return list(zip(starts, ends))


class ForecastWindows(Dataset):
    """Sliding windows: x (C, seq_len), y (C, pred_len), plus time features for both (possibly 0-wide)."""

    def __init__(self, values: np.ndarray, times: np.ndarray, seq_len: int, pred_len: int) -> None:
        self.values = torch.from_numpy(values.T.copy())  # (C, T)
        self.times = torch.from_numpy(times)  # (T, TIME_DIM)
        self.seq_len, self.pred_len = seq_len, pred_len

    def __len__(self) -> int:
        return self.values.size(1) - self.seq_len - self.pred_len + 1

    def __getitem__(self, i: int):
        mid, end = i + self.seq_len, i + self.seq_len + self.pred_len
        return self.values[:, i:mid], self.values[:, mid:end], self.times[i:mid], self.times[mid:end]


def build_splits(
    name: str, seq_len: int, pred_len: int, time_fields: tuple[str, ...] = ("hour", "dow")
) -> dict[str, ForecastWindows]:
    spec = SPECS[name]
    path = DATA_DIR / spec.path
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python demo/download_data.py` first")
    dates, values = load_csv(path)
    borders = split_borders(len(values), seq_len, spec)
    train_start, train_end = borders[0]
    mean = values[train_start:train_end].mean(axis=0)
    std = values[train_start:train_end].std(axis=0) + 1e-8
    values = (values - mean) / std
    times = time_features(dates, time_fields)
    return {
        split: ForecastWindows(values[start:end], times[start:end], seq_len, pred_len)
        for split, (start, end) in zip(("train", "val", "test"), borders)
    }
