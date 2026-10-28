"""Benchmark loading with the standard long-term forecasting protocol (Informer / Autoformer / PatchTST).

- ETT (15-minute): 12 / 4 / 4 months for train / validation / test.
- Weather, Traffic: 70% / 10% / 20%, split in time order.
- Validation and test windows start seq_len steps before their split, so their inputs may overlap the previous
  split while their targets never do.
- Values are z-scored with training-split statistics only.
- Calendar features default to hour of day and day of week. Day of year is opt-in: on datasets spanning one
  or two years it acts as a date ID that the model memorizes, and test months may never appear in training.

FI-2010 (limit order book, classification) follows Setup 2 of the DeepLOB paper: the first 7 days train, the last
3 days test. See build_lob_splits.
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
    """Number of signal columns, read from the CSV header only (FI-2010: the order book features)."""
    if name in LOB_DATASETS:
        return LOB_FEATURES
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


# FI-2010: 5 stocks over 10 trading days, one column per order book event. Rows 0-39 are the ask / bid prices and
# volumes of the 10 best levels (the default features), rows 40-143 the dataset's handcrafted features (spreads,
# mid-prices, differences, means, accumulated differences, derivatives, intensities), rows 144-148 the labels for
# the horizons below.
LOB_DATASETS = ("FI2010",)
LOB_HORIZONS = (10, 20, 30, 50, 100)
LOB_FEATURES = 40
LOB_ALL_FEATURES = 144  # order book + handcrafted features
LOB_CLASSES = ("up", "stationary", "down")  # file labels 1, 2, 3
LOB_TRAIN_FILES = ("Train_Dst_NoAuction_DecPre_CF_7.txt",)  # days 1-7
LOB_TEST_FILES = tuple(f"Test_Dst_NoAuction_DecPre_CF_{i}.txt" for i in (7, 8, 9))  # days 8, 9, 10


def load_lob(path: Path) -> np.ndarray:
    """(149, T) float32 from a whitespace-separated FI-2010 file, cached as <name>.npy next to it."""
    cache = path.with_suffix(".npy")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return np.load(cache)
    data = np.fromfile(path, sep=" ", dtype=np.float32)
    data = data.reshape(149, -1)
    np.save(cache, data)
    return data


def lob_segments(ask_price: np.ndarray, jump: float = 0.05) -> list[tuple[int, int]]:
    """(start, end) of each stock in one file. The files hold the stocks one after another; a stock boundary
    shows up as a jump of the best ask price by more than `jump` (relative), which never happens within a
    stock between two consecutive events."""
    change = np.abs(np.diff(ask_price)) / np.maximum(np.abs(ask_price[:-1]), 1e-8)
    cuts = [0, *(np.flatnonzero(change > jump) + 1).tolist(), len(ask_price)]
    return list(zip(cuts[:-1], cuts[1:]))


class LobWindows(Dataset):
    """Classification windows: x (C, seq_len) ending at event t, y = the label of event t (0 up, 1 stationary,
    2 down). starts lists the first event of every window; no window crosses a stock or file boundary."""

    def __init__(self, values: np.ndarray, labels: np.ndarray, starts: np.ndarray, seq_len: int, horizon: int):
        self.values = torch.from_numpy(np.ascontiguousarray(values.T))  # (C, T)
        self.labels = torch.from_numpy(labels)  # (T,)
        self.starts = torch.from_numpy(starts)  # (num_windows,)
        self.times = torch.zeros(values.shape[0], 0)
        self.seq_len, self.pred_len = seq_len, horizon

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        start = int(self.starts[i])
        return self.values[:, start : start + self.seq_len], self.labels[start + self.seq_len - 1]


def build_lob_splits(seq_len: int, horizon: int = 10, val_fraction: float = 0.2,
                     features: int = LOB_FEATURES) -> dict[str, LobWindows]:
    """FI-2010 Setup 2 on the DecPre-normalized files, with the first `features` rows (40: the order book, 144:
    also the handcrafted features) z-scored again with training statistics. The last val_fraction of every stock's training days is the validation split (its
    windows may start in the training part, as for the forecasting splits). Every event with a full input
    window inside its stock is one window; the window ends at the event whose label it predicts."""
    if horizon not in LOB_HORIZONS:
        raise ValueError(f"FI-2010 horizons are {LOB_HORIZONS}, not {horizon}")
    if features not in (LOB_FEATURES, LOB_ALL_FEATURES):
        raise ValueError(f"FI-2010 uses {LOB_FEATURES} or {LOB_ALL_FEATURES} features, not {features}")
    root = DATA_DIR / "FI2010"
    missing = [f for f in (*LOB_TRAIN_FILES, *LOB_TEST_FILES) if not (root / f).exists()]
    if missing:
        raise FileNotFoundError(f"{root / missing[0]} not found; see the FI-2010 section of demo/README.md")
    row = 144 + LOB_HORIZONS.index(horizon)

    def load(files: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
        values, labels, segments, offset = [], [], [], 0
        for name in files:
            data = load_lob(root / name)
            values.append(data[:features].T)
            labels.append(data[row].astype(np.int64) - 1)
            segments += [(offset + a, offset + b) for a, b in lob_segments(data[0])]
            offset += data.shape[1]
        return np.concatenate(values), np.concatenate(labels), segments

    def window_starts(segments: list[tuple[int, int]]) -> np.ndarray:
        # windows [s, s + seq_len) inside each segment
        return np.concatenate([np.arange(a, b - seq_len + 1) for a, b in segments if b - a >= seq_len])

    train_values, train_labels, stocks = load(LOB_TRAIN_FILES)
    test_values, test_labels, test_segments = load(LOB_TEST_FILES)
    # The validation part of each stock: its windows end in the last val_fraction of the stock's events
    train_parts, val_starts = [], []
    for a, b in stocks:
        cut = b - int((b - a) * val_fraction)
        train_parts.append((a, cut))
        val_starts.append(np.arange(max(a, cut - seq_len + 1), b - seq_len + 1))
    fit = np.concatenate([train_values[a:b] for a, b in train_parts])
    mean, std = fit.mean(axis=0), fit.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)  # some handcrafted features are constant (all zero) in every file
    train_values = ((train_values - mean) / std).astype(np.float32)
    test_values = ((test_values - mean) / std).astype(np.float32)
    return {
        "train": LobWindows(train_values, train_labels, window_starts(train_parts), seq_len, horizon),
        "val": LobWindows(train_values, train_labels, np.concatenate(val_starts), seq_len, horizon),
        "test": LobWindows(test_values, test_labels, window_starts(test_segments), seq_len, horizon),
    }
