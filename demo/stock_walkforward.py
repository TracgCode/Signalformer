"""Walk-forward evaluation of the multi-window polynomial ridge without a fixed alpha (TWStock).

For every year Y from 2022: alpha is chosen by fitting on the years before Y - 1 and taking the best MSE on Y - 1,
then the ridge is refitted with that alpha on all the years before Y and forecasts Y. So every forecast and every
choice uses only the past. Reported per year and pooled: alpha, R2 (against the training mean), the sign score
mean(sign(mu) r) and its split into the market part (the day's mean return times the sign) and the cross-sectional
part, and the score minus always down, with day-clustered standard errors.

Training length: the same with the training set limited to the last L years before Y (L = 1, 2, 3; alpha chosen
the same way on Y - 1 with the L years before it), to see whether more data or more recent data helps.

    python demo/stock_walkforward.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_poly import BP, features, load  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_model import Ridge, clustered_mean  # noqa: E402,F401

ALPHAS = 10.0 ** np.arange(-1, 8)
FIRST = 61
EVAL = (2022, 2023, 2024, 2025, 2026)


def main() -> None:
    dates, r, traded, gap, known = load()
    X, y, _ = features(dates, r, traded, gap, known)
    T, S, K = X.shape
    Xf, yf = X.reshape(-1, K), y.reshape(-1)
    dayf, yearf = np.repeat(np.arange(T), S), np.repeat(dates.year.to_numpy(), S)
    ok = np.repeat(np.arange(T) >= FIRST, S) & np.isfinite(Xf).all(1) & np.isfinite(yf)
    Xf, yf, dayf, yearf = Xf[ok], yf[ok], dayf[ok], yearf[ok]

    def run(Y, L=None):
        lo = -np.inf if L is None else Y - 1 - L
        inner = (yearf < Y - 1) & (yearf >= lo)
        sel = yearf == Y - 1
        m = Ridge(Xf[inner], yf[inner])
        alpha = min(ALPHAS, key=lambda a: np.mean((m.predict(Xf[sel], m.weights(a)) - yf[sel]) ** 2))
        lo = -np.inf if L is None else Y - L
        tr = (yearf < Y) & (yearf >= lo)
        m = Ridge(Xf[tr], yf[tr])
        te = yearf == Y
        return dict(alpha=alpha, mu=m.predict(Xf[te], m.weights(alpha)), y=yf[te], day=dayf[te], c0=m.c0,
                    n=tr.sum())

    def metrics(d):
        mu, ys, day = d["mu"], d["y"], d["day"]
        days, inv = np.unique(day, return_inverse=True)
        mkt = (np.bincount(inv, weights=ys) / np.bincount(inv))[inv]
        sgn = np.where(mu > 0, 1.0, -1.0)
        r2 = 1 - np.mean((mu - ys) ** 2) / np.mean((d["c0"] - ys) ** 2)
        return dict(r2=r2, score=clustered_mean(sgn * ys, day), down=clustered_mean(-ys, day),
                    market=clustered_mean(sgn * mkt, day), cross=clustered_mean(sgn * (ys - mkt), day),
                    gain=clustered_mean((sgn + 1) * ys, day))

    fmt = lambda t: f"{BP * t[0]:+5.1f} ± {BP * t[1]:3.1f}"
    for L in (None, 1, 2, 3):
        title = "all earlier years" if L is None else f"the last {L} year(s)"
        print(f"\n== trained on {title}")
        print(f"{'year':5s} {'train':>7s} {'alpha':>6s} {'R2':>8s} {'always down':>13s} {'score':>13s} "
              f"{'market':>13s} {'cross':>13s} {'vs down':>13s}")
        runs = []
        for Y in EVAL:
            if L is not None and Y - 1 - L < 2020:
                continue
            d = run(Y, L)
            runs.append(d)
            m = metrics(d)
            print(f"{Y:<5d} {d['n']:7d} {'1e%d' % np.log10(d['alpha']):>6s} {m['r2']:+8.4f} {fmt(m['down']):>13s} "
                  f"{fmt(m['score']):>13s} {fmt(m['market']):>13s} {fmt(m['cross']):>13s} {fmt(m['gain']):>13s}",
                  flush=True)
        pooled = dict(mu=np.concatenate([d["mu"] for d in runs]), y=np.concatenate([d["y"] for d in runs]),
                      day=np.concatenate([d["day"] for d in runs]),
                      c0=np.concatenate([np.full(len(d["y"]), d["c0"]) for d in runs]))
        m = metrics(pooled)
        yrs = [Y for Y in EVAL if L is None or Y - 1 - L >= 2020]
        print(f"{'pool':5s} {'':7s} {'':6s} {m['r2']:+8.4f} {fmt(m['down']):>13s} {fmt(m['score']):>13s} "
              f"{fmt(m['market']):>13s} {fmt(m['cross']):>13s} {fmt(m['gain']):>13s}   ({yrs[0]}-{yrs[-1]})")


if __name__ == "__main__":
    main()
