"""Performance profile of the best sign-score model: the d 32 soft-Score set model of stock_score_itoken.py final2.

One walk-forward fold (default Y = 2026, the largest training set) is run stage by stage with wall-clock timers
(CUDA synchronized): data loading and the 52 base features, the pre-open features, the two ĝ_pre gap ridges, the
MSE ridge (alpha choice and the out-of-sample mu that the inputs need), the input tensors, then for a few seeds the
selection fit on Y - 2 and earlier with early stopping on Y - 1, the refit on all years before Y and the forecast of
year Y. Per training step and per epoch times, the peak GPU memory, the parameter count, and a torch.profiler table
of the top operators over 50 training steps; inference latency for one day (GPU and CPU) and for the whole year.
The cost of the full final2 run (5 folds x 40 seeds) is extrapolated from the measured per-epoch times and the epoch
counts in demo/runs/stock_score_itoken_final2.log when present.
Usage: python demo/stock_score_profile.py [year] [n_seeds]
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stock_score_itoken as it  # noqa: E402
from stock_itrans_news import ghat_for  # noqa: E402
from stock_news_attn2 import oos_mu  # noqa: E402
from stock_news_gap import gap_block  # noqa: E402
from stock_overnight import preopen_features  # noqa: E402
from stock_poly import ROOT, load  # noqa: E402
from stock_recurrent import base_features, lagged  # noqa: E402
from stock_walkforward import ALPHAS, FIRST, Ridge  # noqa: E402

CFG = it.FINAL2["d 32, soft Score, 40 seeds"]
T0 = {}


def tick(name, t0):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    T0[name] = T0.get(name, 0.0) + dt
    print(f"  {name:48s} {dt:8.2f} s", flush=True)
    return time.perf_counter()


def main() -> None:
    Y = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    dev = it.DEV
    print(f"fold Y = {Y}, {n_seeds} seeds, device {dev} ({torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}), "
          f"torch {torch.__version__}, CPU threads {torch.get_num_threads()}", flush=True)
    t = time.perf_counter()
    dates, r, traded, gap, known = load()
    T, S = r.shape
    t = tick("load prices", t)
    info = json.loads((ROOT / "info.json").read_text())
    stocks = info["Top100@Y"]
    X, y = base_features(dates, r, traded, gap, known, stocks)
    t = tick("52 base features", t)
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    half = year * 2 + (np.repeat(dates.month.to_numpy()[:, None], S, axis=1) > 6)
    yrs = dates.year.to_numpy()
    g = gap.to_numpy()
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y) & np.isfinite(g)
    trd = traded.to_numpy()
    pre = ok & (year < 2022)
    r0 = np.where(trd, np.nan_to_num(r.to_numpy()), 0.0)
    g0 = np.where(trd, np.nan_to_num(g), 0.0)
    sig = np.stack([lagged(r0, 1) / r0[pre].std(), lagged(g0, 1) / g0[pre].std(), lagged(trd.astype(float), 1)], -1)
    PO = preopen_features(dates, gap, traded)
    XG = np.concatenate([X, PO], -1)
    t = tick("pre-open features", t)
    gh_s = ghat_for(XG, g, ok, year, Y - 1)
    gh_r = ghat_for(XG, g, ok, year, Y)
    t = tick("gap ridges ĝ_pre (selection + refit)", t)
    Zs = np.concatenate([X, gap_block(gh_s, ok)], -1)
    Zr = np.concatenate([X, gap_block(gh_r, ok)], -1)
    inner, sel, rows = ok & (year < Y - 1), ok & (year == Y - 1), ok & (year < Y)
    m = Ridge(Zs[inner], y[inner])
    a = min(ALPHAS, key=lambda a: np.mean((m.predict(Zs[sel], m.weights(a)) - y[sel]) ** 2))
    mu_s = oos_mu(Zs, y, ok, year, half, a, Y - 1)
    mu_r = oos_mu(Zr, y, ok, year, half, a, Y)
    t = tick("MSE ridge (alpha, out-of-sample mu)", t)
    inp_s = it.Inputs("day", Zs, mu_s, y, ok, trd, sig, inner)
    inp_r = it.Inputs("day", Zr, mu_r, y, ok, trd, sig, rows)
    Yt = torch.tensor(np.where(ok, y, 0.0), dtype=torch.float32, device=dev)
    Mt = torch.tensor(ok, device=dev)
    o_s = torch.tensor(np.nan_to_num(mu_s / np.nanstd(mu_s[inner])), dtype=torch.float32, device=dev)
    o_r = torch.tensor(np.nan_to_num(mu_r / np.nanstd(mu_r[rows])), dtype=torch.float32, device=dev)
    dd = lambda cond: torch.tensor(np.nonzero(cond & ok.any(1))[0], device=dev)
    d_in, d_sel, d_rows, d_te = dd(yrs < Y - 1), dd(yrs == Y - 1), dd(yrs < Y), dd(yrs == Y)
    t = tick("input tensors to the GPU", t)
    print(f"  days: selection fit {len(d_in)}, validation {len(d_sel)}, refit {len(d_rows)}, test {len(d_te)}; "
          f"inputs {inp_r.cur_dim}; stock-days in the refit {int(ok[yrs < Y].sum())}", flush=True)

    # parameter count
    net0 = it.Wrap(it.TokNet(0, inp_r.cur_dim, "e4h", 32, False), False, False)
    n_all = sum(p.numel() for p in net0.parameters())
    n_used = n_all - sum(p.numel() for p in net0.net.out2.parameters())
    print(f"  parameters: {n_used} used ({n_all} with the unused out2 head)", flush=True)

    # training: per seed selection + refit + forecast, with per-epoch times
    steps_per_epoch = {k: -(-len(v) // CFG[11]) for k, v in (("sel", d_in), ("refit", d_rows))}
    res = []
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for seed in range(n_seeds):
        t = time.perf_counter()
        _, ep = it.train(CFG, inp_s, Yt, o_s, Mt, d_in, seed, vdays=d_sel)
        t_sel = time.perf_counter() - t
        ep = max(ep, 1)
        n_sel_ep = min(40, ep + 3)  # early stopping runs 3 epochs past the best (or stops at 40)
        t = time.perf_counter()
        net, _ = it.train(CFG, inp_r, Yt, o_r, Mt, d_rows, seed, epochs=ep)
        if dev == "cuda":
            torch.cuda.synchronize()
        t_ref = time.perf_counter() - t
        t = time.perf_counter()
        with torch.no_grad():
            out = torch.cat([net(*inp_r.batch(d_te[i:i + 64]), o_r[d_te[i:i + 64]], Mt[d_te[i:i + 64]])
                             for i in range(0, len(d_te), 64)])
        if dev == "cuda":
            torch.cuda.synchronize()
        t_fc = time.perf_counter() - t
        res.append((seed, ep, n_sel_ep, t_sel, t_ref, t_fc))
        print(f"  seed {seed}: best epoch {ep}; selection {t_sel:6.2f} s ({n_sel_ep} epochs, "
              f"{t_sel / n_sel_ep:5.3f} s/epoch, {1e3 * t_sel / (n_sel_ep * steps_per_epoch['sel']):5.2f} ms/step); "
              f"refit {t_ref:6.2f} s ({ep} epochs, {t_ref / ep:5.3f} s/epoch, "
              f"{1e3 * t_ref / (ep * steps_per_epoch['refit']):5.2f} ms/step); forecast of {len(d_te)} days "
              f"{1e3 * t_fc:6.1f} ms", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else float("nan")
    print(f"  peak GPU memory allocated by torch: {peak:.0f} MiB", flush=True)
    s_ep_sel = np.mean([r_[3] / r_[2] for r_ in res])
    s_ep_ref = np.mean([r_[4] / r_[1] for r_ in res])

    # one-day inference latency
    net.eval()
    b1 = d_te[-1:]
    for name, n_ in ((dev, net), ("cpu", None)):
        if n_ is None:
            n_ = it.Wrap(it.TokNet(0, inp_r.cur_dim, "e4h", 32, False), False, False)
            n_.load_state_dict({k: v.cpu() for k, v in net.state_dict().items()})
            n_.eval()
            args = (None, inp_r.cur[b1].cpu(), o_r[b1].cpu(), Mt[b1].cpu())
        else:
            args = (None, inp_r.cur[b1], o_r[b1], Mt[b1])
        with torch.no_grad():
            for _ in range(10):
                n_(*args)
            if name == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(200):
                n_(*args)
            if name == "cuda":
                torch.cuda.synchronize()
        print(f"  one-day forecast (100 stocks), one network, {name}: {1e3 * (time.perf_counter() - t) / 200:.3f} ms; "
              f"40-seed ensemble about {40e3 * (time.perf_counter() - t) / 200:.1f} ms", flush=True)

    # operator profile of 50 training steps
    from torch.profiler import ProfilerActivity, profile
    torch.manual_seed(0)
    net = it.Wrap(it.TokNet(0, inp_r.cur_dim, "e4h", 32, False), False, False).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    perm = d_rows[torch.randperm(len(d_rows), device=dev)]
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev == "cuda" else [])

    def step(i):
        b = perm[16 * i:16 * i + 16]
        mm = Mt[b]
        f = net(*inp_r.batch(b), o_r[b], mm)[mm]
        loss = it.loss_fn("soft", f, Yt[b][mm])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for i in range(10):
        step(i)
    with profile(activities=acts) as prof:
        for i in range(10, 60):
            step(i)
        if dev == "cuda":
            torch.cuda.synchronize()
    key = "cuda_time_total" if dev == "cuda" else "cpu_time_total"
    print("\n######## torch.profiler, 50 training steps (batch of 16 days), top operators by " + key)
    print(prof.key_averages().table(sort_by=key, row_limit=15), flush=True)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10), flush=True)

    # extrapolation to the full final2 run
    log = next((p_ for p_ in (Path(__file__).resolve().parent / "runs" / "stock_score_itoken_final2.log",
                              Path(__file__).resolve().parents[1] / "runs" / "stock_score_itoken_final2.log") if p_.exists()),
               Path("missing"))
    eps = []
    if log.exists():
        for line in log.read_text().splitlines():
            mm = re.search(r"d 32, soft Score, 40 seeds: epochs \[([0-9, ]+)\]", line)
            if mm:
                eps.append([int(x) for x in mm.group(1).split(",")])
    print("\n######## summary")
    print(f"  per epoch on fold {Y}: selection fit {s_ep_sel:.3f} s ({steps_per_epoch['sel']} steps), refit "
          f"{s_ep_ref:.3f} s ({steps_per_epoch['refit']} steps)")
    if eps:
        tot_sel = sum(min(40, e + 3) for f_ in eps for e in f_)
        tot_ref = sum(e for f_ in eps for e in f_)
        print(f"  final2 d 32 (5 folds x 40 seeds, epochs from the log): {tot_sel} selection and {tot_ref} refit epochs; "
              f"at this fold's per-epoch times (an upper bound, earlier folds have fewer days) about "
              f"{(tot_sel * s_ep_sel + tot_ref * s_ep_ref) / 60:.1f} min of training")
    print("  stage times (s): " + ", ".join(f"{k} {v:.2f}" for k, v in T0.items()), flush=True)


if __name__ == "__main__":
    main()
