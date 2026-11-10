"""Daily cross-sectional stock return models with a news branch, in NumPy and PyTorch.

The forecast of a stock's open-to-close return is a two-stage model, mu + e_hat:

- stage 1, `walk_forward`: a ridge regression on the day's features, refitted every year on all earlier years with
  its penalty chosen on the year before (`Ridge`, `select_alpha`). `gap_block` turns a forecast of the opening gap
  log(open / previous close) into three features (the forecast, its size and the day's mean), so a news-predicted gap
  can enter the ridge;
- stage 2: a neural network forecasts the ridge residual e = r - mu. Its output head starts at zero, so an untrained
  stage 2 is the ridge itself. Two networks share `StockTokenEncoder` (one token per stock from its whole input
  window, iTransformer style, with self-attention across the stocks):
  - `StockNewsAttention`: cross attention between the stock tokens and the day's news items, in four arrangements
    of query, key and value (`mode`);
  - `StockIdentityAttention`: every news item queries the stocks by their identity (keys from a fixed company
    embedding, e.g. `comovement_embedding`, plus a learned one), reads their price states (values), and the item
    outputs are pooled back onto the stocks the item attended to.

`TypedForm` turns a local causal language model into a typed classifier of news texts: the form is forced token by
token, and at each field the next token is restricted to the allowed choice letters, whose renormalized
probabilities are kept as the field's output.

Nothing here reads data; the data pipeline and the experiments are in demo/ (see demo/EXPERIMENTS.md).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

ALPHAS = tuple(10.0 ** np.arange(-1, 8))


# ---------------------------------------------------------------------------------------------------------------------
# Stage 1: walk-forward ridge
# ---------------------------------------------------------------------------------------------------------------------

class Ridge:
    """Closed-form ridge on standardized features with an unpenalized intercept (the target mean).

    The Gram matrix is computed once, so `weights` is cheap for every penalty `alpha` (the penalty is on the sum, not
    the mean, of the squared errors).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        self.c0 = y.mean()
        z = (X - self.mu) / self.sd
        self.gram, self.rhs = z.T @ z, z.T @ (y - self.c0)

    def weights(self, alpha: float) -> np.ndarray:
        return np.linalg.solve(self.gram + alpha * np.eye(len(self.mu)), self.rhs)

    def predict(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ w + self.c0


def select_alpha(X_fit: np.ndarray, y_fit: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                 alphas: Sequence[float] = ALPHAS) -> float:
    """The penalty with the lowest validation MSE of the ridge fitted on (X_fit, y_fit)."""
    m = Ridge(X_fit, y_fit)
    return min(alphas, key=lambda a: np.mean((m.predict(X_val, m.weights(a)) - y_val) ** 2))


def walk_forward(Z: np.ndarray, y: np.ndarray, ok: np.ndarray, year: np.ndarray, eval_years: Sequence[int],
                 alphas: Sequence[float] = ALPHAS, loo: bool = False):
    """Walk-forward ridge forecasts on (T, S) panels.

    For every Y in `eval_years` the penalty is chosen on Y - 1 from the fit on the years before Y - 1, then the ridge
    is refitted on all years before Y and forecasts Y. `ok` marks the usable rows, `year` the year of every row.
    With `loo` it also returns, per Y, forecasts on the years before Y from the fit without that year (so a second
    stage never trains on in-sample forecasts). Returns (mu, {Y: loo forecasts}).
    """
    mu = np.full(y.shape, np.nan)
    inner_oos = {}
    for Y in eval_years:
        inner, sel, rows, te = ok & (year < Y - 1), ok & (year == Y - 1), ok & (year < Y), ok & (year == Y)
        a = select_alpha(Z[inner], y[inner], Z[sel], y[sel], alphas)
        m = Ridge(Z[rows], y[rows])
        mu[te] = m.predict(Z[te], m.weights(a))
        if loo:
            o = np.full(y.shape, np.nan)
            o[te] = mu[te]
            for yy in np.unique(year[rows]):
                fit, at = rows & (year != yy), rows & (year == yy)
                mf = Ridge(Z[fit], y[fit])
                o[at] = mf.predict(Z[at], mf.weights(a))
            inner_oos[Y] = o
    return mu, inner_oos


def block_out_of_sample(X: np.ndarray, y: np.ndarray, ok: np.ndarray, year: np.ndarray, block: np.ndarray,
                        alpha: float, Yb: int) -> np.ndarray:
    """Ridge forecasts for the stage-2 training rows: on the years before Yb by leaving one `block` (e.g. a half
    year) out at a time, and on Yb from the fit on all years before it."""
    mu = np.full(y.shape, np.nan)
    rows = ok & (year < Yb)
    for b in np.unique(block[rows]):
        fit, at = rows & (block != b), rows & (block == b)
        m = Ridge(X[fit], y[fit])
        mu[at] = m.predict(X[at], m.weights(alpha))
    m = Ridge(X[rows], y[rows])
    at = ok & (year == Yb)
    mu[at] = m.predict(X[at], m.weights(alpha))
    return mu


def gap_block(g: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """(T, S) gap forecasts -> (T, S, 3): the forecast, its absolute value and the day's mean over the usable rows."""
    gm = np.where(ok, g, np.nan)
    mk = np.nanmean(gm, 1, keepdims=True).repeat(g.shape[1], 1)
    return np.stack([np.nan_to_num(g), np.abs(np.nan_to_num(g)), np.nan_to_num(mk)], -1)


def clustered_mean(x: np.ndarray, day: np.ndarray) -> tuple[float, float]:
    """The mean of x and its standard error clustered by day (the stocks of one day are not independent)."""
    _, inv = np.unique(day, return_inverse=True)
    s, n = np.bincount(inv, weights=x), np.bincount(inv)
    m = s.sum() / n.sum()
    return m, np.sqrt(np.sum((s - n * m) ** 2)) / n.sum()


def sign_score(pred: np.ndarray, y: np.ndarray, day: np.ndarray) -> tuple[float, float]:
    """The mean of sign(pred) * y (long when the forecast is positive, short otherwise), day-clustered."""
    return clustered_mean(np.where(pred > 0, 1.0, -1.0) * y, day)


def comovement_embedding(returns: np.ndarray, dim: int = 16) -> np.ndarray:
    """(T, S) returns of a training period -> (S, dim) stock embedding from the leading eigenvectors of their
    correlation matrix, scaled by the square root of the eigenvalues (stocks that move together are close).
    Missing returns count as 0 after standardization."""
    z = (returns - np.nanmean(returns, 0)) / (np.nanstd(returns, 0) + 1e-12)
    z = np.nan_to_num(z)
    C = z.T @ z / len(z)
    vals, vecs = np.linalg.eigh(C)
    top = np.argsort(vals)[::-1][:dim]
    return (vecs[:, top] * np.sqrt(np.clip(vals[top], 0, None))).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------------
# Stage 2: stock tokens and news attention
# ---------------------------------------------------------------------------------------------------------------------

PAD, MARKET = -2, -1   # tags of the news items: a stock index, MARKET for market-wide items, PAD for empty slots


class StockTokenEncoder(nn.Module):
    """One token per stock (iTransformer style): Linear of its flattened input window plus Linear of the day's
    features, dropout, then one pre-norm self-attention layer across the stocks.

    win: (B, S, window * n_signals); cur: (B, S, n_features) -> (B, S, d_model)
    """

    def __init__(self, window_dim: int, n_features: int, d_model: int = 16, num_heads: int = 2,
                 dropout: float = 0.3, attn_dropout: float = 0.1):
        super().__init__()
        self.win = nn.Linear(window_dim, d_model)
        self.cur = nn.Linear(n_features, d_model)
        self.drop = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.att = nn.MultiheadAttention(d_model, num_heads, dropout=attn_dropout, batch_first=True)

    def forward(self, win: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.win(win) + self.cur(cur))
        x = self.ln1(h)
        return h + self.att(x, x, x, need_weights=False)[0]


class StockNewsAttention(nn.Module):
    """Stock tokens with a cross attention to the day's news items, forecasting the ridge residual.

    news: (B, N, news_dim) item vectors; tags: (B, N) the stock index each item is about (MARKET, PAD as above).
    Modes (d = d_model; h_s the token of stock s, LN a layer norm; logits scaled by 1 / sqrt(d)):

    - "stockq": q = W_q LN(h_s), keys and values from the items (W_k z_n, W_v z_n), a learned null key and value, and
      a learned bias beta on the items that tag stock s;
    - "newsq": q = W_q u_s from the mean of stock s's own items and log(1 + their count); keys and values from the
      stock tokens (W_k LN(h_j), W_v LN(h_j));
    - "sigval": q = W_q LN(h_s), k = W_k z_n, but v = W_v LN(h_tag(n)), the token of the stock the item tags (a
      learned market token for market items): the news decide which stocks' states a stock reads;
    - "itemq": every item queries the stock tokens (o_n), then each stock attends over the o_n (null key, beta).

    e_hat_s = w . [h_s, c_s] with w = 0 at the start; with `use_news=False` only w . h_s.
    """

    MODES = ("stockq", "newsq", "sigval", "itemq")

    def __init__(self, n_stocks: int, window_dim: int, n_features: int, news_dim: int = 65,
                 use_news: bool = True, mode: str = "stockq", d_model: int = 16, num_heads: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        D = d_model
        self.news, self.mode, self.d = use_news, mode, D
        self.enc = StockTokenEncoder(window_dim, n_features, D, num_heads, dropout)
        if use_news and mode in ("sigval", "itemq"):
            self.ln2 = nn.LayerNorm(D)
            self.k0 = nn.Parameter(torch.zeros(D))
            self.v0 = nn.Parameter(torch.zeros(D))
            self.beta = nn.Parameter(torch.zeros(1))
            if mode == "sigval":
                self.wq = nn.Linear(D, D, bias=False)
                self.wk = nn.Linear(news_dim, D, bias=False)
                self.wv = nn.Linear(D, D, bias=False)
                self.mkt = nn.Parameter(torch.zeros(D))
            else:
                self.wq = nn.Linear(news_dim, D, bias=False)
                self.wk = nn.Linear(D, D, bias=False)
                self.wv = nn.Linear(D, D, bias=False)
                self.wq2 = nn.Linear(D, D, bias=False)
                self.wk2 = nn.Linear(D, D, bias=False)
        elif use_news and mode == "newsq":
            self.ln2 = nn.LayerNorm(D)
            self.wq = nn.Linear(news_dim + 1, D, bias=False)
            self.wk = nn.Linear(D, D, bias=False)
            self.wv = nn.Linear(D, D, bias=False)
        elif use_news:
            self.ln2 = nn.LayerNorm(D)
            self.wq = nn.Linear(D, D, bias=False)
            self.wk = nn.Linear(news_dim, D, bias=False)
            self.wv = nn.Linear(news_dim, D, bias=False)
            self.k0 = nn.Parameter(torch.zeros(D))
            self.v0 = nn.Parameter(torch.zeros(D))
            self.beta = nn.Parameter(torch.zeros(1))
        self.head = nn.Linear(2 * D if use_news else D, 1, bias=False)
        nn.init.zeros_(self.head.weight)
        self.register_buffer("sid", torch.arange(n_stocks))

    def forward(self, win: torch.Tensor, cur: torch.Tensor, news: torch.Tensor | None = None,
                tags: torch.Tensor | None = None) -> torch.Tensor:
        h = self.enc(win, cur)                                                # (B, S, D)
        if not self.news:
            return self.head(h).squeeze(-1)
        D, B = self.d, len(h)
        own = (tags[:, None, :] == self.sid[None, :, None]).float()          # (B, S, N): item n tags stock s
        pad = tags == PAD
        if self.mode == "newsq":
            n = own.sum(-1, keepdim=True)
            u = torch.cat([own @ news / n.clamp(min=1), torch.log1p(n)], -1)  # 0 without an item
            x = self.ln2(h)
            a = torch.softmax(self.wq(u) @ self.wk(x).transpose(1, 2) / D ** 0.5, -1)   # (B, S, S)
            return self.head(torch.cat([h, a @ self.wv(x)], -1)).squeeze(-1)
        if self.mode == "stockq":
            q = self.wq(self.ln2(h))
            kn, vn = self.wk(news), self.wv(news)
        else:
            x = self.ln2(h)
            if self.mode == "sigval":
                q = self.wq(x)
                kn = self.wk(news)
                src = torch.cat([x, self.mkt.expand(B, 1, D)], 1)            # index S = the market token
                ti = torch.where(tags >= 0, tags, torch.full_like(tags, x.shape[1]))
                vn = self.wv(torch.gather(src, 1, ti[..., None].expand(-1, -1, D)))
            else:                                                             # itemq
                a1 = torch.softmax(self.wq(news) @ self.wk(x).transpose(1, 2) / D ** 0.5, -1)   # (B, N, S)
                o = a1 @ self.wv(x)
                q, kn, vn = self.wq2(x), self.wk2(o), o
        k = torch.cat([self.k0.expand(B, 1, D), kn], 1)
        v = torch.cat([self.v0.expand(B, 1, D), vn], 1)
        logit = q @ k.transpose(1, 2) / D ** 0.5                              # (B, S, 1 + N)
        logit[..., 1:] = (logit[..., 1:] + self.beta * own).masked_fill(pad[:, None, :], float("-inf"))
        c = torch.softmax(logit, -1) @ v
        return self.head(torch.cat([h, c], -1)).squeeze(-1)


class StockIdentityAttention(nn.Module):
    """News items query the stocks by identity, read their price states, and are pooled back onto the stocks.

    For item n of the day and stock j (d = d_model):
        q_n = W_q z_n,   k_j = W_k [id_j, e_j],   v_j = W_v LN(h_j),
        a_nj = softmax_j(q_n . k_j / sqrt(d)),   o_n = sum_j a_nj v_j,
    with id_j a fixed company embedding (`identity`, e.g. from `comovement_embedding` or a text encoder of company
    profiles) and e_j a learned embedding (started at 0). An item that tags a stock gets a learned bias beta on that
    stock's logit; market items and fixed weights w_n (e.g. the size of a pre-pooled cluster) are allowed.
    Pooled back onto stock s:
        A_s = sum_n w_n a_ns,   c_s = sum_n w_n a_ns o_n / (eps + A_s),   m_s = log(1 + A_s),
    so c_s is an attention-weighted mean (not a sum, so stocks with many items do not get a larger input), and
    m_s says how much news concerned s. e_hat_s = w . [h_s, c_s, m_s], w = 0 at the start.

    The identity carries the inductive bias: an item about one company reaches the stocks whose identity matches its
    content, and what they contribute is their current price state.
    """

    def __init__(self, n_stocks: int, window_dim: int, n_features: int, identity: torch.Tensor,
                 news_dim: int, d_model: int = 16, d_learned: int = 8, num_heads: int = 2, dropout: float = 0.3,
                 eps: float = 1e-3):
        super().__init__()
        D = d_model
        self.d, self.eps = D, eps
        self.enc = StockTokenEncoder(window_dim, n_features, D, num_heads, dropout)
        self.register_buffer("identity", torch.as_tensor(identity, dtype=torch.float32))   # (S, id_dim)
        self.learned = nn.Parameter(torch.zeros(n_stocks, d_learned))
        self.ln2 = nn.LayerNorm(D)
        self.wq = nn.Linear(news_dim, D, bias=False)
        self.wk = nn.Linear(self.identity.shape[1] + d_learned, D, bias=False)
        self.wv = nn.Linear(D, D, bias=False)
        self.beta = nn.Parameter(torch.zeros(1))
        self.head = nn.Linear(2 * D + 1, 1, bias=False)
        nn.init.zeros_(self.head.weight)
        self.register_buffer("sid", torch.arange(n_stocks))

    def item_attention(self, news: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """(B, N, S) attention of every item over the stocks (0 for padding items)."""
        k = self.wk(torch.cat([self.identity, self.learned], -1))            # (S, D)
        logit = self.wq(news) @ k.T / self.d ** 0.5                           # (B, N, S)
        own = (tags[..., None] == self.sid).float()
        a = torch.softmax(logit + self.beta * own, -1)
        return a * (tags != PAD)[..., None].float()

    def forward(self, win: torch.Tensor, cur: torch.Tensor, news: torch.Tensor, tags: torch.Tensor,
                weight: torch.Tensor | None = None) -> torch.Tensor:
        h = self.enc(win, cur)                                                # (B, S, D)
        a = self.item_attention(news, tags)                                   # (B, N, S)
        o = a @ self.wv(self.ln2(h))                                          # (B, N, D)
        if weight is not None:
            a = a * weight[..., None]
        A = a.sum(1)                                                          # (B, S)
        c = a.transpose(1, 2) @ o / (self.eps + A[..., None])                 # (B, S, D)
        return self.head(torch.cat([h, c, torch.log1p(A)[..., None]], -1)).squeeze(-1)


# ---------------------------------------------------------------------------------------------------------------------
# Typed fields from a local language model
# ---------------------------------------------------------------------------------------------------------------------

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class TypedForm:
    """A logits processor (for `transformers` `generate`) that writes a fixed form and makes each field a choice.

    fields: (name, label, options) triples. The generated text is
        "<label 1>：<letter>\\n<label 2>：<letter>…\\n<tail>"
    where every label token is forced and every letter is restricted to the first len(options) letters. At each
    choice the softmax over the allowed letters is stored in `probs[name]` (B, len(options)), and the most likely
    letter is written, so later fields are conditioned on the earlier answers. After the form (`len(plan)` tokens)
    the model generates freely (e.g. a summary). Use a fresh instance for every `generate` call.
    """

    def __init__(self, encode: Callable[[str], list[int]], fields: Sequence[tuple[str, str, Sequence[str]]],
                 tail: str = "\n摘要：", sep: str = "："):
        self.plan: list[tuple] = []
        for k, (name, label, opts) in enumerate(fields):
            self.plan += [("force", t) for t in encode(("" if k == 0 else "\n") + label + sep)]
            self.plan.append(("choice", name, [encode(LETTERS[i])[0] for i in range(len(opts))]))
        self.plan += [("force", t) for t in encode(tail)]
        self.start: int | None = None
        self.probs: dict[str, np.ndarray] = {}

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.start is None:
            self.start = input_ids.shape[1]
        step = input_ids.shape[1] - self.start
        if step >= len(self.plan):
            return scores
        p = self.plan[step]
        out = torch.full_like(scores, float("-inf"))
        if p[0] == "force":
            out[:, p[1]] = 0.0
        else:
            ids = torch.tensor(p[2], device=scores.device)
            pr = torch.softmax(scores[:, ids].float(), -1)
            self.probs[p[1]] = pr.cpu().numpy()
            out[torch.arange(len(scores)), ids[pr.argmax(-1)]] = 0.0
        return out
