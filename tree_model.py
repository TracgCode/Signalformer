"""Tree-merging forecaster (experimental): signals are merged pairwise by cross-attention until one sequence is left.

    input (N, C, L)
    -> RevIN                                  per-signal window normalization
    -> SignalTokenizer per signal             (N, C, T, d)    channel-independent segment tokens, shared weights
    -> merge levels                           (N, ceil(C / 2), T, d_1) -> ... -> (N, 1, T, d_last)
    -> output queries                         (N, C, d_last)  one learnable query per output signal
    -> head                                   (N, C, H)
    -> RevIN denormalize

Each merge level pairs up its sequences; a pair (A, B) becomes one sequence through a bidirectional causal
cross-attention (A reads B and B reads A, averaged) followed by an FFN that maps the width d_in to the level's
width d_out. A sequence left without a partner (odd count) is carried to the next level through the same
residual projection. All pairs of a level share weights, so the pairing can change without changing shapes.

Pairing methods (`pairing`):
- "index", "corr", "random": a fixed tree (see index_tree, correlation_tree, random_tree).
- "soft": every merge node takes learnable softmax mixtures of the level's sequences as its two inputs;
  pairing_penalty() pushes the mixtures toward one-hot (negative normalized variance) and toward using every
  input once (usage penalty). Initialized from a fixed tree.
- "sampled": every signal has a learnable embedding; group embeddings are averaged up the tree and their cosine
  similarity scores candidate partners. Training samples a matching per window (log-probability kept in
  last_log_prob for a REINFORCE update); evaluation takes the greedy most likely matching.

Normalization notes:
- Pre-norm everywhere: both inputs of a merge are RMS-normalized separately, so neither sequence dominates by
  scale, and every level has its own norms because widths differ.
- The two directions of a merge are averaged, not summed, otherwise the residual stream would double in
  variance at every level (2^10 on Traffic).
- When the width grows, the residual goes through a projection initialized as identity plus zero padding.
- The last projection of each attention and FFN is scaled by 1 / sqrt(2 * levels) (GPT-2 style), so a deep
  tree starts close to the identity.
- A final RMSNorm closes the pre-norm stack before the output queries.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from model import CrossAttention, RevIN, SignalTokenizer

PAIRINGS = ("index", "corr", "random", "soft", "sampled")

# One list per level; each group holds 2 indices (merged) or 1 index (carried) into the previous level's
# sequences. A level's outputs are its groups in order.
Tree = list[list[tuple[int, ...]]]


def level_sizes(num_signals: int) -> list[int]:
    """Number of sequences entering each merge level."""
    sizes = []
    while num_signals > 1:
        sizes.append(num_signals)
        num_signals = -(-num_signals // 2)
    return sizes


def greedy_matching(scores: torch.Tensor) -> list[tuple[int, ...]]:
    """Pairs with the highest scores first (ties by index); an odd one out is carried as a 1-group at the end."""
    m = scores.size(0)
    rows, cols = torch.triu_indices(m, m, 1).tolist()
    values = scores[rows, cols]
    order = torch.argsort(values, descending=True, stable=True).tolist()
    used = [False] * m
    groups: list[tuple[int, ...]] = []
    for k in order:
        i, j = rows[k], cols[k]
        if not used[i] and not used[j]:
            used[i] = used[j] = True
            groups.append((i, j))
            if len(groups) == m // 2:
                break
    return groups + [(i,) for i in range(m) if not used[i]]


def index_tree(num_signals: int) -> Tree:
    """Adjacent indices: (0, 1), (2, 3), ... at every level."""
    tree = []
    for m in level_sizes(num_signals):
        tree.append([(i, i + 1) for i in range(0, m - 1, 2)] + ([(m - 1,)] if m % 2 else []))
    return tree


def correlation_tree(values: torch.Tensor) -> Tree:
    """Greedy pairing by |Pearson correlation| of first differences of the training split.

    values: (C, T), training data only, so no validation or test information leaks into the tree. First
    differences remove the shared level and daily profile that would make every pair look correlated. A
    merged group is represented by the mean of its two members, with a negatively correlated member flipped
    first so the pair does not cancel out.
    """
    x = values.double().diff(dim=1)
    tree: Tree = []
    while x.size(0) > 1:
        corr = torch.corrcoef(x).nan_to_num(0.0)
        level = greedy_matching(corr.abs())
        merged = []
        for group in level:
            if len(group) == 1:
                merged.append(x[group[0]])
            else:
                i, j = group
                sign = 1.0 if corr[i, j] >= 0 else -1.0
                merged.append((x[i] + sign * x[j]) / 2)
        x = torch.stack(merged)
        tree.append(level)
    return tree


def random_tree(num_signals: int, seed: int = 0) -> Tree:
    """Same greedy procedure as correlation_tree on random scores (a control for the pairing rule)."""
    generator = torch.Generator().manual_seed(seed)
    tree: Tree = []
    for m in level_sizes(num_signals):
        scores = torch.rand(m, m, generator=generator)
        tree.append(greedy_matching(scores + scores.T))
    return tree


def correlation_embedding(values: torch.Tensor, dim: int = 16) -> torch.Tensor:
    """(C, dim) embedding whose inner products approximate the correlation of first differences (top eigenvectors)."""
    corr = torch.corrcoef(values.double().diff(dim=1)).nan_to_num(0.0)
    eigenvalues, eigenvectors = torch.linalg.eigh(corr)  # ascending
    dim = min(dim, corr.size(0))
    embedding = eigenvectors[:, -dim:] * eigenvalues[-dim:].clamp(min=0).sqrt()
    return embedding.float()


class TreeMerge(nn.Module):
    """One merge level: bidirectional causal cross-attention between paired sequences, then a widening FFN."""

    def __init__(self, d_in: int, d_out: int, num_heads: int, dropout: float, num_levels: int) -> None:
        super().__init__()
        self.query_norm = nn.RMSNorm(d_in)
        self.memory_norm = nn.RMSNorm(d_in)
        self.attn = CrossAttention(d_in, num_heads, dropout)
        self.ffn_norm = nn.RMSNorm(d_in)
        ffn_out = nn.Linear(4 * d_in, d_out)
        self.ffn = nn.Sequential(nn.Linear(d_in, 4 * d_in), nn.GELU(), ffn_out)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = None
        if d_out != d_in:
            # Identity on the first d_in dimensions, zeros for the new ones: the residual keeps its scale
            self.res_proj = nn.Linear(d_in, d_out, bias=False)
            with torch.no_grad():
                self.res_proj.weight.zero_()
                self.res_proj.weight[: min(d_in, d_out), : min(d_in, d_out)] = torch.eye(min(d_in, d_out))
        with torch.no_grad():
            scale = 1 / math.sqrt(2 * num_levels)
            self.attn.out_proj.weight.mul_(scale)
            ffn_out.weight.mul_(scale)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.res_proj is None else self.res_proj(x)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # (B, T, d_in) x 2 -> (B, T, d_out)
        # Both directions in one attention call: queries [a; b] read memories [b; a]. Paired sequences are
        # time-aligned, so the attention is causal (position t reads the partner's positions <= t).
        queries = self.query_norm(torch.cat([a, b]))
        memories = self.memory_norm(torch.cat([b, a]))
        ab, ba = self.attn(queries, memories, is_causal=True).chunk(2)
        x = 0.5 * (a + b) + self.dropout(0.5 * (ab + ba))
        return self.residual(x) + self.dropout(self.ffn(self.ffn_norm(x)))


@torch.compiler.disable
def sample_matching(
    embedding: torch.Tensor, temperature: float, greedy: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Sequentially sample a perfect matching per window from cosine similarities of group embeddings.

    embedding: (N, M, r). The first unmatched sequence (by position) picks its partner from a softmax over the
    remaining ones; greedy=True takes the most likely partner instead. Returns first members (N, P), second
    members (N, P), the carried index (N,) or None, and the log-probability of the matching (N,).
    """
    n, m, _ = embedding.shape
    unit = F.normalize(embedding.float(), dim=-1)
    scores = unit @ unit.transpose(1, 2) / temperature  # (N, M, M)
    rows = torch.arange(n, device=embedding.device)
    available = torch.ones(n, m, dtype=torch.bool, device=embedding.device)
    log_prob = embedding.new_zeros(n, dtype=torch.float32)
    firsts, seconds = [], []
    for _ in range(m // 2):
        i = available.float().argmax(dim=1)  # first unmatched position
        available[rows, i] = False
        logits = scores[rows, i].masked_fill(~available, float("-inf"))
        log_probs = logits.log_softmax(dim=-1)
        if greedy:
            j = logits.argmax(dim=-1)
        else:
            gumbel = -torch.empty_like(logits).exponential_().log()
            j = (logits.detach() + gumbel).argmax(dim=-1)
        log_prob = log_prob + log_probs[rows, j]
        available[rows, j] = False
        firsts.append(i)
        seconds.append(j)
    carry = available.float().argmax(dim=1) if m % 2 else None
    return torch.stack(firsts, 1), torch.stack(seconds, 1), carry, log_prob


class TreeSignalformer(nn.Module):
    """See the module docstring. tree is required for the fixed pairings "corr" and "random" and initializes
    "soft"; pair_embedding (C, r) initializes "sampled"."""

    def __init__(
        self,
        in_channels: int,
        in_len: int,
        d_model: int,
        kernel_size: int | tuple[int, ...],
        output_length: int,
        num_heads: int,
        tree_widths: tuple[int, ...] = (),
        pairing: str = "index",
        tree: Tree | None = None,
        pair_embedding: torch.Tensor | None = None,
        ma_windows: tuple[int, ...] = (4, 16, 48),
        n_rank: int = 1,
        dropout: float = 0.1,
        conv_stride: int = 1,
        revin: bool = True,
        front_end: str = "ma",
        bank_filters: int = 8,
        raw_skip: bool = False,
    ) -> None:
        super().__init__()
        assert pairing in PAIRINGS, f"pairing must be one of {PAIRINGS}"
        assert in_channels > 1, "the tree needs at least two signals"
        self.pairing = pairing
        self.revin = RevIN(in_channels) if revin else None
        self.tokenizer = SignalTokenizer(
            1, d_model, kernel_size, ma_windows, n_rank, 0, conv_stride, True, front_end, bank_filters, raw_skip
        )
        self.sizes = level_sizes(in_channels)
        num_levels = len(self.sizes)
        widths = [d_model] + [tree_widths[min(i, len(tree_widths) - 1)] if tree_widths else d_model
                              for i in range(num_levels)]
        self.merges = nn.ModuleList(
            TreeMerge(widths[i], widths[i + 1], num_heads, dropout, num_levels) for i in range(num_levels)
        )
        if pairing in ("index", "corr", "random", "soft"):
            if tree is None:
                assert pairing in ("index", "soft"), f"pairing '{pairing}' needs a precomputed tree"
                tree = index_tree(in_channels)
            assert [sum(len(g) for g in level) for level in tree] == self.sizes, "tree does not match in_channels"
        self.tree = tree
        if pairing == "soft":
            assert tree is not None
            # Selector logits (outputs x inputs) per level, starting close to the given tree
            self.query_logits = nn.ParameterList()
            self.memory_logits = nn.ParameterList()
            for level, m in zip(tree, self.sizes):
                q = torch.randn(len(level), m) * 0.01
                kv = torch.randn(len(level), m) * 0.01
                for p, group in enumerate(level):
                    q[p, group[0]] += 4.0
                    kv[p, group[-1]] += 4.0  # a carried sequence starts by merging with itself
                self.query_logits.append(nn.Parameter(q))
                self.memory_logits.append(nn.Parameter(kv))
        if pairing == "sampled":
            init = pair_embedding if pair_embedding is not None else torch.randn(in_channels, 16)
            self.pair_embedding = nn.Parameter(init.clone().float())
            self.pair_temperature = 1.0  # annealed by the training loop
        self.last_log_prob: torch.Tensor | None = None
        width = widths[-1]
        self.final_norm = nn.RMSNorm(width)
        # Output: one learnable query per signal reads the merged sequence (it is all past data, so no mask)
        self.out_queries = nn.Parameter(torch.randn(in_channels, width) * 0.02)
        self.out_query_norm = nn.RMSNorm(width)
        self.out_attn = CrossAttention(width, num_heads, dropout, use_rope=False)
        self.head = nn.Sequential(
            nn.RMSNorm(width), nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, output_length)
        )

    def _merge(self, level: int) -> TreeMerge:
        merge = self.merges[level]
        assert isinstance(merge, TreeMerge)
        return merge

    def _merge_fixed(self, level: int, x: torch.Tensor) -> torch.Tensor:
        # x: (N, M, T, d) -> (N, M', T, d_out)
        assert self.tree is not None
        n, _, t, d = x.shape
        groups = self.tree[level]
        pairs = [g for g in groups if len(g) == 2]
        carried = [g[0] for g in groups if len(g) == 1]
        merge = self._merge(level)
        out = []
        if pairs:
            a = x[:, [g[0] for g in pairs]].reshape(-1, t, d)
            b = x[:, [g[1] for g in pairs]].reshape(-1, t, d)
            out.append(merge(a, b).view(n, len(pairs), t, -1))
        if carried:
            out.append(merge.residual(x[:, carried]))
        return torch.cat(out, dim=1)

    def _merge_soft(self, level: int, x: torch.Tensor) -> torch.Tensor:
        n, _, t, d = x.shape
        wq = self.query_logits[level].softmax(dim=-1)  # (P, M)
        wkv = self.memory_logits[level].softmax(dim=-1)
        a = torch.einsum("pm,nmtd->nptd", wq.to(x.dtype), x)
        b = torch.einsum("pm,nmtd->nptd", wkv.to(x.dtype), x)
        p = wq.size(0)
        return self._merge(level)(a.reshape(-1, t, d), b.reshape(-1, t, d)).view(n, p, t, -1)

    def _merge_sampled(
        self, level: int, x: torch.Tensor, embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, _, t, d = x.shape
        greedy = not self.training
        # torch.compiler.disable hides the signature from type checkers
        first, second, carry, log_prob = sample_matching(embedding, self.pair_temperature, greedy)  # pyright: ignore[reportCallIssue]
        rows = torch.arange(n, device=x.device)[:, None]
        merge = self._merge(level)
        merged = merge(x[rows, first].reshape(-1, t, d), x[rows, second].reshape(-1, t, d))
        out = [merged.view(n, first.size(1), t, -1)]
        new_embedding = [(embedding[rows, first] + embedding[rows, second]) / 2]
        if carry is not None:
            out.append(merge.residual(x[rows, carry[:, None]]))
            new_embedding.append(embedding[rows, carry[:, None]])
        return torch.cat(out, dim=1), torch.cat(new_embedding, dim=1), log_prob

    def forward(self, input: torch.Tensor, *unused) -> torch.Tensor:  # (N, C, L) -> (N, C, H)
        n, c, length = input.shape
        stats = None
        if self.revin is not None:
            input, stats = self.revin.normalize(input)
        tokens = self.tokenizer(input.reshape(n * c, 1, length))  # (N * C, T, d)
        x = tokens.view(n, c, tokens.size(1), -1)  # (N, C, T, d)
        log_prob = None
        embedding = self.pair_embedding.expand(n, -1, -1) if self.pairing == "sampled" else None
        for level in range(len(self.merges)):
            if self.pairing == "soft":
                x = self._merge_soft(level, x)
            elif self.pairing == "sampled":
                assert embedding is not None
                x, embedding, level_log_prob = self._merge_sampled(level, x, embedding)
                log_prob = level_log_prob if log_prob is None else log_prob + level_log_prob
            else:
                x = self._merge_fixed(level, x)
        self.last_log_prob = log_prob
        merged = self.final_norm(x[:, 0])  # (N, T, d_last)
        queries = self.out_queries.expand(n, -1, -1)  # (N, C, d_last)
        y = queries + self.out_attn(self.out_query_norm(queries), merged)
        output = self.head(y)  # (N, C, H)
        if self.revin is not None and stats is not None:
            output = self.revin.denormalize(output, stats)
        return output

    def penalty(self) -> torch.Tensor:
        return self.tokenizer.penalty()

    def pairing_penalty(self) -> torch.Tensor:
        """Soft pairing only: -(normalized variance of the selector weights) + usage penalty, averaged over levels.

        For softmax weights over M inputs the variance is maximal (1/M)(1 - 1/M) for a one-hot row, so the first
        term lies in [-1, 0]. The usage term pushes every input to be selected about once over the level's
        2 * P query and memory slots.
        """
        if self.pairing != "soft":
            return self.final_norm.weight.new_zeros(())
        total = self.final_norm.weight.new_zeros(())
        for q_logits, kv_logits in zip(self.query_logits, self.memory_logits):
            p, m = q_logits.shape
            wq, wkv = q_logits.softmax(dim=-1), kv_logits.softmax(dim=-1)
            max_var = (1 / m) * (1 - 1 / m)
            concentration = torch.cat([wq.var(dim=-1, unbiased=False), wkv.var(dim=-1, unbiased=False)]) / max_var
            usage = (wq.sum(dim=0) + wkv.sum(dim=0) - 2 * p / m).pow(2).mean()
            total = total - concentration.mean() + usage
        return total / len(self.query_logits)
