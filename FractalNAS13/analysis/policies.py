"""Search policies on FractalNAS-13, beyond the as-run GA.

`run` is the vectorized GA of fractalbench with the extra knobs the policy experiments need:
a warm-started initial population, per-gene mutation weights, a gene permutation for one-point
crossover, uniform crossover, frozen genes, and a noise level on the selection signal.
`screen_then_measure` and `random_measure` are the two policies that replace search altogether,
and the three functions at the end of this file are the NAS baselines the GA is measured
against: regularized evolution, best-improvement local search, and a predictor-guided search
whose acquisition model is genuinely refitted on what has been measured.
"""
import numpy as np
from surrogate import CARD, NG


def _tournament(rng, P, f, n, k, R, pop):
    c = rng.integers(0, pop, (R, n, k))
    fv = np.take_along_axis(f[:, None, :].repeat(n, 1), c, 2)
    w = np.take_along_axis(c, fv.argmax(2)[:, :, None], 2)[:, :, 0]
    return np.take_along_axis(P, w[:, :, None].repeat(NG, 2), 1)


def run(bench, R=400, pop=20, gens=26, elite=5, k=3, pm=0.08, seed=1, order=None,
        uniform=False, gene_w=None, freeze=None, init=None, noise=0.0):
    """R replicates of a (gens x pop) GA. Returns per-replicate best / top-10 / mean."""
    rng = np.random.default_rng(seed)
    order = np.arange(NG) if order is None else np.asarray(order)
    inv = np.argsort(order)
    fz = np.array([freeze.get(g, -1) if freeze else -1 for g in range(NG)]) \
        if isinstance(freeze, dict) else np.full(NG, -1)
    P = rng.integers(0, CARD, (R, pop, NG))
    if init is not None:
        m = min(len(init), pop)
        P[:, :m] = np.asarray(init)[:m]
    for gi in np.where(fz >= 0)[0]:
        P[..., gi] = fz[gi]
    pmg = np.full(NG, pm) if gene_w is None else np.clip(pm * np.asarray(gene_w, float), 0, 1)
    best = np.full(R, -np.inf)
    sampled = []
    for g in range(gens):
        t = bench.score(P.reshape(-1, NG)).reshape(R, pop)
        f = t + rng.normal(0, noise, t.shape) if noise else t
        best = np.maximum(best, t.max(1))
        sampled.append(t)
        if g == gens - 1:
            break
        o = np.argsort(-f, 1)
        new = np.empty_like(P)
        new[:, :elite] = np.take_along_axis(P, o[:, :elite, None].repeat(NG, 2), 1)
        n = pop - elite
        A = _tournament(rng, P, f, n, k, R, pop)
        B = _tournament(rng, P, f, n, k, R, pop)
        if uniform:
            ch = np.where(rng.random((R, n, NG)) < 0.5, A, B)
        else:
            pt = rng.integers(1, NG, (R, n, 1))
            ch = np.where(np.arange(NG)[None, None, :] < pt,
                          A[:, :, order], B[:, :, order])[:, :, inv]
        mm = rng.random((R, n, NG)) < pmg[None, None, :]
        sh = rng.integers(1, np.broadcast_to(CARD, (R, n, NG)))
        new[:, elite:] = np.where(mm, (ch + sh) % CARD, ch)
        P = new
        for gi in np.where(fz >= 0)[0]:
            P[..., gi] = fz[gi]
    S = np.concatenate(sampled, 1)
    return dict(best=best, top10=np.sort(S, 1)[:, -10:].mean(1), mean=S.mean(1),
                n_measured=gens * pop)


def screen_then_measure(bench, rng, n_screen, n_measure, chunk=50000):
    """The policy that replaces search entirely: screen `n_screen` free, measure the best.

    Screening is chunked: the surrogate's design matrix is 1,572 columns wide, so a
    half-million-candidate screen is scored in blocks and only the running best kept.
    """
    keep_P, keep_s = None, None
    left = n_screen
    while left > 0:
        n = min(chunk, left)
        left -= n
        c = rng.integers(0, CARD, (n, NG))
        v = bench.score(c, tabular=False)
        if keep_P is None:
            keep_P, keep_s = c, v
        else:
            keep_P = np.concatenate([keep_P, c]); keep_s = np.concatenate([keep_s, v])
        if len(keep_s) > 4 * n_measure:
            o = np.argsort(-keep_s)[:n_measure]
            keep_P, keep_s = keep_P[o], keep_s[o]
    sel = keep_P[np.argsort(-keep_s)[:n_measure]]
    v = bench.score(sel)
    return float(v.max()), float(np.sort(v)[-10:].mean())


def random_measure(bench, rng, n_measure):
    """The floor: measure `n_measure` architectures drawn uniformly at random."""
    v = bench.score(rng.integers(0, CARD, (n_measure, NG)))
    return float(v.max()), float(np.sort(v)[-10:].mean())


# ---------------------------------------------------------------- competitive baselines
# Three search methods from the NAS and evolutionary literature, all run on FractalNAS-13 at
# the same budget of measured architectures as the GA, so that the GA is compared against
# current practice rather than against random search alone. Each returns per-replicate arrays
# so the comparisons can be tested as paired blocks.

def regularized_evolution(bench, R=400, n_measured=520, pop=20, sample=10, seed=1):
    """Aging (regularized) evolution: tournament over a sample, mutate one gene, kill oldest.

    The NAS baseline of Real et al.; the only structural difference from the study's GA is
    that the oldest individual is discarded rather than the worst, which is what the method
    is named for.
    """
    rng = np.random.default_rng(seed)
    P = rng.integers(0, CARD, (R, pop, NG))
    f = bench.score(P.reshape(-1, NG)).reshape(R, pop)
    best = f.max(1)
    age = np.tile(np.arange(pop), (R, 1))
    clock = pop
    while clock < n_measured:
        c = rng.integers(0, pop, (R, sample))
        w = np.take_along_axis(c, np.take_along_axis(f, c, 1).argmax(1)[:, None], 1)[:, 0]
        child = np.take_along_axis(P, w[:, None, None].repeat(NG, 2), 1)[:, 0].copy()
        gi = rng.integers(0, NG, R)
        sh = rng.integers(1, CARD[gi])
        child[np.arange(R), gi] = (child[np.arange(R), gi] + sh) % CARD[gi]
        v = bench.score(child)
        oldest = age.argmin(1)
        P[np.arange(R), oldest] = child
        f[np.arange(R), oldest] = v
        age[np.arange(R), oldest] = clock
        best = np.maximum(best, v)
        clock += 1
    return dict(best=best, n_measured=n_measured)


def local_search(bench, R=400, n_measured=520, seed=1):
    """Best-improvement local search over the 1-gene neighbourhood, restarting at an optimum.

    White et al. report local search as a strong NAS baseline; the neighbourhood here is every
    genotype one gene value away, 37 neighbours on this grid.
    """
    rng = np.random.default_rng(seed)
    nbr = [(g, v) for g in range(NG) for v in range(int(CARD[g]))]
    best = np.full(R, -np.inf)
    cur = rng.integers(0, CARD, (R, NG))
    curv = bench.score(cur)
    best = np.maximum(best, curv)
    used = np.ones(R, int)
    while used.max() < n_measured:
        cand = np.repeat(cur[:, None, :], len(nbr), 1)
        for j, (g, v) in enumerate(nbr):
            cand[:, j, g] = v
        vals = bench.score(cand.reshape(-1, NG)).reshape(R, len(nbr))
        used = used + len(nbr)
        best = np.maximum(best, vals.max(1))
        j = vals.argmax(1)
        nxt = np.take_along_axis(cand, j[:, None, None].repeat(NG, 2), 1)[:, 0]
        nxtv = np.take_along_axis(vals, j[:, None], 1)[:, 0]
        stuck = nxtv <= curv
        cur = np.where(stuck[:, None], rng.integers(0, CARD, (R, NG)), nxt)
        curv = np.where(stuck, bench.score(cur), nxtv)
        used = used + stuck.astype(int)
        best = np.maximum(best, curv)
    return dict(best=best, n_measured=int(used.max()))


def predictor_guided(bench, R=200, n_measured=520, n_init=60, batch=20, pool=20000,
                     seed=1, n_models=5):
    """Predictor-guided search in the style of BANANAS: fit an ensemble on what has been
    measured, propose the pool's most promising candidates, measure them, refit.

    The ensemble is a ridge on gene main effects only, bootstrapped; it is deliberately a
    different and weaker model class than the benchmark's own, because an acquisition model
    drawn from the benchmark's own hypothesis class would be scoring itself. The acquisition
    is an independent-Thompson draw from the ensemble mean and spread.
    """
    from surrogate import OFF, NMAIN
    rng = np.random.default_rng(seed)

    def mains(C):
        A = np.zeros((len(C), NMAIN), np.float32)
        A[np.arange(len(C))[:, None], OFF[None, :] + C] = 1.0
        return A

    best = np.full(R, -np.inf)
    for r in range(R):
        X = rng.integers(0, CARD, (n_init, NG))
        y = bench.score(X)
        best[r] = y.max()
        while len(y) < n_measured:
            A = mains(X)
            preds = []
            for _ in range(n_models):
                idx = rng.integers(0, len(y), len(y))
                Ab, yb = A[idx], y[idx]
                G = Ab.T @ Ab + 1.0 * np.eye(NMAIN)
                b = yb.mean()
                w = np.linalg.solve(G, Ab.T @ (yb - b))
                preds.append((w, b))
            C = rng.integers(0, CARD, (pool, NG))
            AC = mains(C)
            M = np.stack([AC @ w + b for w, b in preds])
            acq = M.mean(0) + rng.normal(0, 1, pool) * M.std(0)
            take = min(batch, n_measured - len(y))
            sel = C[np.argsort(-acq)[:take]]
            v = bench.score(sel)
            X = np.concatenate([X, sel]); y = np.concatenate([y, v])
            best[r] = max(best[r], v.max())
    return dict(best=best, n_measured=n_measured)
