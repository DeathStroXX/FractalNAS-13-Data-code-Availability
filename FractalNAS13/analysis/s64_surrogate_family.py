"""Do the recommendations survive a change of surrogate?

The benchmark's landscape is a ridge on gene indicators and gene-pair interactions, which is
smooth by construction. A reader is entitled to ask whether the settings it recommends are
properties of the search space or of that smoothness. The record answers it without new runs:
the same architectures support surrogates from other model families, and the settings sweep
can be repeated on each of them.

Three landscapes are compared:

  ridge      the benchmark's own model, a ridge on indicators plus all gene pairs;
  forest     a random forest on the 13 raw gene codes, which represents interactions of any
             order and is not smooth in the gene level at all;
  boosted    gradient-boosted trees on the same codes, a third hypothesis class.

Each is refitted on the same architectures, its held-out fidelity is measured on the same
leave-one-run-out split, and the four setting families of the main paper's Table IV are swept
on it. What the comparison reports is not whether the three agree on an absolute score -- they
cannot, being different models of the same points -- but whether they agree on the recommended
value of each setting and on the sign of the gain against the value the pipeline fixed, which
is the only thing the paper claims.

Writes results/surrogate_family.json, tables/tab_family.tex.
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

import common
import fractalbench as fb
import surrogate as sg

R = 200
FAMILIES = [('tournament size $k$', 'k', 3, [2, 3, 5, 7, 10]),
            ('mutation rate $p_m$', 'pm', 0.08, [0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4]),
            ('population size', 'pop', 20, [10, 20, 40, 52]),
            ('elite count', 'elite', 5, [0, 1, 2, 4, 5, 8])]
#: an equal budget of 520 evaluated individuals, as in the main sweep
POPGENS = {10: 52, 20: 26, 40: 13, 52: 10}


class ModelBench(fb.Bench):
    """A benchmark whose landscape is any regressor over the 13 gene codes.

    Inherits the tabular/surrogate query of `fb.Bench`, so a genotype that was measured still
    returns its measured accuracy and only the rest of the grid changes model.
    """

    def __init__(self, ds, make_model):
        super().__init__(ds)
        P = np.array([sg.codes_of(r) for r in self.rows])
        y = np.array([r['acc3'] for r in self.rows], float)
        self.model = make_model().fit(P, y)
        self.P, self.y = P, y


def _models():
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    return {
        'forest': lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                                n_jobs=-1, random_state=0),
        'boosted': lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                                         random_state=0),
    }


def _loo_rho(ds, make_model=None):
    """Leave-one-run-out Spearman, on the split the benchmark's own fidelity is measured on.

    `make_model=None` evaluates the benchmark's ridge through the same loop, so that the three
    landscapes are compared on identical folds rather than on each model's own convention.
    """
    rows = [r for r in common.gene_table()
            if r['dataset'] == ds and r['acc3'] is not None]
    P = np.array([sg.codes_of(r) for r in rows])
    y = np.array([r['acc3'] for r in rows], float)
    run = np.array([r['run'] for r in rows])
    gf = sg.GramFitter(P, y) if make_model is None else None
    pred = np.zeros(len(y))
    for rid in sorted(set(run.tolist())):
        te = run == rid
        if make_model is None:
            pred[te] = gf.fit_mask(~te)(np.flatnonzero(te))
        else:
            pred[te] = make_model().fit(P[~te], y[~te]).predict(P[te])
    return float(spearmanr(pred, y).statistic)


def _query_ms(bench, n=20000, reps=3, seed=0):
    """Milliseconds to score one batch of `n` genotypes: what a GA generation costs."""
    import time
    P = np.random.default_rng(seed).integers(0, fb.CARD, (n, fb.NG))
    bench.score(P, tabular=False)
    t0 = time.time()
    for _ in range(reps):
        bench.score(P, tabular=False)
    return 1e3 * (time.time() - t0) / reps


def _sweep(bench):
    o = {}
    for lab, key, dflt, values in FAMILIES:
        cells = {}
        for v in values:
            kw = dict(n_seg=1, gens=26)
            if key == 'pop':
                kw.update(pop=v, gens=POPGENS[v])
            else:
                kw.update(fb.setting_kw(key, v))
            r = fb.run_ga(bench, R=R, seed=1, **kw)
            cells[str(v)] = float(r['best'].mean())
        bestv = max(values, key=lambda v: cells[str(v)])
        o[lab] = dict(cells=cells, pipeline=str(dflt), best=str(bestv),
                      gain=cells[str(bestv)] - cells[str(dflt)])
    return o


def main():
    out = {}
    mk = _models()
    for ds in common.DATASETS:
        o = {'landscapes': {}}
        ridge = fb.Bench(ds)
        o['landscapes']['ridge'] = dict(sweep=_sweep(ridge), rho_loo=_loo_rho(ds),
                                        query_ms_20k=_query_ms(ridge))
        print(f'  {ds} ridge done', flush=True)
        for name, make in mk.items():
            b = ModelBench(ds, make)
            o['landscapes'][name] = dict(sweep=_sweep(b), rho_loo=_loo_rho(ds, make),
                                         query_ms_20k=_query_ms(b))
            print(f'  {ds} {name} done (loo rho {o["landscapes"][name]["rho_loo"]:.3f})',
                  flush=True)
        agree_best, agree_sign, n = 0, 0, 0
        for lab, _, _, _ in FAMILIES:
            ref = o['landscapes']['ridge']['sweep'][lab]
            for name in mk:
                alt = o['landscapes'][name]['sweep'][lab]
                n += 1
                agree_best += int(alt['best'] == ref['best'])
                agree_sign += int(np.sign(alt['gain']) == np.sign(ref['gain']))
        o['agreement'] = dict(n=n, same_recommended_value=agree_best,
                              same_sign_of_gain=agree_sign)
        out[ds] = o
    #: the replicate count is quoted in main Table III's caption, which is what explains a
    #: level inside a flat region differing from the 1,000-replicate sweep of Table V. It
    #: is written beside the per-dataset entries rather than into them, so nothing that
    #: iterates the datasets sees it.
    json.dump(dict(out, replicates=R),
              open(os.path.join(common.RES, 'surrogate_family.json'), 'w'), indent=1)
    _tex(out)

    for ds, o in out.items():
        print(f'\n== {ds}')
        for name, L in o['landscapes'].items():
            print(f"   {name:8s} (held-out rho {L['rho_loo']:.3f}, "
                  f"{L['query_ms_20k']:.0f} ms per 20k queries)")
            for lab, c in L['sweep'].items():
                print(f"      {lab:24s} best {c['best']:>5s} "
                      f"(pipeline {c['pipeline']}), gain {c['gain']:+.2f}")
        a = o['agreement']
        print(f"   agreement with the ridge: {a['same_recommended_value']}/{a['n']} on the "
              f"recommended value, {a['same_sign_of_gain']}/{a['n']} on the sign of the gain")


def _tex(out):
    names = ['ridge', 'forest', 'boosted']
    nice = {'ridge': 'Ridge $+$ gene pairs (the benchmark)', 'forest': 'Random forest',
            'boosted': 'Gradient-boosted trees'}
    L = [r'\begin{tabular}{l c cccc cccc}', r'\toprule',
         r'& & \multicolumn{4}{c}{CIFAR-10} & \multicolumn{4}{c}{CIFAR-100} \\',
         r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}',
         r'\textbf{Landscape} & \textbf{held-out} $\rho$ & $k$ & $p_m$ & pop. & elite '
         r'& $k$ & $p_m$ & pop. & elite \\', r'\midrule']
    for nm in names:
        cells = []
        rho = ' / '.join(f"{out[d]['landscapes'][nm]['rho_loo']:.3f}"
                         for d in ('cifar10', 'cifar100'))
        for ds in ('cifar10', 'cifar100'):
            sw = out[ds]['landscapes'][nm]['sweep']
            for lab, _, _, _ in FAMILIES:
                c = sw[lab]
                cells.append(rf"{c['best']} (${c['gain']:+.2f}$)")
        L.append(f"{nice[nm]} & {rho} & " + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(common.HERE, '..', 'tables', 'tab_family.tex'),
         'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
