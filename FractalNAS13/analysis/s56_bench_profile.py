"""Profile the benchmark itself: fidelity, cost, resolution, and the segmentation sweep.

The three result sections of this paper all read differences off FractalNAS-13, so the
benchmark's own properties have to be measured before any of them is quoted:

  fidelity     how well the surrogate orders architectures it was not fitted on, and how that
               fidelity grows with the number of labels it is fitted on (from s16_surrogate.py,
               plus the held-out predictions in data/surrogate_predictions.csv for Fig. 1a);
  cost         wall-clock of one benchmarked search against the GPU-hours one real run of the
               same search took, read from the pipeline's own logs;
  resolution   the standard error of a configuration's benchmark score as a function of the
               number of replicates, which is what decides how small a difference the benchmark
               can resolve -- and how it compares with the run-to-run spread of the three real
               runs per arm the original study could afford;
  segmentation the equal-budget sweep over the number of segments a run is chopped into, which
               is the one schedule question the benchmark can answer and the real record
               cannot, because the record holds one segment count per arm;
  coverage     the share of fitness queries a search serves from a measured architecture rather
               than from the surrogate.

    python3 analysis/s56_bench_profile.py          # -> results/bench_profile.json
"""
import csv
import json
import os
import time

import numpy as np

import common
import fractalbench as fb
from surrogate import NG

DSETS = ['cifar10', 'cifar100']
#: replicate counts at which the benchmark's resolution is reported
REPS = [1, 3, 10, 30, 100, 300, 1000]
POP = 20
BUDGET = 520


def _real_run_hours():
    """GPU-hours per real search run, per arm, from the released run statistics."""
    st = json.load(open(os.path.join(common.RES, 'stats.json')))
    out = {}
    for ds in DSETS:
        w = st[ds]['metrics'].get('wall_h')
        if not w:
            continue
        out[ds] = {arm: float(np.mean(v)) for arm, v in w['per'].items()}
    return out


def _heldout_scatter(ds, cap=4000):
    """Leave-one-run-out predictions against the measured accuracy, for Fig. 1(a)."""
    x, y = [], []
    path = os.path.join(common.HERE, '..', 'data', 'surrogate_predictions.csv')
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r['dataset'] != ds or not r['surrogate_loo'] or not r['acc_epoch3_true']:
                continue
            x.append(float(r['surrogate_loo']))
            y.append(float(r['acc_epoch3_true']))
    x, y = np.array(x), np.array(y)
    if len(x) > cap:                      # thin the scatter, keep the statistics on all of it
        idx = np.random.default_rng(0).choice(len(x), cap, replace=False)
        return x, y, x[idx], y[idx]
    return x, y, x, y


def main():
    sur = json.load(open(os.path.join(common.RES, 'surrogate.json')))
    hours = _real_run_hours()
    out = {'reps': REPS, 'pop': POP,
           'budget': BUDGET, 'real_run_hours': hours}

    for ds in DSETS:
        b = fb.Bench(ds)
        o = out[ds] = {'n_fit': b.n_fit, 'n_features': sur[ds]['n_features'],
                       'rho_heldout': sur[ds]['rho']['surrogate'],
                       'rho_epoch1': sur[ds]['rho']['epoch1'],
                       'mae': sur[ds]['mae'],
                       'sample_efficiency': sur[ds]['sample_efficiency']}

        # -- held-out scatter, and the statistics over all of it
        xa, ya, xs, ys = _heldout_scatter(ds)
        o['scatter'] = {'pred': [round(float(v), 3) for v in xs],
                        'true': [round(float(v), 3) for v in ys], 'n_all': int(len(xa))}

        # -- cost: one benchmarked search of the same size as one real run.  Wall-clock is
        # machine- and load-dependent, so each timing is the median of five repetitions and
        # the paper quotes it to one significant figure with an "about".
        def timed(R, reps=5):
            ts = []
            for i in range(reps):
                t0 = time.perf_counter()
                fb.run_ga(b, n_seg=1, gens=26, pop=POP, R=R, seed=11 + i)
                ts.append(time.perf_counter() - t0)
            return float(np.median(ts))
        o['sec_per_search'] = timed(1)
        o['sec_per_1000_searches'] = timed(1000, reps=3)
        if ds in hours:
            o['real_hours_mean'] = float(np.mean(list(hours[ds].values())))
            o['speedup_1000x'] = (o['real_hours_mean'] * 3600.0) / o['sec_per_1000_searches']

        # -- resolution: s.e. of the best-score as the replicate count grows
        r = fb.run_ga(b, n_seg=1, gens=26, pop=POP, R=max(REPS), seed=3)
        best = np.asarray(r['best'], float)
        rng = np.random.default_rng(5)
        o['resolution'] = {}
        for R in REPS:
            se = [float(rng.choice(best, R, replace=True).mean()) for _ in range(400)]
            o['resolution'][str(R)] = {'se': float(np.std(se, ddof=1)),
                                       'mean': float(np.mean(se))}
        o['replicate_sd'] = float(best.std(ddof=1))

        # -- what three real runs per arm can resolve, on the same metric
        st = json.load(open(os.path.join(common.RES, 'stats.json')))
        per = st[ds]['metrics']['best1']['per']
        o['real_arm_sd'] = {arm: float(np.std(v, ddof=1)) for arm, v in per.items()
                            if len(v) > 1}
        o['real_baseline_se'] = float(np.std(per['baseline'], ddof=1) / np.sqrt(3))

        # -- the segmentation sweep is computed by s19_levers.py under the same seed and
        # replicate count; it is carried here so that the benchmark profile is one file
        lv = json.load(open(os.path.join(common.RES, 'levers.json')))[ds]['segments']
        o['segmentation'] = {k.replace('_measured', ''): dict(v)
                             for k, v in lv.items() if k.endswith('_measured')}
        o['segmentation_predictor'] = {k.replace('_predictor', ''): dict(v)
                                       for k, v in lv.items() if k.endswith('_predictor')}
        v = o['segmentation']
        o['segmentation_cost'] = {
            'best_setting': max(v, key=lambda k: v[k]['best']),
            'worst_setting': min(v, key=lambda k: v[k]['best']),
            'spread': max(v[k]['best'] for k in v) - min(v[k]['best'] for k in v),
            'one_vs_five': v['1x26']['best'] - v['5x5']['best']}

        # -- coverage: how often a query is served from a measured architecture
        P = np.random.default_rng(2).integers(0, 5, (4000, NG))
        hit = sum(1 for row in P if tuple(int(x) for x in row) in b.table)
        rr = fb.run_ga(b, n_seg=1, gens=26, pop=POP, R=200, seed=9, record_genes=True)
        o['coverage'] = {'random_draws': hit / len(P),
                         'unique_per_search': float(np.mean(rr['n_unique']))}

    json.dump(out, open(os.path.join(common.RES, 'bench_profile.json'), 'w'), indent=1)
    _tex(out)
    print('wrote analysis/results/bench_profile.json and tables/tab_profile.tex')
    for ds in DSETS:
        o = out[ds]
        print(f"  {ds}: rho={o['rho_heldout']:.3f}  1 search={o['sec_per_search']*1e3:.0f} ms"
              f"  1000 searches={o['sec_per_1000_searches']:.1f} s"
              f"  segmentation spread={o['segmentation_cost']['spread']:.2f} pts")


def _tex(out):
    """Table I of the main paper, written from the same numbers Fig. 1 is drawn from.

    The cost rows are wall-clock and therefore machine-dependent, which is exactly why they are
    generated rather than typed: a rerun on another machine moves the table and the figure
    together, and `s60_verify_claims.py` re-checks the prose against the same file.
    """
    a, b = out['cifar10'], out['cifar100']
    ms = a['sec_per_search'] * 1e3
    s1k = a['sec_per_1000_searches']
    L = [r'\begin{tabular}{l rr}', r'\toprule',
         r'& \textbf{CIFAR-10} & \textbf{CIFAR-100} \\', r'\midrule',
         r'\multicolumn{3}{l}{\textit{Fidelity (held out, against measured 3-epoch accuracy)}} \\',
         rf"Spearman $\rho$, genotype model & {a['rho_heldout']:.3f} & {b['rho_heldout']:.3f} \\",
         rf"Spearman $\rho$, one measured training epoch & {a['rho_epoch1']:.3f} "
         rf"& {b['rho_epoch1']:.3f} \\"]
    sur = json.load(open(os.path.join(common.RES, 'surrogate.json')))
    L += [rf"Spearman $\rho$, genotype model $+$ one epoch "
          rf"& {sur['cifar10']['rho']['surrogate_plus_epoch1']:.3f} "
          rf"& {sur['cifar100']['rho']['surrogate_plus_epoch1']:.3f} \\",
          rf"Mean absolute error (pts) & {a['mae']:.2f} & {b['mae']:.2f} \\",
          rf"Architectures fitted on & {a['n_fit']:,} & {b['n_fit']:,} \\".replace(',', '{,}'),
          r'\midrule',
          r'\multicolumn{3}{l}{\textit{Cost of one 26-generation, 520-evaluation search}} \\',
          rf"On the benchmark & \multicolumn{{2}}{{c}}{{$\approx${ms:.0f}\,ms}} \\",
          rf"1{{,}}000 replicates & \multicolumn{{2}}{{c}}{{$\approx${s1k:.1f}\,s}} \\",
          rf"Trained for real (mean over 34 runs) "
          rf"& \multicolumn{{2}}{{c}}{{{a['real_hours_mean']:.1f} GPU-hours}} \\",
          r'\midrule',
          r'\multicolumn{3}{l}{\textit{Resolution: s.e.\ of the best architecture found (pts)}} \\']
    for R in (1, 100, 1000):
        lab = '1 replicate' if R == 1 else f'{R:,}'.replace(',', '{,}') + ' replicates'
        L.append(rf"{lab} & {a['resolution'][str(R)]['se']:.2f} "
                 rf"& {b['resolution'][str(R)]['se']:.2f} \\")
    L += [rf"3 real runs (the baseline arm) & {a['real_baseline_se']:.2f} "
          rf"& {b['real_baseline_se']:.2f} \\",
          r'\bottomrule', r'\end{tabular}']
    open(os.path.join(common.HERE, '..', 'tables', 'tab_profile.tex'),
         'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
