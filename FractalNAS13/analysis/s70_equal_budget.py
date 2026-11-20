"""The real-run contrasts at an equal evaluation budget, and how sure the outcome-level
agreement of the benchmark is.

Two questions a reader of Section IV-A is entitled to ask, both answered from the released
logs without a new run.

  budget     The arms did not evaluate the same number of unique architectures. A run's
             outcome is the best 3-epoch accuracy it found, which can only rise with the
             number of architectures it trains, so an arm with a larger budget is favoured.
             Each run is therefore cut at the same number N of unique architectures, in the
             order the run first evaluated them, with N the smallest run of the dataset, and
             the three contrasts of the record are recomputed on those truncated outcomes.

  agreement  The Spearman agreement between held-out predicted and measured run outcomes
             (s63_real_validation.py) is a statistic of 17 runs per dataset. Its 95% interval
             is the percentile bootstrap over runs (10,000 resamples, fixed seed), and its
             p-value is the exact-table / t-approximation one scipy reports.

Reads the logs through common.runs() and results/real_validation.json.
Writes results/equal_budget.json.
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

import common
from statsutil import perm_p, hedges_g

CONTRASTS = [('schedule (restart $-$ one long run)', ['frozen'], ['baseline']),
             ('operator loop (LLM $-$ frozen)', common.LLMS, ['frozen']),
             ('LLM arms $-$ baseline', common.LLMS, ['baseline'])]
N_BOOT = 10000


def _contrast(best, left, right):
    a = [x for m in left for x in best[m]]
    b = [x for m in right for x in best[m]]
    p, _ = perm_p(a, b)
    return dict(n_left=len(a), n_right=len(b), delta=float(np.mean(a) - np.mean(b)),
                g=float(hedges_g(a, b)), p=float(p))


def main():
    rv = json.load(open(os.path.join(common.RES, 'real_validation.json')))
    out = {}
    for ds in common.DATASETS:
        runs = {m: common.runs(m, ds) for m in common.METHODS}
        n_unique = {m: [int(len(r['acc3'])) for r in runs[m]] for m in runs}
        n_common = min(min(v) for v in n_unique.values())

        def best(n):
            return {m: [float(np.nanmax(r['acc3'][:n])) for r in runs[m]] for m in runs}

        full, cut = best(None), best(n_common)
        o = dict(n_common=n_common, n_unique=n_unique,
                 n_unique_mean={m: float(np.mean(v)) for m, v in n_unique.items()},
                 n_unique_restart_mean=float(np.mean(
                     [x for m in common.RESTART for x in n_unique[m]])),
                 contrasts={lab: dict(full=_contrast(full, le, ri),
                                      equal_budget=_contrast(cut, le, ri))
                            for lab, le, ri in CONTRASTS})

        meas = np.array([r['measured_best'] for r in rv[ds]['runs']])
        pred = np.array([r['predicted_best'] for r in rv[ds]['runs']])
        rng = np.random.default_rng(0)
        boot = []
        while len(boot) < N_BOOT:
            i = rng.integers(0, len(meas), len(meas))
            if len(set(meas[i].tolist())) > 1 and len(set(pred[i].tolist())) > 1:
                boot.append(spearmanr(pred[i], meas[i]).statistic)
        sp = spearmanr(pred, meas)
        o['agreement'] = dict(n_runs=int(len(meas)), rho=float(sp.statistic),
                              p=float(sp.pvalue), ci_lo=float(np.percentile(boot, 2.5)),
                              ci_hi=float(np.percentile(boot, 97.5)), n_boot=N_BOOT)
        out[ds] = o

    json.dump(out, open(os.path.join(common.RES, 'equal_budget.json'), 'w'), indent=1)
    for ds, o in out.items():
        print(f"\n== {ds}: every run cut at N = {o['n_common']} unique architectures")
        print('   unique architectures per run, mean by arm: '
              + ', '.join(f"{m} {v:.0f}" for m, v in o['n_unique_mean'].items()))
        for lab, c in o['contrasts'].items():
            f, e = c['full'], c['equal_budget']
            print(f"   {lab:36s} as run {f['delta']:+6.2f} (g={f['g']:+.2f}, p={f['p']:.3f})"
                  f"   equal budget {e['delta']:+6.2f} (g={e['g']:+.2f}, p={e['p']:.3f})")
        a = o['agreement']
        print(f"   outcome agreement rho = {a['rho']:.3f} over {a['n_runs']} runs, "
              f"95% CI [{a['ci_lo']:.2f}, {a['ci_hi']:.2f}], p = {a['p']:.4f}")


if __name__ == '__main__':
    main()
