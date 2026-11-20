"""External validation: does the benchmark reproduce the outcomes of the real runs?

Every difference this paper reports is read off a fitted landscape, so the question a reader
is entitled to ask first is whether that landscape reproduces anything that was actually
measured. The released record answers it without a single new GPU hour, because it contains
34 completed searches whose outcomes the benchmark never saw in the form it is asked about
here.

The test is leave-one-run-out at the level of the *run*, not of the architecture. For each of
the 34 runs the surrogate is refitted on the other 33, and is then asked for the outcome of
the held-out run -- the best score among the architectures that run actually visited. That
quantity is the same endpoint the real record reports, so the two can be compared directly:

  run level   Spearman and Kendall agreement between the predicted and the measured outcome
              over the 34 held-out runs, and the mean absolute error on the accuracy scale;
  arm level   the two contrasts the original study drew, the schedule (baseline against the
              frozen restart control) and the operator loop (frozen against the three LLM
              arms pooled), computed on the measured outcomes and on the held-out predicted
              outcomes, each with Hedges' g and an exact permutation test, and compared for
              sign and size.

This is the strongest statement the record can support without new runs: it does not confirm
any effect the benchmark predicts for a configuration nobody ran, which is what the
pre-registered confirmation of the main paper's Section IX is for. It does establish that the
surrogate's ordering of real outcomes is not an artefact of the architectures it was fitted on.

Writes results/real_validation.json, tables/tab_realval.tex.
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr, kendalltau

import common
import surrogate as sg
from statsutil import perm_p, hedges_g

ARMS = ['baseline', 'frozen', 'qwen', 'deepseek', 'mistral']
#: the two contrasts the original study drew, as (label, arms on the left, arms on the right)
CONTRASTS = [('schedule (restart $-$ one long run)', ['frozen'], ['baseline']),
             ('operator loop (LLM $-$ frozen)', common.LLMS, ['frozen'])]


def _per_run(ds):
    """Held-out predicted and measured outcome for every run of one dataset."""
    rows = [r for r in common.gene_table()
            if r['dataset'] == ds and r['acc3'] is not None]
    P = np.array([sg.codes_of(r) for r in rows])
    y = np.array([r['acc3'] for r in rows], float)
    run = np.array([r['run'] for r in rows])
    arm = np.array([r['run'].split('/')[0] for r in rows])
    gf = sg.GramFitter(P, y)
    out = []
    for rid in sorted(set(run.tolist())):
        te = np.flatnonzero(run == rid)
        keep = np.ones(len(y), bool)
        keep[te] = False
        pred = gf.fit_mask(keep)(te)
        out.append(dict(run=rid, arm=str(arm[te][0]), n=int(len(te)),
                        measured_best=float(y[te].max()),
                        predicted_best=float(pred.max()),
                        measured_mean=float(y[te].mean()),
                        predicted_mean=float(pred.mean()),
                        rho_within=float(spearmanr(pred, y[te]).statistic)))
    return out


def _contrast(runs, left, right, key):
    a = [r[key] for r in runs if r['arm'] in left]
    b = [r[key] for r in runs if r['arm'] in right]
    if len(a) < 2 or len(b) < 2:
        return None
    p, _ = perm_p(a, b)
    return dict(n_left=len(a), n_right=len(b), mean_left=float(np.mean(a)),
                mean_right=float(np.mean(b)), delta=float(np.mean(a) - np.mean(b)),
                g=float(hedges_g(a, b)), p=float(p))


def main():
    out = {}
    for ds in common.DATASETS:
        runs = _per_run(ds)
        m = np.array([r['measured_best'] for r in runs])
        p = np.array([r['predicted_best'] for r in runs])
        o = dict(n_runs=len(runs), runs=runs,
                 rho_runs=float(spearmanr(p, m).statistic),
                 tau_runs=float(kendalltau(p, m).statistic),
                 mae_runs=float(np.abs(p - m).mean()),
                 rho_within_mean=float(np.mean([r['rho_within'] for r in runs])),
                 contrasts={})
        for lab, left, right in CONTRASTS:
            cm = _contrast(runs, left, right, 'measured_best')
            cp = _contrast(runs, left, right, 'predicted_best')
            if cm is None or cp is None:
                continue
            o['contrasts'][lab] = dict(
                measured=cm, predicted=cp,
                sign_agrees=bool(np.sign(cm['delta']) == np.sign(cp['delta'])),
                size_ratio=float(cp['delta'] / cm['delta']) if cm['delta'] else None)
        out[ds] = o
    json.dump(out, open(os.path.join(common.RES, 'real_validation.json'), 'w'), indent=1)
    _tex(out)

    for ds, o in out.items():
        print(f"\n== {ds}: {o['n_runs']} held-out runs")
        print(f"   outcome agreement  rho={o['rho_runs']:.3f}  tau={o['tau_runs']:.3f}  "
              f"MAE={o['mae_runs']:.2f} pts   within-run rho {o['rho_within_mean']:.3f}")
        for lab, c in o['contrasts'].items():
            print(f"   {lab:38s} measured {c['measured']['delta']:+6.2f} "
                  f"(g={c['measured']['g']:+.2f}, p={c['measured']['p']:.3f})   "
                  f"held-out surrogate {c['predicted']['delta']:+6.2f} "
                  f"(g={c['predicted']['g']:+.2f}, p={c['predicted']['p']:.3f})   "
                  f"{'sign agrees' if c['sign_agrees'] else 'SIGN DISAGREES'}")


def _tex(out):
    A, B = out['cifar10'], out['cifar100']
    L = [r'\begin{tabular}{l rr rr}', r'\toprule',
         r'& \multicolumn{2}{c}{CIFAR-10} & \multicolumn{2}{c}{CIFAR-100} \\',
         r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
         r'& \textbf{measured} & \textbf{held out} & \textbf{measured} & \textbf{held out} \\',
         r'\midrule',
         r'\multicolumn{5}{l}{\textit{Outcome of each held-out run (Spearman over runs,'
         r' MAE in points)}} \\',
         rf"Agreement $\rho$ & \multicolumn{{2}}{{c}}{{{A['rho_runs']:.3f}}} "
         rf"& \multicolumn{{2}}{{c}}{{{B['rho_runs']:.3f}}} \\",
         rf"Mean absolute error & \multicolumn{{2}}{{c}}{{{A['mae_runs']:.2f}}} "
         rf"& \multicolumn{{2}}{{c}}{{{B['mae_runs']:.2f}}} \\",
         r'\midrule',
         r'\multicolumn{5}{l}{\textit{The two contrasts the record supports (points, '
         "Hedges' $g$ in brackets)}} " r'\\']
    for lab, _, _ in CONTRASTS:
        cells = []
        for o in (A, B):
            c = o['contrasts'].get(lab)
            if c is None:
                cells += ['--', '--']
            else:
                cells += [rf"${c['measured']['delta']:+.2f}$ [{c['measured']['g']:+.2f}]",
                          rf"${c['predicted']['delta']:+.2f}$ [{c['predicted']['g']:+.2f}]"]
        L.append(lab + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(common.HERE, '..', 'tables', 'tab_realval.tex'),
         'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
