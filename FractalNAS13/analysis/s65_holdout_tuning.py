"""Tuning on one landscape, scoring on another: how much of a priced gain is self-confirmation?

The benchmark is both the object tuned in the settings sweep and the signal that scores the
tuning, so a gain read off it is exposed to selection at the level of the configuration. This
script measures that exposure instead of conceding it, and it needs no GPU hour: the record
holds 34 runs, and a landscape fitted on half of them is a different landscape from one fitted
on the other half.

Three transfer tests, all on material the study already owns:

  split-half   the runs of a dataset are partitioned into two halves, stratified by arm, and a
               landscape is fitted on each. Every setting family is swept on half A, the value
               it recommends is carried to half B, and the gain of that value against the one
               the pipeline fixed is re-measured there. The in-sample gain is what a paper that
               tuned and scored on the same landscape would report; the held-out gain is what
               the recommendation is worth on a landscape that had no part in choosing it, and
               the difference between them is the optimism.

  cross-arm    a landscape fitted only on the baseline-GA runs recommends; a landscape fitted
               only on the restart runs scores, and the reverse. The two arms explored the
               space under different schedules, so this is the least favourable split the
               record supports.

  cross-dataset  the recommendation from CIFAR-10 is scored on CIFAR-100 and the reverse.

What transfers is the recommended value and the sign and rough size of its gain; the absolute
score cannot, since two landscapes fitted on different architectures are different functions.

Writes results/holdout_tuning.json, tables/tab_holdout.tex.
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

import common
import fractalbench as fb
import surrogate as sg

R = 500
N_SPLITS = 3
FAMILIES = [('tournament size $k$', 'k', 3, [2, 3, 5, 7, 10]),
            ('mutation rate $p_m$', 'pm', 0.08, [0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4]),
            ('population size', 'pop', 20, [10, 20, 40, 52]),
            ('elite count', 'elite', 5, [0, 1, 2, 4, 5, 8])]
#: an equal budget of 520 evaluated individuals, as everywhere else
POPGENS = {10: 52, 20: 26, 40: 13, 52: 10}


class SubsetBench(fb.Bench):
    """The benchmark's own model, fitted on the architectures of a subset of the runs.

    Both the ridge and the tabular table are restricted to that subset, so an architecture a
    held-out run visited is predicted by this landscape and never served from the record.
    """

    def __init__(self, ds, runs, alpha=10.0):
        rows = [r for r in common.gene_table()
                if r['dataset'] == ds and r['acc3'] is not None and r['run'] in runs]
        self.ds, self.rows, self.runs = ds, rows, sorted(runs)
        P = np.array([sg.codes_of(r) for r in rows])
        y = np.array([r['acc3'] for r in rows], float)
        self.model = sg.PairwiseRidge(alpha).fit(P, y)
        tab = {}
        for p, v in zip(P, y):
            tab.setdefault(tuple(int(x) for x in p), []).append(float(v))
        self.table = {k: float(np.mean(v)) for k, v in tab.items()}
        self.y_sd, self.y_mean = float(y.std()), float(y.mean())
        self.y_lo, self.y_hi = float(y.min()), float(y.max())
        self.n_fit = len(rows)


def _rows(ds):
    return [r for r in common.gene_table()
            if r['dataset'] == ds and r['acc3'] is not None]


def _run_index(ds):
    """run id -> arm, for stratified splitting."""
    idx = {}
    for r in _rows(ds):
        idx.setdefault(r['run'], r['arm'])
    return idx


def _fidelity(fit_bench, rows):
    """Spearman of a landscape against architectures no part of it was fitted on."""
    P = np.array([sg.codes_of(r) for r in rows])
    y = np.array([r['acc3'] for r in rows], float)
    return float(spearmanr(fit_bench.model.predict(P), y).statistic)


def _sweep(bench, R=R):
    """Every setting family swept on one landscape. Returns value -> score and the argmax."""
    out = {}
    for lab, key, dflt, values in FAMILIES:
        cells = {}
        for v in values:
            kw = dict(n_seg=1, gens=26)
            if key == 'pop':
                kw.update(pop=v, gens=POPGENS[v])
            else:
                kw.update(fb.setting_kw(key, v))
            cells[str(v)] = float(fb.run_ga(bench, R=R, seed=1, **kw)['best'].mean())
        best = max(values, key=lambda v: cells[str(v)])
        out[lab] = dict(cells=cells, pipeline=str(dflt), best=str(best),
                        gain=cells[str(best)] - cells[str(dflt)])
    return out


def _carry(rec_sweep, score_sweep):
    """The value one landscape recommends, re-scored on another landscape's sweep."""
    out = {}
    for lab, key, dflt, values in FAMILIES:
        rec = rec_sweep[lab]['best']
        sc = score_sweep[lab]['cells']
        out[lab] = dict(recommended=rec,
                        in_sample_gain=rec_sweep[lab]['gain'],
                        heldout_gain=sc[rec] - sc[str(dflt)],
                        heldout_best=score_sweep[lab]['best'],
                        same_value=rec == score_sweep[lab]['best'],
                        heldout_best_gain=score_sweep[lab]['gain'])
    return out


def _pairs(ds, splits):
    """Every (recommending landscape, scoring landscape) pair this record supports."""
    idx = _run_index(ds)
    runs = sorted(idx)
    out = []
    rng = np.random.default_rng(20260101)
    for s in range(splits):
        a, b = [], []
        for arm in sorted(set(idx.values())):
            g = [r for r in runs if idx[r] == arm]
            g = [g[i] for i in rng.permutation(len(g))]
            a += g[::2]
            b += g[1::2]
        out.append((f'split-half {s + 1}', set(a), set(b)))
    base = {r for r in runs if idx[r] == 'baseline'}
    rest = set(runs) - base
    if base and rest:
        out.append(('baseline runs vs restart runs', base, rest))
    return out


def main():
    out = {}
    full = {}
    for ds in common.DATASETS:
        rows = _rows(ds)
        by_run = {}
        for r in rows:
            by_run.setdefault(r['run'], []).append(r)
        o = {'n_runs': len(by_run), 'n_arch': len(rows), 'transfers': []}
        for name, A, B in _pairs(ds, N_SPLITS):
            bA, bB = SubsetBench(ds, A), SubsetBench(ds, B)
            rowsA = [r for r in rows if r['run'] in A]
            rowsB = [r for r in rows if r['run'] in B]
            sA, sB = _sweep(bA), _sweep(bB)
            rec = dict(name=name,
                       n_fit=[bA.n_fit, bB.n_fit],
                       rho_across=[_fidelity(bA, rowsB), _fidelity(bB, rowsA)],
                       a_to_b=_carry(sA, sB), b_to_a=_carry(sB, sA))
            o['transfers'].append(rec)
            print(f"  {ds} {name}: fitted on {bA.n_fit}/{bB.n_fit} architectures, "
                  f"rho across {rec['rho_across'][0]:.3f}/{rec['rho_across'][1]:.3f}",
                  flush=True)
        full[ds] = _sweep(fb.Bench(ds))
        out[ds] = o

    # --- cross-dataset: what CIFAR-10 recommends, priced on CIFAR-100, and the reverse
    out['cross_dataset'] = {
        'cifar10_to_cifar100': _carry(full['cifar10'], full['cifar100']),
        'cifar100_to_cifar10': _carry(full['cifar100'], full['cifar10'])}
    out['full_sweeps'] = full

    # --- the summary the paper quotes
    summ = {}
    for ds in common.DATASETS:
        ins, hel, same, n = [], [], 0, 0
        for t in out[ds]['transfers']:
            for d in (t['a_to_b'], t['b_to_a']):
                for lab, c in d.items():
                    ins.append(c['in_sample_gain'])
                    hel.append(c['heldout_gain'])
                    same += int(c['same_value'])
                    n += 1
        summ[ds] = dict(n=n, in_sample=float(np.mean(ins)), heldout=float(np.mean(hel)),
                        optimism=float(np.mean(ins) - np.mean(hel)),
                        same_value=same, sign_kept=int(sum(h > 0 for h in hel)))
        # per family, averaged over every transfer this dataset supports
        per = {}
        for lab, _, _, _ in FAMILIES:
            i = [d[lab]['in_sample_gain'] for t in out[ds]['transfers']
                 for d in (t['a_to_b'], t['b_to_a'])]
            h = [d[lab]['heldout_gain'] for t in out[ds]['transfers']
                 for d in (t['a_to_b'], t['b_to_a'])]
            v = [d[lab]['same_value'] for t in out[ds]['transfers']
                 for d in (t['a_to_b'], t['b_to_a'])]
            per[lab] = dict(in_sample=float(np.mean(i)), heldout=float(np.mean(h)),
                            optimism=float(np.mean(i) - np.mean(h)),
                            same_value=int(sum(v)), n=len(v),
                            heldout_min=float(np.min(h)), heldout_max=float(np.max(h)))
        summ[ds]['per_family'] = per
    out['summary'] = summ

    json.dump(out, open(os.path.join(common.RES, 'holdout_tuning.json'), 'w'), indent=1)
    _tex(out)

    for ds in common.DATASETS:
        s = summ[ds]
        print(f"\n== {ds}: {s['n']} transfers of a recommended value onto a landscape that "
              f"did not choose it")
        for lab, c in s['per_family'].items():
            print(f"   {lab:22s} in-sample {c['in_sample']:+.2f}  held out {c['heldout']:+.2f}"
                  f"  (optimism {c['optimism']:+.2f}; same value {c['same_value']}/{c['n']})")
        print(f"   overall: in-sample {s['in_sample']:+.2f}, held out {s['heldout']:+.2f}, "
              f"optimism {s['optimism']:+.2f}; {s['same_value']}/{s['n']} identical "
              f"recommendations, {s['sign_kept']}/{s['n']} keep a positive gain")
    for k, d in out['cross_dataset'].items():
        print(f"\n== {k}")
        for lab, c in d.items():
            print(f"   {lab:22s} recommends {c['recommended']:>5s}  "
                  f"gain there {c['heldout_gain']:+.2f}  (same value: {c['same_value']})")


def _tex(out):
    """Main Table III (both transfer checks on one page) and the supplement's long form."""
    s = out['summary']
    fam_path = os.path.join(common.RES, 'surrogate_family.json')
    fam = json.load(open(fam_path)) if os.path.exists(fam_path) else None
    names = ['ridge', 'forest', 'boosted']

    L = [r'\begin{table}[t]', r'\centering',
         (r'\caption{The recommendation, checked against the landscape that produced it. '
          r'\textbf{Recommended value} is where each setting peaks on three landscapes '
          r'refitted to the same architectures. \textbf{in-sample} is its gain on the '
          r'landscape that chose it, \textbf{held out} its gain on a landscape fitted from '
          r'runs that did not, averaged over every split in Supplementary Table~S7; the '
          r'difference is the optimism. These two checks sweep at %s and %d replicates '
          r'rather than the $1{,}000$ of Table~\ref{tab:settings}, which is why their gains '
          r'differ from it slightly and why the ridge can peak one level away inside the '
          r'flat mutation-rate region. '
          r'Produced by \texttt{analysis/s64\_surrogate\_family.py} and '
          r'\texttt{analysis/s65\_holdout\_tuning.py}.}'
          % ('{:,}'.format(fam['replicates']).replace(',', '{,}')
             if fam and 'replicates' in fam else '200', R)),
         r'\label{tab:transfer}', r'\footnotesize',
         r'\resizebox{\columnwidth}{!}{%',
         r'\begin{tabular}{l c ccc rr}', r'\toprule',
         r'& & \multicolumn{3}{c}{\textbf{Recommended value}} '
         r'& \multicolumn{2}{c}{\textbf{Gain (points)}} \\',
         r'\cmidrule(lr){3-5}\cmidrule(lr){6-7}',
         r'\textbf{Setting} & \textbf{Pipeline} & ridge & forest & boosted '
         r'& in-sample & held out \\']
    for ds, nice in (('cifar10', 'CIFAR-10'), ('cifar100', 'CIFAR-100')):
        L += [r'\midrule', r'\multicolumn{7}{l}{\textit{%s}} \\' % nice]
        for lab, _, dflt, _ in FAMILIES:
            if fam is not None:
                rec = [fam[ds]['landscapes'][n]['sweep'][lab]['best'] for n in names]
            else:
                rec = ['--'] * 3
            c = s[ds]['per_family'][lab]
            L.append(f"{lab} & {dflt} & " + ' & '.join(rec) +
                     f" & ${c['in_sample']:+.2f}$ & ${c['heldout']:+.2f}$ \\\\")
        o = s[ds]
        L.append(r'\textit{Optimism, and identical recommendations} & & '
                 r'\multicolumn{3}{c}{%d/%d} & \multicolumn{2}{c}{$%+.2f$} \\'
                 % (o['same_value'], o['n'], o['optimism']))
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    open(os.path.join(common.HERE, '..', 'tables', 'tab_transfer.tex'),
         'w').write('\n'.join(L) + '\n')

    # the supplement's long form: every transfer, both directions
    L = [r'\begin{tabular}{l l c rr c}', r'\toprule',
         r'\textbf{Dataset} & \textbf{Fit / scored on} & \textbf{Setting} & '
         r'\textbf{in-sample} & \textbf{held out} & \textbf{same value} \\', r'\midrule']
    for ds, nice in (('cifar10', 'CIFAR-10'), ('cifar100', 'CIFAR-100')):
        for t in out[ds]['transfers']:
            for d, arrow in ((t['a_to_b'], r'$A\!\to\!B$'), (t['b_to_a'], r'$B\!\to\!A$')):
                for i, (lab, c) in enumerate(d.items()):
                    first = f"{nice} & {t['name']} {arrow}" if i == 0 else ' & '
                    L.append(f"{first} & {lab} & ${c['in_sample_gain']:+.2f}$ & "
                             f"${c['heldout_gain']:+.2f}$ & "
                             f"{'yes' if c['same_value'] else 'no'} \\\\")
            L.append(r'\addlinespace')
    for k, nice in (('cifar10_to_cifar100', 'CIFAR-10 $\\to$ CIFAR-100'),
                    ('cifar100_to_cifar10', 'CIFAR-100 $\\to$ CIFAR-10')):
        d = out['cross_dataset'][k]
        for i, (lab, c) in enumerate(d.items()):
            first = f"cross & {nice}" if i == 0 else ' & '
            L.append(f"{first} & {lab} & ${c['in_sample_gain']:+.2f}$ & "
                     f"${c['heldout_gain']:+.2f}$ & "
                     f"{'yes' if c['same_value'] else 'no'} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(common.HERE, '..', 'tables', 'tab_holdout_sm.tex'),
         'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
