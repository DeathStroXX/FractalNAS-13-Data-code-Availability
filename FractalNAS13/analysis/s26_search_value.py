"""What is the search worth, once the free surrogate exists?

If a zero-cost surrogate ranks at rho ~ 0.95, the obvious question is whether one should
search at all, or simply screen a large random sample for free and train the best of it.
This script answers it at an equal number of *measured* architectures, on the identical
landscape, against three kinds of alternative:

  screening       the free surrogate used as a filter and nothing else, at four sample sizes;
  the study's GA  at the settings it fixed by convention and at the settings the benchmark
                  recommends;
  current NAS practice   regularized evolution, best-improvement local search, and a
                  predictor-guided search in the style of BANANAS (policies.py).

Every policy is charged the same 520 measured architectures, and every per-replicate outcome
is kept, so that s62_significance.py can test the differences rather than assert them.

The main paper's table reports every policy as a *difference* against the GA as run, because
a benchmark score is an estimate on the accuracy scale and not an accuracy: outside the
sampled region the landscape extrapolates above anything the record measured, so an absolute
level printed next to a dataset name invites exactly the reading the protocol forbids. The
absolute levels, with the record's measured maximum printed beside them, go to the supplement.

Writes results/search_value.json, tables/tab_search_value.tex (differences, main paper) and
tables/tab_search_value_abs.tex (absolute levels, supplement).

    python3 s26_search_value.py              # run the policies and write both tables
    python3 s26_search_value.py --tex-only   # rewrite both tables from the stored results
"""
import json
import os
import sys

import numpy as np

import common
import fractalbench as fb
import policies

R = 400                 # replicates of a search policy
R_SCREEN = 50           # replicates of a screening policy
R_SCREEN_BIG = 30       # ... of the half-million-candidate screen, which costs seconds each
R_PRED = 60             # ... of the predictor-guided search, which refits inside the loop
BUDGET = 520
SCREENS = [1040, 5200, 52000, 520000]


def _summary(v):
    v = np.asarray(v, float)
    return dict(best=float(v.mean()), sd=float(v.std(ddof=1)),
                se=float(v.std(ddof=1) / np.sqrt(len(v))), reps=[float(x) for x in v])


def main():
    out = {}
    for ds in common.DATASETS:
        bench = fb.Bench(ds)
        # the best 3-epoch accuracy anywhere in the record the landscape is fitted from; the
        # supplement prints it beside the absolute levels so that no level can be read as an
        # accuracy the record supports
        e = {'n_measured': BUDGET, 'measured_max': bench.y_hi, 'measured_min': bench.y_lo}

        rng = np.random.default_rng(11)
        e['random'] = _summary([policies.random_measure(bench, rng, BUDGET)[0]
                                for _ in range(R_SCREEN)])
        e['screen'] = {}
        for ns in SCREENS:
            n_rep = R_SCREEN if ns <= 52000 else R_SCREEN_BIG
            e['screen'][ns] = _summary([policies.screen_then_measure(bench, rng, ns, BUDGET)[0]
                                        for _ in range(n_rep)])
            print(f'  {ds} screen {ns:7d} -> {e["screen"][ns]["best"]:.2f}', flush=True)

        # the GA as the pipeline ran it: k=3, 5 elites, effective per-gene mutation rate 0.08
        # (0.2 x the handcrafted operator's inner 0.4); and the same GA with the two settings
        # the sweeps of s19_levers.py price -- tournament size and mutation rate -- at their best
        e['ga_asrun'] = _summary(policies.run(bench, R=R, pop=20, gens=26, k=3, pm=0.08,
                                              elite=5, seed=1)['best'])
        st = json.load(open(os.path.join(common.RES, 'levers.json')))[ds]['settings']
        k_best = max((2, 3, 5, 7, 10), key=lambda v: st[f'k={v}']['best'])
        pm_best = max((0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4), key=lambda v: st[f'pm={v}']['best'])
        e['tuned_settings'] = dict(k=k_best, pm=pm_best, elite=5)
        e['ga_tuned'] = _summary(policies.run(bench, R=R, pop=20, gens=26, k=k_best,
                                              pm=pm_best, elite=5, seed=1)['best'])
        e['regularized_evolution'] = _summary(
            policies.regularized_evolution(bench, R=R, n_measured=BUDGET, seed=1)['best'])
        e['regularized_evolution_s3'] = _summary(
            policies.regularized_evolution(bench, R=R, n_measured=BUDGET, sample=3,
                                           seed=1)['best'])
        e['local_search'] = _summary(
            policies.local_search(bench, R=R, n_measured=BUDGET, seed=1)['best'])
        e['predictor_guided'] = _summary(
            policies.predictor_guided(bench, R=R_PRED, n_measured=BUDGET, seed=1)['best'])

        best_screen = max(SCREENS, key=lambda n: e['screen'][n]['best'])
        e['best_screen_n'] = best_screen
        e['ga_over_best_screen'] = e['ga_tuned']['best'] - e['screen'][best_screen]['best']
        e['ga_asrun_over_best_screen'] = (e['ga_asrun']['best']
                                          - e['screen'][best_screen]['best'])
        e['regevo_over_ga_tuned'] = (e['regularized_evolution']['best']
                                     - e['ga_tuned']['best'])
        e['regevo_pressure'] = (e['regularized_evolution']['best']
                                - e['regularized_evolution_s3']['best'])
        out[ds] = e
        print(f'  {ds} done', flush=True)

    json.dump(out, open(f'{common.RES}/search_value.json', 'w'), indent=1)
    _tex(out)

    for ds, e in out.items():
        print(f'\n== {ds} ({BUDGET} measured architectures) ==')
        print(f'   random                      {e["random"]["best"]:6.2f}')
        for ns in SCREENS:
            print(f'   screen {ns:7d} free        {e["screen"][ns]["best"]:6.2f}')
        for k, lab in [('ga_asrun', 'GA as run'), ('ga_tuned', 'GA tuned'),
                       ('predictor_guided', 'predictor-guided'),
                       ('local_search', 'local search'),
                       ('regularized_evolution_s3', 'reg. evolution (S=3)'),
                       ('regularized_evolution', 'reg. evolution (S=10)')]:
            print(f'   {lab:27s} {e[k]["best"]:6.2f}')
        print(f'   tuned GA over best screening  {e["ga_over_best_screen"]:+.2f}')
        print(f'   reg. evolution over tuned GA  {e["regevo_over_ga_tuned"]:+.2f} '
              f'(its own selection pressure is worth {e["regevo_pressure"]:+.2f})')


def _screen(e, n):
    """One screening level, whether `e` came from this run or from the stored JSON, whose
    object keys are strings."""
    d = e['screen']
    return (d[n] if n in d else d[str(n)])['best']


ROWS = [('Random sample, no surrogate', lambda e: e['random']['best'])]
for _ns in SCREENS:
    ROWS.append((rf'Screen {_ns:,} free, measure the best 520'.replace(',', '{,}'),
                 (lambda n: (lambda e: _screen(e, n)))(_ns)))
ROWS += [(r'\midrule', None),
         ('GA tuned (best $k$: 5 and 7)', lambda e: e['ga_tuned']['best']),
         (r'\midrule', None),
         ('Predictor-guided search', lambda e: e['predictor_guided']['best']),
         ('Local search', lambda e: e['local_search']['best']),
         ('Regularized evolution ($S{=}3$)', lambda e: e['regularized_evolution_s3']['best']),
         ('Regularized evolution ($S{=}10$, default)',
          lambda e: e['regularized_evolution']['best'])]

#: the reference every row of the main table is a difference against
REF = ('GA as run ($k{=}3$, $p_m{=}0.08$, 5 elites)', lambda e: e['ga_asrun']['best'])


def _tex_delta(out):
    """The main paper's table: every policy as a difference against the GA as run."""
    a, b = out['cifar10'], out['cifar100']
    L = [r'\begin{table}[t]', r'\centering',
         r'\caption{What the search is worth once the free surrogate exists. Every policy',
         r'trains the same number of architectures, 520; only the way they are chosen differs.',
         r'Entries are \emph{differences} in benchmark score against the GA at the settings the',
         r'original study fixed, measured on the identical landscape from the same seeds. A',
         r'benchmark score is an estimate on the accuracy scale and not an accuracy: the best',
         rf'3-epoch accuracy anywhere in the record is {a["measured_max"]:.2f} on CIFAR-10 and',
         rf'{b["measured_max"]:.2f} on CIFAR-100, and outside the sampled region the landscape',
         r'extrapolates above it, so the absolute levels are reported only in Supplementary',
         r'Table~S3. Screening a large free sample beats the GA as run; only the \emph{tuned}',
         r'GA beats screening. Regularized evolution, whose defaults already sit at the',
         r'selection pressure and per-gene mutation rate Table~\ref{tab:settings} recommends,',
         r"beats both; at the study's own selection pressure ($S{=}3$) it does not.",
         r'Differences are tested in Table~\ref{tab:tests}.}',
         r'\label{tab:searchvalue}', r'\footnotesize',
         r'\resizebox{\columnwidth}{!}{%', r'\begin{tabular}{lrr}', r'\toprule',
         r'& \multicolumn{2}{c}{$\Delta$ against the GA as run (pts)} \\',
         r'\cmidrule(lr){2-3}',
         r'Policy (520 measured architectures) & CIFAR-10 & CIFAR-100 \\', r'\midrule',
         rf'{REF[0]} \emph{{(reference)}} & $0.00$ & $0.00$ \\', r'\midrule']
    for name, fn in ROWS:
        if fn is None:
            L.append(name)
            continue
        L.append(rf'{name} & ${fn(a) - REF[1](a):+.2f}$ & ${fn(b) - REF[1](b):+.2f}$ \\')
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    open(f'{common.HERE}/../tables/tab_search_value.tex', 'w').write('\n'.join(L) + '\n')


def _tex_absolute(out):
    """The supplement's table: the absolute levels, with the record's maximum beside them."""
    a, b = out['cifar10'], out['cifar100']
    L = [r'%% Generated by analysis/s26_search_value.py -- do not edit.', r'\begin{table}[t]',
         r'\centering',
         r'\caption{The absolute benchmark levels behind main Table~IV, which reports the same',
         r'policies as differences. Every policy trains 520 architectures. A level here is the',
         r"landscape's estimate on the accuracy scale and not an accuracy: the last row is the",
         r'best 3-epoch accuracy anywhere in the record the landscape is fitted from, and a',
         r'policy that drives the search into the region no run entered is scored by',
         r'extrapolation above it. The last row is the one place this archive sets a measured',
         r'accuracy beside a benchmark level, and it is there to mark that ceiling rather than',
         r'to compare the two; the three rules of Section~\ref{sec:sm_surrogate} apply to',
         r'every number above it.}',
         r'\label{tab:sm_searchvalue}', r'\footnotesize',
         r'\resizebox{\columnwidth}{!}{%', r'\begin{tabular}{lrr}', r'\toprule',
         r'& \multicolumn{2}{c}{Benchmark score (not an accuracy)} \\',
         r'\cmidrule(lr){2-3}',
         r'Policy (520 measured architectures) & CIFAR-10 & CIFAR-100 \\', r'\midrule']
    rows = ROWS[:1] + ROWS[1:5] + [(r'\midrule', None), REF] + ROWS[5:]
    for name, fn in rows:
        if fn is None:
            L.append(name)
            continue
        L.append(rf'{name} & {fn(a):.2f} & {fn(b):.2f} \\')
    L += [r'\midrule',
          rf'\textit{{Best 3-epoch accuracy measured in the record}} & '
          rf'\textit{{{a["measured_max"]:.2f}}} & \textit{{{b["measured_max"]:.2f}}} \\',
          r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    open(f'{common.HERE}/../tables/tab_search_value_abs.tex', 'w').write('\n'.join(L) + '\n')


def _tex(out):
    _tex_delta(out)
    _tex_absolute(out)


if __name__ == '__main__':
    if '--tex-only' in sys.argv:
        _tex(json.load(open(f'{common.RES}/search_value.json')))
        print('rewrote tables/tab_search_value.tex and tables/tab_search_value_abs.tex')
    else:
        main()
