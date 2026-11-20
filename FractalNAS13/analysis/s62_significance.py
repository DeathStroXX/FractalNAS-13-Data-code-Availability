"""Paired non-parametric tests for every difference the paper claims.

The journal asks for a test of significance before one configuration is said to outperform
another, and for a correction when several such claims are made from the same experiment.
This script supplies both, reading the per-replicate outcomes the three result scripts keep
in `analysis/results/` rather than recomputing anything:

  design        every configuration of a family is run from the same seed, so replicate r of
                one configuration and replicate r of another start from the identical
                population. The replicate index is therefore a block, and the family of
                configurations is compared with a Friedman test across those blocks, followed
                by per-pair Wilcoxon signed-rank tests. Two policies that share no randomness
                -- a GA against a screening policy -- are not blocked, and are compared with
                the unpaired Mann-Whitney U instead, which is stated per row.
  correction    Holm-Bonferroni over a declared family of comparisons, one family per table,
                so that the corrected p-values are the ones the paper quotes.
  effect size   the rank-biserial correlation for a Wilcoxon or Mann-Whitney comparison, and
                Hedges' g on the same pair, because a p-value computed over a thousand
                benchmark replicates is small for any difference that is not exactly zero and
                says nothing on its own about whether the difference matters.

Reads results/search_value.json, results/levers.json, results/operator_bench.json.
Writes results/significance.json and tables/tab_tests.tex.

    python3 analysis/s62_significance.py
"""
import json
import os

import numpy as np
from scipy import stats

import common
from statsutil import hedges_g, holm

ALPHA = 0.05


def _load(name):
    return json.load(open(os.path.join(common.RES, name + '.json')))


def wilcoxon(a, b):
    """Paired signed-rank test, with the rank-biserial correlation as its effect size."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    nz = d[d != 0]
    if len(nz) == 0:
        return dict(test='wilcoxon', n=len(d), stat=0.0, p=1.0, rbc=0.0,
                    g=0.0, delta=0.0)
    res = stats.wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
    r = stats.rankdata(np.abs(nz))
    rp = r[nz > 0].sum()
    rm = r[nz < 0].sum()
    return dict(test='wilcoxon', n=int(len(d)), stat=float(res.statistic),
                p=float(res.pvalue), rbc=float((rp - rm) / (rp + rm)),
                g=float(hedges_g(a, b)), delta=float(np.mean(d)))


def mannwhitney(a, b):
    """Unpaired rank-sum test, with the rank-biserial correlation as its effect size."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    res = stats.mannwhitneyu(a, b, alternative='two-sided')
    u = float(res.statistic)
    return dict(test='mann-whitney', n=int(len(a)), m=int(len(b)), stat=u,
                p=float(res.pvalue), rbc=float(2 * u / (len(a) * len(b)) - 1),
                g=float(hedges_g(a, b)), delta=float(np.mean(a) - np.mean(b)))


def strat_sign_permutation(groups, n_perm=200000, seed=0):
    """Dataset-stratified sign-flip permutation test on per-rewrite effects.

    A class holds as few as two rewrites per dataset, so the per-dataset signed-rank test
    cannot reach a corrected 0.05 however large the effect, and pooling the two datasets
    ignores that a rewrite belongs to one of them. This test keeps the stratification: the
    statistic is the unweighted mean of the per-dataset mean effects, and the null flips the
    sign of each rewrite's effect independently *within* its dataset, which is the exact null
    of "this class of rewrite is as likely to help as to hurt, on each dataset separately".
    Enumerated exactly when the total number of rewrites allows it, and sampled otherwise.
    """
    groups = [np.asarray(g, float) for g in groups if len(g)]
    if not groups:
        return dict(test='stratified sign permutation', n=0, stat=0.0, p=1.0, exact=True,
                    n_perm=0)
    n = sum(len(g) for g in groups)

    def stat_of(gs):
        return float(np.mean([g.mean() for g in gs]))

    obs = stat_of(groups)
    exact = n <= 20
    if exact:
        count, total = 0, 0
        for mask in range(1 << n):
            i, flipped = 0, []
            for g in groups:
                sgn = np.array([1.0 if not (mask >> (i + j)) & 1 else -1.0
                                for j in range(len(g))])
                flipped.append(g * sgn)
                i += len(g)
            total += 1
            if abs(stat_of(flipped)) >= abs(obs) - 1e-12:
                count += 1
        p = count / total
    else:
        rng = np.random.default_rng(seed)
        count = 0
        for _ in range(n_perm):
            flipped = [g * rng.choice((-1.0, 1.0), len(g)) for g in groups]
            if abs(stat_of(flipped)) >= abs(obs) - 1e-12:
                count += 1
        p = (count + 1) / (n_perm + 1)
        total = n_perm
    return dict(test='stratified sign permutation', n=int(n), stat=obs, p=float(p),
                exact=bool(exact), n_perm=int(total))


def bootstrap_ci(d, level=0.95, n_boot=20000, seed=0):
    """Percentile bootstrap interval for the mean effect of a class, over its rewrites."""
    d = np.asarray(d, float)
    if len(d) < 2:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    m = rng.choice(d, (n_boot, len(d)), replace=True).mean(1)
    lo, hi = np.quantile(m, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def friedman(blocks, labels):
    """Friedman test over the configurations of one family, blocked by replicate index."""
    arrs = [np.asarray(b, float) for b in blocks]
    n = min(len(a) for a in arrs)
    arrs = [a[:n] for a in arrs]
    res = stats.friedmanchisquare(*arrs)
    ranks = stats.rankdata(-np.column_stack(arrs), axis=1).mean(0)
    k = len(arrs)
    w = float(res.statistic / (n * (k - 1))) if n and k > 1 else 0.0
    return dict(test='friedman', k=k, n=n, stat=float(res.statistic),
                p=float(res.pvalue), kendall_w=w,
                mean_rank=dict(zip(labels, [float(x) for x in ranks])))


# ---------------------------------------------------------------- the three families
def family_search_value(sv):
    """Table III: the seven policies, and the comparisons the text draws from them."""
    out = {}
    for ds in common.DATASETS:
        e = sv[ds]
        best_screen = str(e['best_screen_n'])
        pol = {'GA as run': e['ga_asrun']['reps'],
               'GA tuned': e['ga_tuned']['reps'],
               'Predictor-guided': e['predictor_guided']['reps'],
               'Local search': e['local_search']['reps'],
               'Reg. evolution $S{=}3$': e['regularized_evolution_s3']['reps'],
               'Reg. evolution $S{=}10$': e['regularized_evolution']['reps']}
        # the search policies share a seed stream and are therefore blocked; the predictor-
        # guided search is run at fewer replicates, so the block count is its replicate count
        o = {'friedman': friedman(list(pol.values()), list(pol))}
        tests = {
            'tuned GA $-$ best screening':
                mannwhitney(e['ga_tuned']['reps'], e['screen'][best_screen]['reps']),
            'as-run GA $-$ best screening':
                mannwhitney(e['ga_asrun']['reps'], e['screen'][best_screen]['reps']),
            'tuned GA $-$ as-run GA':
                wilcoxon(e['ga_tuned']['reps'], e['ga_asrun']['reps']),
            'reg. evolution $-$ tuned GA':
                wilcoxon(e['regularized_evolution']['reps'], e['ga_tuned']['reps']),
            'reg. evolution $S{=}10$ $-$ $S{=}3$':
                wilcoxon(e['regularized_evolution']['reps'],
                         e['regularized_evolution_s3']['reps']),
            'local search $-$ as-run GA':
                wilcoxon(e['local_search']['reps'], e['ga_asrun']['reps']),
        }
        corr = holm({k: v['p'] for k, v in tests.items()})
        for k, v in tests.items():
            v['p_holm'] = float(corr[k])
        o['tests'] = tests
        o['best_screen_n'] = int(best_screen)
        out[ds] = o
    return out


def family_settings(lv):
    """Table IV: each GA setting swept, against the value the pipeline fixed."""
    fams = [('tournament size $k$', 'k=3', ['k=2', 'k=3', 'k=5', 'k=7', 'k=10']),
            ('mutation rate $p_m$', 'pm=0.08',
             ['pm=0.02', 'pm=0.05', 'pm=0.08', 'pm=0.1', 'pm=0.2', 'pm=0.3', 'pm=0.4']),
            ('population size', 'pop=20', ['pop=10', 'pop=20', 'pop=40', 'pop=52']),
            ('elite count', 'elite=5', ['elite=0', 'elite=1', 'elite=2', 'elite=4', 'elite=5',
                                        'elite=8'])]
    out = {}
    for ds in common.DATASETS:
        st = lv[ds]['settings']
        o, tests = {'friedman': {}}, {}
        for lab, dflt, keys in fams:
            o['friedman'][lab] = friedman([st[k]['reps'] for k in keys], keys)
            bk = max(keys, key=lambda k: st[k]['best'])
            t = wilcoxon(st[bk]['reps'], st[dflt]['reps'])
            t['best'] = bk
            t['pipeline'] = dflt
            tests[lab] = t
        corr = holm({k: v['p'] for k, v in tests.items()})
        for k, v in tests.items():
            v['p_holm'] = float(corr[k])
        o['tests'] = tests
        out[ds] = o
    return out


def family_operators(ob):
    """Table VI: every in-space rewrite against the handcrafted operator it replaced."""
    nice = {'mutate_gene': 'Mutation', '_create_random_chromosome': 'Initialization',
            'select_competitor': 'Selection', 'combine_genes': 'Crossover'}
    out = {}
    for ds in common.DATASETS:
        e = ob[ds]
        # every rewrite against the operator it replaced; a mutation rewrite replaced the
        # pipeline's own mutate_gene, replayed through the same code path (s20_operator_bench)
        base = e['baseline_reps']
        base_mut = e.get('mutation_reference_reps', base)
        per, by_kind = {}, {}
        for r in e['rows']:
            if 'reps' not in r:
                continue
            t = wilcoxon(r['reps'], base_mut if r['kind'] == 'mutate_gene' else base)
            t['kind'] = r['kind']
            per[r['file']] = t
            by_kind.setdefault(r['kind'], []).append((r['file'], t))
        corr = holm({k: v['p'] for k, v in per.items()})
        for k, v in per.items():
            v['p_holm'] = float(corr[k])
        kind = {}
        for k, items in by_kind.items():
            d = np.array([per[f]['delta'] for f, _ in items], float)
            # one-sample signed-rank over the per-rewrite effects of that class
            if len(d) >= 2 and np.any(d != 0):
                res = stats.wilcoxon(d, alternative='two-sided')
                p = float(res.pvalue)
                stat = float(res.statistic)
            else:
                p, stat = 1.0, 0.0
            lo, hi = bootstrap_ci(d)
            kind[nice.get(k, k)] = dict(test='wilcoxon (one sample)', n=len(d), stat=stat,
                                        p=p, mean=float(d.mean()), ci_lo=lo, ci_hi=hi,
                                        n_positive=int((d > 0).sum()),
                                        n_holm_sig=int(sum(per[f]['p_holm'] < ALPHA
                                                           for f, _ in items)))
        corr_k = holm({k: v['p'] for k, v in kind.items()})
        for k, v in kind.items():
            v['p_holm'] = float(corr_k[k])
        groups = [np.array([per[f]['delta'] for f, _ in items], float)
                  for items in by_kind.values() if len(items) >= 2]
        kw = stats.kruskal(*groups) if len(groups) >= 2 else None
        out[ds] = dict(per_rewrite=per, by_kind=kind, deltas_by_kind={
                           nice.get(k, k): [float(per[f]['delta']) for f, _ in items]
                           for k, items in by_kind.items()},
                       n_rewrites=len(per),
                       n_holm_sig=int(sum(v['p_holm'] < ALPHA for v in per.values())),
                       n_holm_sig_pos=int(sum(v['p_holm'] < ALPHA and v['delta'] > 0
                                              for v in per.values())),
                       kruskal=None if kw is None else dict(
                           test='kruskal-wallis', k=len(groups), stat=float(kw.statistic),
                           p=float(kw.pvalue)))

    # Per dataset a class holds as few as 2 and at most 17 rewrites, and the smallest
    # two-sided signed-rank p attainable at n=7 is 2/2^7 = 0.0156, which Holm over four
    # classes cannot take below 0.05 however large the effect. The class-level claim is a
    # claim about both datasets, so it is tested on the rewrites of both. The primary test is
    # the stratified sign permutation above, which keeps each rewrite inside its own dataset;
    # the pooled signed-rank below is reported beside it as the less conservative companion,
    # and the two are required to agree.
    pooled = {}
    for k in set().union(*(set(out[d]['deltas_by_kind']) for d in common.DATASETS)):
        d = np.array(sum((out[ds]['deltas_by_kind'].get(k, []) for ds in common.DATASETS),
                         []), float)
        if len(d) >= 2 and np.any(d != 0):
            res = stats.wilcoxon(d, alternative='two-sided')
            p, stat = float(res.pvalue), float(res.statistic)
        else:
            p, stat = 1.0, 0.0
        n_pos = int((d > 0).sum())
        sign_p = float(stats.binomtest(n_pos, len(d), 0.5).pvalue) if len(d) else 1.0
        perm = strat_sign_permutation(
            [out[ds]['deltas_by_kind'].get(k, []) for ds in common.DATASETS])
        lo, hi = bootstrap_ci(d)
        pooled[k] = dict(test='wilcoxon (one sample, datasets pooled)', n=len(d), stat=stat,
                         p=p, mean=float(d.mean()), ci_lo=lo, ci_hi=hi, n_positive=n_pos,
                         sign_test_p=sign_p, min_attainable_p=float(2 ** (1 - len(d))),
                         perm=perm)
    corr_p = holm({k: v['p'] for k, v in pooled.items()})
    corr_perm = holm({k: v['perm']['p'] for k, v in pooled.items()})
    for k, v in pooled.items():
        v['p_holm'] = float(corr_p[k])
        v['perm_p_holm'] = float(corr_perm[k])
        v['tests_agree'] = bool((v['p_holm'] < ALPHA) == (v['perm_p_holm'] < ALPHA))
    out['pooled_by_kind'] = pooled
    return out


def main():
    sv, lv, ob = _load('search_value'), _load('levers'), _load('operator_bench')
    out = {'alpha': ALPHA,
           'search_value': family_search_value(sv),
           'settings': family_settings(lv),
           'operators': family_operators(ob)}
    json.dump(out, open(os.path.join(common.RES, 'significance.json'), 'w'), indent=1)
    _tex(out)

    for fam in ('search_value', 'settings'):
        for ds in common.DATASETS:
            print(f'\n== {fam} / {ds}')
            o = out[fam][ds]
            fr = o['friedman']
            if isinstance(fr, dict) and 'stat' in fr:
                print(f"   Friedman chi2={fr['stat']:.1f} (k={fr['k']}, n={fr['n']}), "
                      f"p={fr['p']:.3g}, W={fr['kendall_w']:.3f}")
            else:
                for lab, f in fr.items():
                    print(f"   Friedman [{lab}] chi2={f['stat']:.1f} "
                          f"(k={f['k']}, n={f['n']}), p={f['p']:.3g}, W={f['kendall_w']:.3f}")
            for lab, t in o['tests'].items():
                print(f"   {lab:36s} {t['delta']:+6.2f}  {t['test']:<13s} "
                      f"p={t['p']:.3g}  Holm p={t['p_holm']:.3g}  rbc={t['rbc']:+.2f}")
    for ds in common.DATASETS:
        o = out['operators'][ds]
        print(f'\n== operators / {ds}: {o["n_rewrites"]} rewrites, '
              f'{o["n_holm_sig"]} significant after Holm ({o["n_holm_sig_pos"]} of them positive)')
        if o['kruskal']:
            print(f"   Kruskal-Wallis across operator classes: H={o['kruskal']['stat']:.1f}, "
                  f"p={o['kruskal']['p']:.3g}")
        for k, v in o['by_kind'].items():
            print(f"   {k:16s} n={v['n']:2d} mean {v['mean']:+5.2f}  p={v['p']:.3g}  "
                  f"Holm p={v['p_holm']:.3g}  ({v['n_holm_sig']} rewrites individually "
                  f"significant)")
    print('\n== operator classes, both datasets pooled')
    for k, v in out['operators']['pooled_by_kind'].items():
        print(f"   {k:16s} n={v['n']:2d} mean {v['mean']:+5.2f}  {v['n_positive']} positive  "
              f"p={v['p']:.3g}  Holm p={v['p_holm']:.3g}  sign-test p={v['sign_test_p']:.3g}  "
              f"(smallest attainable p at this n: {v['min_attainable_p']:.3g})")


def _p(x):
    if x < 1e-4:
        return r'$<10^{-4}$'
    if x < 1e-3:
        return r'$<10^{-3}$'
    return f'${x:.3f}$'


def _tex(out):
    """Table VIII.

    One column of an IEEEtran journal page is 252pt, and the table is set inside a
    \resizebox, so its natural width decides the type size the reader actually gets.
    Keeping delta and its corrected p-value in one cell per dataset, rather than in
    four numeric columns, and keeping the group headers short, is what holds that
    width down; the test name carries the shared permutation p-value for the rewrite
    classes, whose p-value is one number for both datasets.
    """
    A, B = 'cifar10', 'cifar100'
    L = [r'\begin{tabular}{l l r r}', r'\toprule',
         r'& & \multicolumn{2}{c}{$\Delta$ points (Holm $p$)} \\',
         r'\cmidrule(lr){3-4}',
         r'\textbf{Comparison} & \textbf{Test} & CIFAR-10 & CIFAR-100 \\',
         r'\midrule',
         r'\multicolumn{4}{l}{\textit{Policies, at an equal budget '
         r'(Table~\ref{tab:searchvalue})}} \\']
    sv = out['search_value']
    for lab in ('tuned GA $-$ best screening', 'as-run GA $-$ best screening',
                'reg. evolution $-$ tuned GA', 'reg. evolution $S{=}10$ $-$ $S{=}3$'):
        a, b = sv[A]['tests'][lab], sv[B]['tests'][lab]
        tn = 'MWU' if a['test'] == 'mann-whitney' else 'WSR'
        L.append(rf"{lab} & {tn} & ${a['delta']:+.2f}$ ({_p(a['p_holm'])}) "
                 rf"& ${b['delta']:+.2f}$ ({_p(b['p_holm'])}) \\")
    L += [r'\midrule',
          r'\multicolumn{4}{l}{\textit{Settings, best value against the one the pipeline '
          r'fixed (Table~\ref{tab:settings})}} \\']
    st = out['settings']
    for lab in st[A]['tests']:
        a, b = st[A]['tests'][lab], st[B]['tests'][lab]
        L.append(rf"{lab} & WSR & ${a['delta']:+.2f}$ ({_p(a['p_holm'])}) "
                 rf"& ${b['delta']:+.2f}$ ({_p(b['p_holm'])}) \\")
    L += [r'\midrule',
          r'\multicolumn{4}{l}{\textit{Rewrite classes, against the operator each '
          r'replaced (Supplementary Table~S11)}} \\']
    op = out['operators']
    for k in ('Mutation', 'Initialization', 'Selection', 'Crossover'):
        a, b = op[A]['by_kind'].get(k), op[B]['by_kind'].get(k)
        pl = op['pooled_by_kind'].get(k)

        def cell(x):
            if x is None:
                return '--'
            return (rf"${x['mean']:+.2f}$ \scriptsize$[{x['ci_lo']:+.2f},"
                    rf"{x['ci_hi']:+.2f}]$")
        pp = '--' if pl is None else _p(pl['perm_p_holm'])
        L.append(rf"{k} rewrites & SSP ({pp}) & {cell(a)} & {cell(b)} \\")
    L += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(common.HERE, '..', 'tables', 'tab_tests.tex'),
         'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
