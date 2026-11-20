"""The operator effect at segment level, where there are 90 observations, not 9 runs.

Each accepted rewrite is active for exactly one inner segment, and the logs record that
segment's own best 3-epoch accuracy. That gives 90 paired observations against the 9 runs the
original operator contrast used. A mixed-effects model with the run as a random intercept
prices each audit class in accuracy points, which is what justifies (or does not justify) the
property-based testing step.

Writes results/segment_model.json, tables/tab_segment.tex.
"""
import json, os, collections
import numpy as np
import common

CATS = ['modified', 'equivalent', 'off-grid', 'out-of-space', 'raises/hangs']
DEFECT = {'off-grid', 'out-of-space', 'raises/hangs'}


def main():
    audit = json.load(open(os.path.join(common.RES, 'audit.json')))
    rec = [a for a in audit if a.get('inner_best3') is not None]
    out = {}
    for ds in common.DATASETS:
        rows = [a for a in rec if a['ds'] == ds]
        y = np.array([a['inner_best3'] for a in rows], float)
        run = np.array([f"{a['model']}_r{a['run']}" for a in rows])
        cat = np.array([a['cat'] for a in rows])
        it = np.array([a['it'] for a in rows], float)
        # centre within run: removes every between-run difference without discarding segments
        yc = y.copy()
        for r in set(run):
            m = run == r
            yc[m] -= y[m].mean()
        o = {'n_segments': len(rows), 'n_runs': len(set(run)),
             'by_cat': {}, 'defect_cost': None}
        for c in CATS:
            m = cat == c
            if m.sum() == 0:
                continue
            o['by_cat'][c] = dict(n=int(m.sum()), raw=float(y[m].mean()),
                                  within_run=float(yc[m].mean()),
                                  sd=float(yc[m].std()))
        d = cat_in(cat, DEFECT)
        if d.sum() and (~d).sum():
            diff = float(yc[~d].mean() - yc[d].mean())
            o['defect_cost'] = dict(delta=diff, n_defect=int(d.sum()), n_clean=int((~d).sum()),
                                    p=perm(yc[~d], yc[d]))
        m = cat == 'modified'; e = cat == 'equivalent'
        if m.sum() and e.sum():
            o['modified_vs_equivalent'] = dict(delta=float(yc[m].mean() - yc[e].mean()),
                                               p=perm(yc[m], yc[e]),
                                               n_mod=int(m.sum()), n_eq=int(e.sum()))
        # does the reward the pipeline fed back track the segment outcome?
        rw = np.array([a['reward'] if a.get('reward') is not None else np.nan for a in rows])
        ok = ~np.isnan(rw)
        if ok.sum() > 10:
            from scipy.stats import spearmanr
            o['reward_vs_outcome_rho'] = float(spearmanr(rw[ok], yc[ok]).statistic)
            o['reward_by_cat'] = {c: float(np.nanmean(rw[cat == c]))
                                  for c in CATS if (cat == c).sum()}
        # mixed model, if statsmodels is available
        try:
            import pandas as pd, statsmodels.formula.api as smf
            df = pd.DataFrame(dict(y=y, run=run, cat=cat, it=it))
            df['cat'] = pd.Categorical(df['cat'],
                                       categories=['modified'] + [c for c in CATS if c != 'modified'])
            fit = smf.mixedlm('y ~ C(cat)', df, groups=df['run']).fit()
            o['mixedlm'] = {k: [float(fit.params[k]), float(fit.pvalues[k])]
                            for k in fit.params.index if k != 'Group Var'}
        except Exception as ex:
            o['mixedlm'] = f'unavailable: {type(ex).__name__}'
        out[ds] = o
    json.dump(out, open(os.path.join(common.RES, 'segment_model.json'), 'w'), indent=1)
    _tex(out)
    for ds, o in out.items():
        print(f"\n== {ds}  {o['n_segments']} segments over {o['n_runs']} runs")
        for c, v in o['by_cat'].items():
            print(f"   {c:14s} n={v['n']:2d}  raw {v['raw']:6.2f}  within-run {v['within_run']:+6.2f}")
        if o['defect_cost']:
            d = o['defect_cost']
            print(f"   defective vs clean: {d['delta']:+.2f} points per segment "
                  f"(n={d['n_defect']} vs {d['n_clean']}, permutation p={d['p']:.4f})")
        if 'modified_vs_equivalent' in o:
            m = o['modified_vs_equivalent']
            print(f"   sound modification vs equivalent: {m['delta']:+.2f} (p={m['p']:.3f})")
        if 'reward_vs_outcome_rho' in o:
            print(f"   pipeline reward vs segment outcome: rho={o['reward_vs_outcome_rho']:+.3f}")
            print(f"   mean reward by class: "
                  + ", ".join(f"{k}={v:.2f}" for k, v in o['reward_by_cat'].items()))
        print(f"   mixedlm: {o['mixedlm']}")


def cat_in(cat, s):
    return np.array([c in s for c in cat])


def perm(a, b, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = abs(a.mean() - b.mean())
    z = np.concatenate([a, b]); na = len(a)
    cnt = 0
    for _ in range(n):
        rng.shuffle(z)
        cnt += abs(z[:na].mean() - z[na:].mean()) >= obs - 1e-12
    return (cnt + 1) / (n + 1)


def _tex(out):
    lab = {'modified': 'Sound modification', 'equivalent': 'Behaviourally equivalent',
           'off-grid': 'Off-grid output', 'out-of-space': 'Out-of-space output',
           'raises/hangs': 'Raises or hangs'}
    L = ['\\begin{tabular}{l rr rr}', '\\toprule',
         '& \\multicolumn{2}{c}{CIFAR-10} & \\multicolumn{2}{c}{CIFAR-100} \\\\',
         '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
         '\\textbf{Audit class of the active rewrite} & $n$ & \\textbf{Within-run} & $n$ '
         '& \\textbf{Within-run} \\\\', '\\midrule']
    for c in CATS:
        a = out['cifar10']['by_cat'].get(c); b = out['cifar100']['by_cat'].get(c)
        if not (a or b):
            continue
        L.append(f"{lab[c]} & {a['n'] if a else 0} & "
                 f"{('$%+.2f$' % a['within_run']) if a else '--'} & {b['n'] if b else 0} & "
                 f"{('$%+.2f$' % b['within_run']) if b else '--'} \\\\")
    a, b = out['cifar10']['defect_cost'], out['cifar100']['defect_cost']
    L += ['\\midrule',
          f"\\textit{{Clean $-$ defective}} & & $+{a['delta']:.2f}$ & & $+{b['delta']:.2f}$ \\\\",
          '\\bottomrule', '\\end{tabular}']
    open(os.path.join(common.HERE, '..', 'tables', 'tab_segment.tex'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
