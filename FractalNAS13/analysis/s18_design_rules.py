"""Design rules for the FractalNet space, from 9,508 evaluated architectures.

Three things the released record supports and the original manuscript did not extract:
  (a) how much of the accuracy variance each of the 13 genes carries;
  (b) the best and worst value of each gene, as a marginal effect in accuracy points;
  (c) whether the rules composed greedily from (b) reach what a 34-run search found.

Writes results/design_rules.json, tables/tab_genes_effect.tex, tables/tab_rules.tex.
"""
import json, os, itertools
import numpy as np
import common
from sklearn.ensemble import RandomForestRegressor


def onehot(rows, cols=None):
    """One-hot design matrix over the 13 genes.

    `cols` fixes the column order, so a second call (a candidate genotype, say) lands in the
    same space as the fit.  A value absent from `cols` contributes nothing, which is the
    behaviour the greedy design below relies on.
    """
    if cols is None:
        seen = {}
        for r in rows:
            for g in common.GENES:
                seen.setdefault('%s=%s' % (g, r[g]), None)
        cols = sorted(seen)
    index = {c: i for i, c in enumerate(cols)}
    X = np.zeros((len(rows), len(cols)), float)
    for i, r in enumerate(rows):
        for g in common.GENES:
            j = index.get('%s=%s' % (g, r[g]))
            if j is not None:
                X[i, j] = 1.0
    return X, list(cols)

GRID = {'n_columns': [2, 3, 4], 'n_blocks': [2, 3, 4], 'base_channels': [16, 32, 64],
        'dropout_prob': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
        'fc_dropout': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
        'lr': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01],
        'momentum': [0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95, 0.98],
        'activation': ['ReLU', 'GELU', 'LeakyReLU', 'SiLU'],
        'kernel_size': [1, 3, 5, 7], 'pooling_type': ['Max', 'Avg'],
        'conv_type': ['Standard', 'Depthwise'], 'norm_type': ['BatchNorm', 'InstanceNorm'],
        'optimizer_type': ['SGD', 'AdamW', 'RMSprop']}
NICE = {'n_columns': '$n_\\text{columns}$', 'n_blocks': '$n_\\text{blocks}$',
        'base_channels': 'base channels', 'dropout_prob': 'dropout', 'fc_dropout': 'fc dropout',
        'lr': 'learning rate', 'momentum': 'momentum', 'activation': 'activation',
        'kernel_size': 'kernel size', 'pooling_type': 'pooling', 'conv_type': 'convolution',
        'norm_type': 'normalization', 'optimizer_type': 'optimizer'}


def main():
    table = [r for r in common.gene_table() if r['acc3'] is not None]
    out = {}
    for ds in common.DATASETS:
        rows = [r for r in table if r['dataset'] == ds]
        X, cols = onehot(rows)
        y = np.array([r['acc3'] for r in rows])
        rf = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1).fit(X, y)
        imp = dict(zip(cols, rf.feature_importances_))
        per_gene, marg = {}, {}
        for g in common.GENES:
            per_gene[g] = float(sum(v for k, v in imp.items() if k.startswith(g + '=')))
            m = {}
            for v in GRID[g]:
                sel = [r['acc3'] for r in rows if str(r[g]) == str(v)]
                if len(sel) >= 20:
                    m[str(v)] = [float(np.mean(sel)), len(sel)]
            if m:
                best = max(m, key=lambda k: m[k][0]); worst = min(m, key=lambda k: m[k][0])
                marg[g] = dict(values=m, best=best, worst=worst,
                               spread=m[best][0] - m[worst][0])
        order = sorted(common.GENES, key=lambda g: -per_gene[g])
        # (c) greedy design from the marginals, scored by the surrogate, against the search
        greedy = {g: marg[g]['best'] if g in marg else str(GRID[g][0]) for g in common.GENES}
        Xg, _ = onehot([{g: greedy[g] for g in common.GENES}], cols)
        pred_greedy = float(rf.predict(Xg)[0])
        # the same for the surrogate's own argmax over a large random sample of the grid
        rng = np.random.default_rng(3)
        cand = [{g: str(rng.choice([str(v) for v in GRID[g]])) for g in common.GENES}
                for _ in range(200000)]
        Xc, _ = onehot(cand, cols)
        pc = rf.predict(Xc)
        j = int(np.argmax(pc))
        out[ds] = dict(n=len(rows), importance=per_gene, order=order, marginal=marg,
                       best_observed=float(np.max(y)), mean_observed=float(np.mean(y)),
                       greedy_genotype=greedy, greedy_predicted=pred_greedy,
                       surrogate_argmax=cand[j], surrogate_argmax_predicted=float(pc[j]),
                       grid_size=int(np.prod([len(v) for v in GRID.values()])))
        # rule compliance: the top-4 genes at their best value
        top4 = order[:4]
        comp = []
        for k in range(5):
            sel = [r['acc3'] for r in rows
                   if sum(str(r[g]) == marg[g]['best'] for g in top4) == k]
            if len(sel) >= 15:
                comp.append([k, float(np.mean(sel)), len(sel)])
        out[ds]['compliance'] = comp
    json.dump(out, open(os.path.join(common.RES, 'design_rules.json'), 'w'), indent=1)
    _tex(out)
    for ds, o in out.items():
        print(f"\n== {ds}  n={o['n']}  (grid {o['grid_size']:,} points)")
        for g in o['order']:
            m = o['marginal'].get(g)
            if m:
                print(f"   {g:16s} importance {o['importance'][g]:.3f}  spread {m['spread']:5.2f} "
                      f"best={m['best']:>11s} worst={m['worst']}")
        print(f"   greedy-from-marginals genotype, surrogate estimate: {o['greedy_predicted']:.2f}")
        print(f"   surrogate argmax over 200k grid draws:              {o['surrogate_argmax_predicted']:.2f}")
        print(f"   best architecture the 34-run search actually found: {o['best_observed']:.2f}"
              f"  (search mean {o['mean_observed']:.2f})")
        print("   top-4 rules satisfied -> mean acc:", [(k, round(v, 2), n) for k, v, n in o['compliance']])


def _tex(out):
    L = ['\\begin{tabular}{l rr l rr l}', '\\toprule',
         '& \\multicolumn{3}{c}{CIFAR-10} & \\multicolumn{3}{c}{CIFAR-100} \\\\',
         '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}',
         '\\textbf{Gene} & \\textbf{Imp.} & \\textbf{Spread} & \\textbf{Best} '
         '& \\textbf{Imp.} & \\textbf{Spread} & \\textbf{Best} \\\\', '\\midrule']
    for g in out['cifar10']['order']:
        a, b = out['cifar10'], out['cifar100']
        ma, mb = a['marginal'].get(g), b['marginal'].get(g)
        if not (ma and mb):
            continue
        L.append(f"{NICE[g]} & {a['importance'][g]:.3f} & {ma['spread']:.1f} & "
                 f"{ma['best']} & {b['importance'][g]:.3f} & {mb['spread']:.1f} & {mb['best']} \\\\")
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(common.HERE, '..', 'tables', 'tab_genes_effect.tex'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
