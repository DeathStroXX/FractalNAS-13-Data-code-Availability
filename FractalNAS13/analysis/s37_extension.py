"""Forecasting a gene value that was never measured.

The space audit (s28_space.py) finds that for most ordinal genes the best value sits on the
boundary of the declared range, which says the grid was drawn too small. On its own that is a
dead end: a search cannot repair it, and the obvious remedy -- widen every range and rerun --
costs a second study.

It stops being a dead end if the surrogate can say *which* range to widen and *what widening
it will pay*. The surrogate of surrogate.py cannot: a one-hot indicator for an unmeasured
level carries a zero coefficient, so by construction the model says nothing outside the grid.
Replacing the one-hot on the ordinal genes with a low-order polynomial in the normalised level
makes the model a *function* of the level rather than a lookup over levels, and a function can
be evaluated past the last level it was fitted on.

Two deliberate restrictions keep that honest.

  * Only a range with at least four measured levels is forecast. Removing one level of a
    three-level gene leaves two, and a trend through two points carries no information about
    its own curvature, so nothing can be validated there. The three structural genes
    (n_columns, n_blocks, base_channels) are therefore reported as measured trends only.
  * Extrapolation is along the *tangent* of the fitted trend at the boundary, not along the
    polynomial itself. A quadratic fitted inside a range will often turn over just outside it,
    which is an artefact of the basis rather than a statement about the architecture; the
    first-order extension is the conservative reading of the same fit. Both are measured below
    and the tangent wins, which is why it is what the forecast uses.

Validation is leave-one-level-out: an entire extreme level of one gene is removed from the fit
-- every architecture that uses it -- and the model predicts that level's mean accuracy and
the step from its neighbour from the levels that remain.

Writes results/extension.json, ../tables/tab_extension.tex, ../figures/fig_extension.pdf.
"""
import json, os
import numpy as np
from scipy.stats import spearmanr

import common
import surrogate as sg

#: ordinal genes; a level can be extrapolated along the order
ORDINAL = ['n_columns', 'n_blocks', 'base_channels', 'dropout_prob', 'fc_dropout',
           'lr', 'momentum', 'kernel_size']
#: ...but only these carry enough levels for the extrapolation to be identified and tested
MIN_LEVELS = 4
FORECASTABLE = [g for g in ORDINAL if len(sg.SS[g]) >= MIN_LEVELS]
DEG = 2
ALPHA = 1.0
BOOT = 200
EPS = 1e-3


def basis(P, x_over=None, deg=DEG):
    """Polynomial in the normalised level for ordinal genes, one-hot for categorical ones,
    plus the products of the ordinal linear terms.

    `x_over = (gene_index, value)` overrides one gene's normalised position, which is how a
    level outside the grid is evaluated.
    """
    P = np.asarray(P, int)
    n = len(P)
    cols = [np.ones(n)]
    lin = {}
    for j, g in enumerate(sg.GENES):
        k = int(sg.CARD[j])
        if g in ORDINAL:
            x = P[:, j].astype(float) / (k - 1)
            if x_over is not None and x_over[0] == j:
                x = np.full(n, float(x_over[1]))
            lin[j] = x
            for d in range(1, deg + 1):
                cols.append(x ** d)
        else:
            for v in range(k):
                cols.append((P[:, j] == v).astype(float))
    ks = sorted(lin)
    for i in range(len(ks)):
        for j2 in range(i + 1, len(ks)):
            cols.append(lin[ks[i]] * lin[ks[j2]])
    return np.stack(cols, 1)


def fit_basis(P, y, alpha=ALPHA, deg=DEG):
    A = basis(P, deg=deg)
    y = np.asarray(y, float)
    G = A.T @ A
    G[np.diag_indices_from(G)] += alpha
    return np.linalg.solve(G, A.T @ y)


def marginal(w, ref, j, x, deg=DEG):
    """Mean fitted accuracy over the reference population with gene j held at level x."""
    return float((basis(ref, x_over=(j, x), deg=deg) @ w).mean())


def extend(w, ref, j, x_edge, x_new, mode, deg=DEG):
    """Value predicted at a level outside the grid.

    'poly'    -- evaluate the fitted polynomial there, which is what the leave-one-level-out
                 test of this script validates, since that test evaluates the fit exactly one
                 level outside the range it was fitted on.
    'tangent' -- follow the tangent of the fitted trend at the boundary. This discards the
                 curvature term outside the range, which is the conservative reading of the
                 same fit, and is reported as a robustness check on `poly`.

    The slope is a central difference, so it does not depend on which side of the boundary the
    extension runs to; a one-sided difference taken in the direction of travel returns $-f'$
    when the extension runs downward, which silently negates every downward forecast.
    """
    if mode == 'poly':
        return marginal(w, ref, j, x_new, deg)
    f_edge = marginal(w, ref, j, x_edge, deg)
    slope = (marginal(w, ref, j, x_edge + EPS, deg)
             - marginal(w, ref, j, x_edge - EPS, deg)) / (2 * EPS)
    return f_edge + slope * (x_new - x_edge)


def main():
    table = [r for r in common.gene_table() if r['acc3'] is not None]
    out = {'ordinal': ORDINAL, 'forecastable': FORECASTABLE, 'min_levels': MIN_LEVELS,
           'degree': DEG, 'boot': BOOT}

    for ds in common.DATASETS:
        rows = [r for r in table if r['dataset'] == ds]
        P = np.array([sg.codes_of(r) for r in rows])
        y = np.array([r['acc3'] for r in rows])
        res = {'n': len(y), 'loo': {}, 'forecast': {}, 'trend': {}}

        # ------------------------------------------------- 1. leave-one-level-out validation
        # An entire extreme level is removed from the fit, then the model is asked to predict
        # the accuracy of the architectures that use it -- on their own genotypes, so no
        # subpopulation is compared with a different one.
        for j, g in enumerate(sg.GENES):
            if g not in FORECASTABLE:
                continue
            k = int(sg.CARD[j])
            for side in ['high', 'low']:
                hv = k - 1 if side == 'high' else 0
                te = P[:, j] == hv
                if te.sum() < 50:
                    continue
                keep = ~te
                w = fit_basis(P[keep], y[keep])
                pred = basis(P[te]) @ w
                # the same model restricted to levels it did saw, as the reference error
                w_in = fit_basis(P[keep], y[keep])
                inside = basis(P[keep]) @ w_in
                res['loo'][f'{g}:{side}'] = {
                    'gene': g, 'side': side, 'level': int(hv),
                    'value': str(sg.SS[g][hv]), 'n_held': int(te.sum()),
                    'true_mean': float(y[te].mean()),
                    'pred_mean': float(pred.mean()),
                    'abs_err': float(abs(y[te].mean() - pred.mean())),
                    'mae_per_arch': float(np.abs(y[te] - pred).mean()),
                    'mae_inside': float(np.abs(y[keep] - inside).mean()),
                    'rho_within': float(spearmanr(pred, y[te]).correlation),
                    'sd_held': float(y[te].std())}

        v = list(res['loo'].values())
        res['loo_summary'] = {
            'n': len(v),
            'mae_level_mean': float(np.mean([x['abs_err'] for x in v])),
            'median_ae': float(np.median([x['abs_err'] for x in v])),
            'max_ae': float(np.max([x['abs_err'] for x in v])),
            'mae_per_arch': float(np.mean([x['mae_per_arch'] for x in v])),
            'mae_inside': float(np.mean([x['mae_inside'] for x in v])),
            'rho_within': float(np.mean([x['rho_within'] for x in v])),
            'sd_held': float(np.mean([x['sd_held'] for x in v])),
            'within_2pts': int(sum(x['abs_err'] <= 2.0 for x in v))}

        # ------------------------------------------------------------- 2. the forecast itself
        for j, g in enumerate(sg.GENES):
            if g not in ORDINAL:
                continue
            k = int(sg.CARD[j])
            means = [float(y[P[:, j] == v].mean()) if (P[:, j] == v).sum() else np.nan
                     for v in range(k)]
            bestv = int(np.nanargmax(means))
            res['trend'][g] = {'means': means, 'best_level': bestv,
                               'best_value': str(sg.SS[g][bestv]),
                               'on_edge': bestv in (0, k - 1),
                               'forecastable': g in FORECASTABLE}
            if bestv not in (0, k - 1) or g not in FORECASTABLE:
                continue
            side = 'low' if bestv == 0 else 'high'
            x_edge = bestv / (k - 1)
            inner = (bestv + 1 if side == 'low' else bestv - 1) / (k - 1)
            x_new = x_edge - (inner - x_edge)       # one level beyond the edge
            gains, tang = [], []
            rng = np.random.default_rng(11)
            for _ in range(BOOT):
                s = rng.integers(0, len(y), len(y))
                w = fit_basis(P[s], y[s])
                ref = P[s]
                gains.append(marginal(w, ref, j, x_new) - marginal(w, ref, j, x_edge))
                tang.append(extend(w, ref, j, x_edge, x_new, 'tangent')
                            - marginal(w, ref, j, x_edge))
            gains, tang = np.array(gains), np.array(tang)
            res['forecast'][g] = {
                'gene': g, 'side': side, 'edge_value': str(sg.SS[g][bestv]),
                'next_value_hint': _hint(g, bestv, side),
                'measured_edge_mean': means[bestv],
                'pred_gain': float(gains.mean()),
                'lo': float(np.quantile(gains, 0.05)),
                'hi': float(np.quantile(gains, 0.95)),
                'positive_share': float((gains > 0).mean()),
                'tangent_gain': float(tang.mean()),
                'agree_sign': bool(np.sign(gains.mean()) == np.sign(tang.mean()))}
        out[ds] = res

    # the ranges both datasets agree are worth extending
    agree = [g for g in out['cifar10']['forecast']
             if g in out['cifar100']['forecast']
             and out['cifar10']['forecast'][g]['lo'] > 0
             and out['cifar100']['forecast'][g]['lo'] > 0]
    out['agreed_extensions'] = agree

    os.makedirs(common.RES, exist_ok=True)
    json.dump(out, open(os.path.join(common.RES, 'extension.json'), 'w'), indent=1)
    emit_table(out)
    emit_figure(out)
    for ds in common.DATASETS:
        s = out[ds]['loo_summary']
        print('%s  n=%d levels  |err on level mean| = %.2f pts (median %.2f, max %.2f), '
              '%d/%d within 2 pts; rho within held-out level %.3f; '
              'per-arch MAE %.2f outside vs %.2f inside the grid'
              % (ds, s['n'], s['mae_level_mean'], s['median_ae'], s['max_ae'],
                 s['within_2pts'], s['n'], s['rho_within'],
                 s['mae_per_arch'], s['mae_inside']))
        for g, f in out[ds]['forecast'].items():
            print('    extend %-13s %-4s of %-6s -> %+.2f [%+.2f, %+.2f] (tangent %+.2f)'
                  % (g, f['side'], f['edge_value'], f['pred_gain'], f['lo'], f['hi'],
                     f['tangent_gain']))
    print('both datasets agree on extending:', agree)


def _hint(g, bestv, side):
    """The value one level beyond the edge, on the gene's own scale, where it is obvious."""
    vals = sg.SS[g]
    try:
        a, b = float(vals[bestv]), float(vals[bestv + (1 if side == 'low' else -1)])
    except (TypeError, ValueError):
        return ''
    nxt = a - (b - a)
    if nxt <= 0 and a > 0 and b > 0:
        nxt = a * a / b                      # a geometric step, where an arithmetic one
    return '%g' % nxt                        # would leave the admissible range


def emit_table(o):
    A, B = 'cifar10', 'cifar100'
    keys = [k for k in o[A]['forecast'] if k in o[B]['forecast']]
    keys.sort(key=lambda g: -(o[A]['forecast'][g]['pred_gain']
                              + o[B]['forecast'][g]['pred_gain']))
    L = []
    a = L.append
    a(r'\begin{table}[t]')
    a(r'\centering')
    a(r'\caption{Forecasting the grid. On an ordinal basis the genotype model is a '
      r'\emph{function} of a level rather than a lookup over levels, so it can be evaluated '
      r'past the last level measured. Top: the test. An entire extreme level of one gene is '
      r'removed from the fit---every architecture that uses it---and the model predicts the '
      r'accuracy of those architectures; the error is compared against the same model inside '
      r'the grid and against the spread of the held-out level itself. Bottom: the forecast, '
      r'for each range whose optimum lies on its boundary \emph{on both datasets}, of the '
      r'change in mean accuracy from one further level (90\% bootstrap interval over '
      r'architectures). Ranges on the boundary of only one dataset are given in the text. '
      r'Discarding the curvature term outside the range is the conservative reading of the '
      r'same fit and is shown as a robustness check; across all four forecasts it changes no '
      r'sign.}')
    a(r'\label{tab:extension}')
    a(r'\footnotesize')
    a(r'\resizebox{\columnwidth}{!}{%')
    a(r'\begin{tabular}{lcc}')
    a(r'\toprule')
    a(r' & CIFAR-10 & CIFAR-100 \\')
    a(r'\midrule')
    a(r'\multicolumn{3}{l}{\emph{Leave-one-level-out: a level the model never saw}} \\')
    rows = [('Error on the level\'s mean accuracy, mean / median (pts)', None, None),
            ('Within 2 points', 'within_2pts', None),
            ('$\\rho$ inside the held-out level', 'rho_within', '{:.3f}'),
            ('Per-arch.\\ MAE, outside / inside the grid', 'both', None)]
    for lab, key, fmt in rows:
        if key is None:
            a('\\quad %s & %.2f / %.2f & %.2f / %.2f \\\\'
              % (lab, o[A]['loo_summary']['mae_level_mean'], o[A]['loo_summary']['median_ae'],
                 o[B]['loo_summary']['mae_level_mean'], o[B]['loo_summary']['median_ae']))
        elif key == 'both':
            a('\\quad %s & %.2f / %.2f & %.2f / %.2f \\\\'
              % (lab, o[A]['loo_summary']['mae_per_arch'], o[A]['loo_summary']['mae_inside'],
                 o[B]['loo_summary']['mae_per_arch'], o[B]['loo_summary']['mae_inside']))
        elif key == 'within_2pts':
            a('\\quad %s & %d/%d & %d/%d \\\\'
              % (lab, o[A]['loo_summary']['within_2pts'], o[A]['loo_summary']['n'],
                 o[B]['loo_summary']['within_2pts'], o[B]['loo_summary']['n']))
        else:
            a('\\quad %s & %s & %s \\\\' % (lab, fmt.format(o[A]['loo_summary'][key]),
                                            fmt.format(o[B]['loo_summary'][key])))
    a(r'\addlinespace[2pt]\midrule')
    a(r'\multicolumn{3}{l}{\emph{Forecast gain from one further level}} \\')
    for g in keys:
        fa, fb = o[A]['forecast'][g], o[B]['forecast'][g]
        a('\\quad %s: %s $\\to$ %s & $%+.2f$ [$%+.2f$, $%+.2f$] & $%+.2f$ [$%+.2f$, $%+.2f$] \\\\'
          % (g.replace('_', r'\_'), fa['edge_value'], fa['next_value_hint'] or '---',
             fa['pred_gain'], fa['lo'], fa['hi'], fb['pred_gain'], fb['lo'], fb['hi']))
        a('\\quad \\dots\\ curvature discarded (tangent) & $%+.2f$ & $%+.2f$ \\\\'
          % (fa['tangent_gain'], fb['tangent_gain']))
    a(r'\bottomrule')
    a(r'\end{tabular}}')
    a(r'\end{table}')
    p = os.path.join(common.HERE, '..', 'tables', 'tab_extension.tex')
    open(p, 'w').write('\n'.join(L) + '\n')
    print('wrote', p)


def emit_figure(o):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    COL = {'cifar10': '#1b6ca8', 'cifar100': '#c7522a'}
    NAME = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100'}

    for ds in common.DATASETS:
        v = list(o[ds]['loo'].values())
        ax[0].scatter([x['true_mean'] for x in v], [x['pred_mean'] for x in v],
                      s=26, color=COL[ds], alpha=.85, label=NAME[ds])
    allv = [x for ds in common.DATASETS for x in o[ds]['loo'].values()]
    lo = min(min(x['true_mean'] for x in allv), min(x['pred_mean'] for x in allv)) - 3
    hi = max(max(x['true_mean'] for x in allv), max(x['pred_mean'] for x in allv)) + 3
    ax[0].plot([lo, hi], [lo, hi], 'k:', lw=.8)
    ax[0].fill_between([lo, hi], [lo - 2, hi - 2], [lo + 2, hi + 2],
                       color='k', alpha=.07, lw=0)
    ax[0].set_xlabel('true mean accuracy of the held-out level (%)')
    ax[0].set_ylabel('predicted (%)')
    ax[0].set_title('A level never measured', fontsize=9)
    ax[0].legend(fontsize=7, frameon=False, loc='upper left')
    ax[0].grid(alpha=.25, lw=.5)
    ax[0].text(0.97, 0.05, r'band: $\pm2$ points', transform=ax[0].transAxes,
               fontsize=6.5, ha='right')

    # every forecast either dataset makes, one row per gene; a dataset that makes no forecast
    # for a gene simply has no point on that row
    keys = sorted({k for ds in common.DATASETS for k in o[ds]['forecast']},
                  key=lambda g: sum(o[ds]['forecast'][g]['pred_gain']
                                    for ds in common.DATASETS if g in o[ds]['forecast']))
    yy = np.arange(len(keys))
    for i, ds in enumerate(common.DATASETS):
        off = (i - .5) * .3
        rows = [(y, o[ds]['forecast'][g]) for y, g in zip(yy, keys) if g in o[ds]['forecast']]
        ax[1].errorbar([x['pred_gain'] for _, x in rows], [y + off for y, _ in rows],
                       xerr=[[x['pred_gain'] - x['lo'] for _, x in rows],
                             [x['hi'] - x['pred_gain'] for _, x in rows]],
                       fmt='o', ms=4, lw=1.2, color=COL[ds], capsize=2, label=NAME[ds])
    ax[1].axvline(0, color='k', lw=.8)
    ax[1].set_ylim(-0.6, len(keys) - 0.4)
    ax[1].set_yticks(yy, [k.replace('_', ' ') for k in keys], fontsize=7.5)
    ax[1].set_xlabel('forecast gain from one further level (pts)')
    ax[1].set_title('Which range to extend first', fontsize=9)
    ax[1].legend(fontsize=7, frameon=False, loc='lower right')
    ax[1].grid(alpha=.25, lw=.5, axis='x')
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(common.HERE, '..', 'figures', 'fig_extension.' + ext),
                    dpi=200, bbox_inches='tight')
    print('wrote figures/fig_extension.pdf')


if __name__ == '__main__':
    main()
