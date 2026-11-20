"""Multi-panel figures for the paper.  Writes figures/fig_*.pdf (and .png previews).

Every panel reads a file in analysis/results/, so the figures carry exactly the numbers the
tables quote.  Palette, rcParams and the two column widths come from figstyle.py, so the whole
paper reads as one set.

Two figures go to the main paper (fig_bench, fig_settings) and the rest to the supplement:
fig_ops beside the gate tables it duplicates, plus fig_space and fig_pareto.  fig_extension
and fig_gate are drawn by s37_extension.py and s38_gate.py.

    python3 analysis/s57_figures.py            # every figure of both documents
    python3 analysis/s57_figures.py settings   # one of them
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

import common
from figstyle import C, save, J, W1, W2

DS = [('cifar10', 'CIFAR-10', C['c10']), ('cifar100', 'CIFAR-100', C['c100'])]


def _panel(ax, letter, title):
    ax.set_title(f'({letter}) {title}', loc='left', pad=3.0)


# ======================================================================== Fig. 1
def fig_bench():
    """Fidelity, cost and resolution: what the benchmark is before it is used."""
    p = J('bench_profile')
    #: four panels in one row; the height is what the main paper's page budget allows
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.78), constrained_layout=True)

    # (a) held-out predictions against measured accuracy
    a = ax[0]
    for key, lab, col in DS:
        s = p[key]['scatter']
        a.scatter(s['pred'], s['true'], s=0.7, alpha=0.18, color=col, linewidths=0,
                  label=f"{lab}, $\\rho$={p[key]['rho_heldout']:.3f}")
    lim = [0, 100]
    a.plot(lim, lim, '--', lw=0.7, color=C['grey'])
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.set_xlabel('surrogate, fitted on other runs')
    a.set_ylabel('measured 3-epoch accuracy')
    a.legend(loc='upper left', fontsize=5.0, markerscale=6, handletextpad=0.1)
    _panel(a, 'a', 'fidelity, held out')

    # (b) how fidelity grows with the labels the model is fitted on
    a = ax[1]
    for key, lab, col in DS:
        se = p[key]['sample_efficiency']
        n = sorted((int(k) for k in se), key=int)
        a.errorbar(n, [se[str(k)][0] for k in n], yerr=[se[str(k)][1] for k in n],
                   fmt='o-', ms=2.6, lw=1.1, color=col, label=lab, capsize=1.2, elinewidth=0.6)
        a.axhline(p[key]['rho_epoch1'], color=col, lw=0.8, ls=':')
    a.set_xscale('log')
    a.set_xlabel('architectures fitted on (log)')
    a.set_ylabel('Spearman $\\rho$, held out')
    a.text(0.03, 0.94, 'dotted: one measured training epoch', fontsize=5.0,
           color=C['grey'], transform=a.transAxes)
    a.legend(loc='lower right', fontsize=5.4)
    a.set_ylim(0.54, 0.99)
    _panel(a, 'b', 'what the record buys')

    # (c) cost of a search, three ways
    a = ax[2]
    c10 = p['cifar10']
    vals = [c10['sec_per_search'], c10['sec_per_1000_searches'],
            c10['real_hours_mean'] * 3600.0]
    labs = ['one\nsearch', '1,000\nreplicates', 'one real\nrun']
    a.bar(np.arange(3), vals, 0.56, color=[C['good'], C['good'], C['grey']])
    for i, v in enumerate(vals):
        txt = f'{v*1e3:.0f} ms' if v < 1 else (f'{v:.1f} s' if v < 1e3 else
                                               f'{v/3600:.1f} GPU-h')
        a.text(i, v * 1.9, txt, ha='center', fontsize=5.4)
    a.set_yscale('log')
    a.set_ylim(5e-3, 2e7)
    a.set_xticks(np.arange(3))
    a.set_xticklabels(labs, fontsize=5.4)
    a.set_xlim(-0.6, 2.6)
    a.set_ylabel('seconds (log)')
    _panel(a, 'c', 'cost')

    # (d) resolution against the replicate count
    a = ax[3]
    reps = p['reps']
    for key, lab, col in DS:
        a.plot(reps, [p[key]['resolution'][str(r)]['se'] for r in reps], 'o-', ms=2.6,
               color=col, label=lab)
        a.axhline(p[key]['real_baseline_se'], color=col, lw=0.9, ls='--')
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_xlabel('replicates (log)')
    a.set_ylabel('s.e. of the best found (pts)')
    a.text(0.30, 0.90, 'dashed: 3 real runs per arm', fontsize=5.0, color=C['grey'],
           transform=a.transAxes)
    a.legend(loc='lower left', fontsize=5.4)
    _panel(a, 'd', 'resolution')
    save(fig, 'fig_bench')


# ======================================================================== Fig. 2
def fig_settings():
    """What the benchmark prices: the search itself, then each setting."""
    sv, lv, p = J('search_value'), J('levers'), J('bench_profile')
    #: six panels over two rows; the height is what the main paper's page budget allows
    #: for a double-column float, and the panel labels are set in points, so shortening
    #: the figure does not shrink them
    fig, ax = plt.subplots(2, 3, figsize=(W2, 2.92), constrained_layout=True)

    # (a) the GA against free screening, at an equal number of trained architectures
    a = ax[0, 0]
    screens = ['1040', '5200', '52000', '520000']
    for key, lab, col in DS:
        s = sv[key]
        a.plot([int(k) for k in screens], [s['screen'][k]['best'] for k in screens], 'o-',
               ms=2.6, color=col, label=lab + ', screening')
        a.axhline(s['ga_asrun']['best'], color=col, lw=0.9, ls=':')
        a.axhline(s['ga_tuned']['best'], color=col, lw=0.9, ls='--')
        a.axhline(s['regularized_evolution']['best'], color=col, lw=0.9, ls='-.')
        # the best 3-epoch accuracy anywhere in the record: every level above this rule is the
        # landscape extrapolating past what was measured, which is the one thing a reader must
        # not mistake for an accuracy
        a.axhline(s['measured_max'], color=col, lw=1.2, ls='-', alpha=0.9)
    a.set_xscale('log')
    a.set_xlabel('candidates screened for free (log)')
    a.set_ylabel('best of 520 trained (pts)')
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], color=C['grey'], lw=0.9, ls=':', label='GA as run'),
            Line2D([], [], color=C['grey'], lw=0.9, ls='--', label='GA tuned'),
            Line2D([], [], color=C['grey'], lw=0.9, ls='-.', label='reg. evolution')]
    a.legend(handles=a.get_legend_handles_labels()[0] + keys, loc='center left',
             fontsize=4.3, borderaxespad=0.3, labelspacing=0.26, handlelength=2.4)
    a.margins(y=0.20)
    # name each rule where it sits, so the extrapolated region is legible without the caption
    for key, lab, col in DS:
        a.annotate('best measured', xy=(5.2e5, sv[key]['measured_max']),
                   xytext=(0, 1.3), textcoords='offset points', fontsize=4.3, color=col,
                   va='bottom', ha='right')
    _panel(a, 'a', 'is the search worth running?')

    # (b)-(e) one panel per setting
    sweeps = [('k', [2, 3, 5, 7, 10], 3, 'tournament size $k$', 'selection pressure', 'b'),
              ('pm', [0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4], 0.08,
               'effective mutation rate $p_m$', 'mutation rate', 'c'),
              ('pop', [10, 20, 40, 52], 20, 'population size', 'population size', 'd'),
              ('elite', [0, 1, 2, 4, 5, 8], 5, 'elite count', 'elitism', 'e')]
    cells = [ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1]]
    for (pre, xs, used, xlab, title, letter), a in zip(sweeps, cells):
        for key, lab, col in DS:
            s = lv[key]['settings']
            base = s[f'{pre}={used}']['best']
            v = [s[f'{pre}={x}']['best'] - base for x in xs]
            e = [s[f'{pre}={x}']['se'] for x in xs]
            a.errorbar(xs, v, yerr=e, fmt='o-', ms=2.6, lw=1.1, color=col, label=lab,
                       capsize=1.2, elinewidth=0.6)
        a.axvline(used, color=C['grey'], lw=0.8, ls=':')
        a.axhline(0, color='k', lw=0.6)
        a.annotate(' as run', xy=(used, 0.02), xycoords=('data', 'axes fraction'),
                   fontsize=5.0, color=C['grey'], va='bottom', ha='left')
        a.set_xlabel(xlab)
        a.set_ylabel('points against the setting used')
        if pre == 'pm':
            a.set_xscale('log')
            a.set_xticks(xs)
            a.set_xticklabels([str(x) for x in xs], fontsize=5.4, rotation=35,
                              ha='right', rotation_mode='anchor')
            a.tick_params(axis='x', pad=1.0)
            a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if letter == 'b':
            a.legend(loc='lower right', fontsize=5.4)
        _panel(a, letter, title)

    # (f) the two levers, and the segmentation sweep behind them
    a = ax[1, 2]
    segs = ['1x26', '2x13', '5x5', '13x2', '26x1']
    for key, lab, col in DS:
        s = p[key]['segmentation']
        base = s['1x26']['best']
        a.plot(range(len(segs)), [s[k]['best'] - base for k in segs], 'o-', ms=2.6,
               color=col, label=lab)
    a.axhline(0, color='k', lw=0.6)
    for key, lab, col in DS:
        sig = lv[key]['cells']['long_measured']['best'] - \
            lv[key]['cells']['long_predictor']['best']
        a.plot([0], [sig], '*', ms=7, color=col)
    a.text(0.34, 0.95, 'star: replacing the predictor\nfitness with the measurement',
           fontsize=5.0, color=C['grey'], transform=a.transAxes, va='top')
    a.set_xticks(range(len(segs)))
    a.set_xticklabels([s.replace('x', '$\\times$') for s in segs], fontsize=5.4)
    a.set_xlabel('segments $\\times$ generations (equal budget)')
    a.set_ylabel('points against one long run')
    a.legend(loc='lower left', fontsize=5.4)
    _panel(a, 'f', 'two levers on one scale')
    save(fig, 'fig_settings')


# ======================================================================== Fig. 3
def fig_space():
    """Reading the space, and forecasting one level past its edge."""
    dr, ex, sp = J('design_rules'), J('extension'), J('space')
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.9), constrained_layout=True)

    # (a) which genes carry the variance
    a = ax[0]
    order = dr['cifar10']['order'][::-1]
    y = np.arange(len(order))
    w = 0.38
    for i, (key, lab, col) in enumerate(DS):
        a.barh(y + (i - 0.5) * w, [dr[key]['importance'][g] for g in order], w,
               color=col, label=lab)
    a.set_yticks(y)
    a.set_yticklabels([g.replace('_', ' ') for g in order], fontsize=4.9)
    a.set_xlabel('share of explained variance')
    a.legend(loc='lower right', fontsize=5.2)
    _panel(a, 'a', 'what matters')

    # (b) the largest effect, and what it costs
    a = ax[1]
    g = 'n_columns'
    lv = sorted(sp['cifar10']['genes'][g]['values'], key=float)
    for key, lab, col in DS:
        v = sp[key]['genes'][g]['values']
        a.plot([float(x) for x in lv], [v[x]['acc'] for x in lv], 'o-', ms=3.0, color=col,
               label=lab)
    a2 = a.twinx()
    v = sp['cifar10']['genes'][g]['values']
    a2.plot([float(x) for x in lv], [v[x]['params'] for x in lv], 's:', ms=2.6,
            color=C['grey'])
    a2.set_ylabel('median params (M)', color=C['grey'], fontsize=6.0, labelpad=1.5)
    a2.tick_params(labelsize=5.6, colors=C['grey'])
    a2.margins(y=0.22)
    a2.grid(False)
    a.set_xticks([float(x) for x in lv])
    a.set_xlabel('fractal columns')
    a.set_ylabel('mean 3-epoch accuracy (pts)')
    a.margins(y=0.18)
    a.legend(loc='center left', fontsize=5.2, bbox_to_anchor=(0.02, 0.42))
    _panel(a, 'b', 'accuracy and cost')

    # (c) leave-one-level-out: a level the model never saw
    a = ax[2]
    for key, lab, col in DS:
        rows = list(ex[key]['loo'].values())
        a.scatter([r['true_mean'] for r in rows], [r['pred_mean'] for r in rows], s=11,
                  color=col, label=lab, zorder=3)
    lim = [min(a.get_xlim()[0], a.get_ylim()[0]), max(a.get_xlim()[1], a.get_ylim()[1])]
    a.plot(lim, lim, '--', lw=0.7, color=C['grey'])
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.set_xlabel('measured mean of the held-out level')
    a.set_ylabel('predicted, never having seen it')
    a.legend(loc='upper left', fontsize=5.2)
    _panel(a, 'c', 'the extrapolator, tested')

    # (d) the forecasts themselves
    a = ax[3]
    names, vals, los, his, cols = [], [], [], [], []
    for key, lab, col in DS:
        for g, f in sorted(ex[key]['forecast'].items()):
            arrow = '$\\downarrow$' if f['side'] == 'low' else '$\\uparrow$'
            names.append(f"{g.replace('_', ' ')} {arrow}\n{lab}")
            vals.append(f['pred_gain'])
            los.append(f['pred_gain'] - f['lo'])
            his.append(f['hi'] - f['pred_gain'])
            cols.append(col)
    y = np.arange(len(names))[::-1]
    a.barh(y, vals, 0.6, xerr=[los, his], color=cols, error_kw=dict(lw=0.6))
    a.axvline(0, color='k', lw=0.7)
    a.set_yticks(y)
    a.set_yticklabels(names, fontsize=4.6)
    a.set_xlabel('forecast gain one level out (pts)')
    _panel(a, 'd', 'which edge to cross')
    save(fig, 'fig_space')


# ======================================================================== Fig. S2
def fig_pareto():
    """Accuracy against parameter count for every evaluated architecture, coloured by fractal
    width, with the Pareto front marked.

    The cloud is data/cost_model.csv, which s28_space.py writes with one row per evaluated
    architecture; the front is results/space.json, which the same script computes over the
    same (params, accuracy) pairs, so the two cannot disagree.  The colour is n_columns, read
    from the record through common.gene_table(), joined on the architecture's uid.
    """
    import csv
    sp = J('space')
    width = {r['uid']: r['n_columns'] for r in common.gene_table()}
    rows = list(csv.DictReader(open(os.path.join(common.DATA, 'cost_model.csv'))))

    fig, axes = plt.subplots(1, 2, figsize=(W1 * 2, 2.5))
    for a, ds, lab in zip(axes, common.DATASETS, ('CIFAR-10', 'CIFAR-100')):
        rs = [r for r in rows if r['dataset'] == ds]
        x = np.array([float(r['params_M']) for r in rs])
        y = np.array([float(r['acc_epoch3_true']) for r in rs])
        w = np.array([width.get(r['uid'], np.nan) for r in rs], float)
        levels = sorted({v for v in np.unique(w) if np.isfinite(v)})
        # the paper's claim is that *fewer* columns is better, so the smallest level is
        # the most visible and the largest fades out
        shades = [C['c10'], C['alt'], C['light'], C['grey']]
        for i, lv_ in enumerate(levels):
            m = w == lv_
            a.scatter(x[m], y[m], s=2.0, lw=0, alpha=0.45,
                      color=shades[i % len(shades)], label=r'$n\_columns=%g$' % lv_)
        front = sorted(sp[ds]['pareto'], key=lambda p: p['params'])
        a.plot([p['params'] for p in front], [p['acc'] for p in front],
               '-o', color=C['good'], lw=1.1, ms=2.6, label='Pareto front')
        a.set_xscale('log')
        a.set_xlabel('parameters (M, log)')
        a.set_ylabel('measured 3-epoch accuracy (pts)')
        a.set_title(lab, fontsize=7)
        a.legend(fontsize=4.6, loc='lower right', frameon=False)
    fig.tight_layout()
    save(fig, 'fig_pareto')


# ======================================================================== Fig. 4
def fig_ops():
    """The 90 rewrites: what they changed, what it was worth, and what the gate is worth."""
    ob, gt, lv = J('operator_bench'), J('gate'), J('levers')
    au = json.load(open(os.path.join(common.RES, 'audit.json')))
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.9), constrained_layout=True)

    KIND = [('mutate_gene', 'mutation'), ('_create_random_chromosome', 'initialization'),
            ('select_competitor', 'selection'), ('combine_genes', 'crossover')]

    # (a) every replayed rewrite, by the operator it replaces
    a = ax[0]
    rng = np.random.default_rng(0)
    for i, (k, lab) in enumerate(KIND):
        for key, _, col in DS:
            d = [r['delta'] for r in ob[key]['rows']
                 if r.get('kind') == k and 'delta' in r]
            if not d:
                continue
            a.scatter(np.array(d), i + rng.uniform(-0.17, 0.17, len(d)), s=7, color=col,
                      alpha=0.8, linewidths=0)
            a.plot([np.mean(d)], [i], '|', ms=11, color=col, mew=1.4)
    a.axvline(0, color='k', lw=0.7)
    a.set_yticks(range(len(KIND)))
    a.set_yticklabels([l for _, l in KIND], fontsize=5.4)
    a.invert_yaxis()
    a.set_xlabel('$\\Delta$ best found (pts)')
    a.set_xlim(-4.5, 2.5)
    _panel(a, 'a', 'replayed one at a time')

    # (b) the parameter they all moved, against the optimum the benchmark locates
    a = ax[1]
    xs = [0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4]
    for key, lab, col in DS:
        s = lv[key]['settings']
        base = s['pm=0.08']['best']
        a.plot(xs, [s[f'pm={x}']['best'] - base for x in xs], 'o-', ms=2.6, color=col,
               label=lab)
    for i, (c, x) in enumerate(zip((2, 3, 4, 5), (0.08, 0.12, 0.16, 0.20))):
        a.axvline(x, color=C['alt'], lw=0.8, ls='--', alpha=0.8)
        a.annotate(f'$c$={c}', xy=(x, 0.03 if i % 2 == 0 else 0.13),
                   xycoords=('data', 'axes fraction'), textcoords='offset points',
                   xytext=(1.5, 0), fontsize=4.8, color=C['alt'], va='bottom', ha='left')
    a.axhline(0, color='k', lw=0.6)
    a.set_xlabel('effective mutation rate (log)')
    a.set_ylabel('points against 0.08 (as run)')
    a.set_xscale('log')
    a.set_xticks(xs)
    a.set_xticklabels([str(x) for x in xs], fontsize=5.4, rotation=35, ha='right',
                      rotation_mode='anchor')
    a.tick_params(axis='x', pad=1.0)
    a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a.margins(y=0.18)
    a.legend(loc='upper right', fontsize=5.2, borderaxespad=0.2)
    _panel(a, 'b', 'the mutation rate')

    # (c) the acceptance gate
    a = ax[2]
    gates = [('none', 'parse $+$ smoke'), ('property', '$+$ property test'),
             ('property+', '$+$ replay')]
    x = np.arange(len(gates))
    for i, (key, lab, col) in enumerate(DS):
        v = [gt[key]['gates'][g]['delta_mean'] for g, _ in gates]
        e = [gt[key]['gates'][g]['delta_se'] for g, _ in gates]
        a.bar(x + (i - 0.5) * 0.34, v, 0.32, yerr=e, color=col, label=lab,
              error_kw=dict(lw=0.6))
    a.axhline(0, color='k', lw=0.7)
    a.set_xticks(x)
    a.set_xticklabels([l for _, l in gates], fontsize=5.0, rotation=14, ha='right')
    a.set_ylabel('$\\Delta$ best found, per proposal (pts)')
    a.legend(loc='upper left', fontsize=5.2)
    _panel(a, 'c', 'what the gate is worth')

    # (d) what the pipeline rewarded, against what the rewrite was worth
    a = ax[3]
    delta = {}
    for key, _, _ in DS:
        for r in ob[key]['rows']:
            if 'delta' in r:
                delta[(key, r['file'])] = r['delta']
    cats = [('modified', 'behaviour changed', C['good']),
            ('equivalent', 'behaviourally equivalent', C['grey'])]
    for cat, lab, col in cats:
        x = [r['reward'] for r in au
             if r['cat'] == cat and (r['ds'], r['file']) in delta]
        y = [delta[(r['ds'], r['file'])] for r in au
             if r['cat'] == cat and (r['ds'], r['file']) in delta]
        a.scatter(x, y, s=9, color=col, alpha=0.85, linewidths=0, label=lab)
    a.axhline(0, color='k', lw=0.7)
    a.set_xlabel('reward the pipeline paid')
    a.set_ylabel('$\\Delta$ best found (pts)')
    a.set_ylim(-4.5, 2.5)
    a.legend(loc='lower right', fontsize=5.0)
    _panel(a, 'd', 'the reward is not the value')
    save(fig, 'fig_ops')


if __name__ == '__main__':
    import sys
    want = sys.argv[1:] or ['bench', 'settings', 'space', 'pareto', 'ops']
    for w in want:
        globals()['fig_' + w]()
