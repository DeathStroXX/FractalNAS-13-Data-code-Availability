"""Auditing the search space itself: what a parameter buys, and where the optimum sits.

(a) Cost. The genotype determines the network, so parameters and multiply-accumulates follow
    analytically: a fractal block of C columns holds 2^C - 1 convolutions, channels double and
    the map halves at every block. The model is validated against the one measured cost signal
    the logs contain -- the training throughput printed by the frozen CIFAR-100 pods.

(b) What a parameter buys. Accuracy is regressed on the gene indicators *and* log2 parameters,
    which separates "this gene helps" from "this gene is simply bigger".

(c) Boundary audit. For every gene, where in its range does the best value sit? A grid whose
    optimum is on its own edge is a space that caps the attainable accuracy, and no search
    algorithm can fix it. This is a free pre-flight check for any study of this kind.

Writes results/space.json, tables/tab_cost.tex, tables/tab_boundary.tex, data/cost_model.csv,
the numbers analysis/s57_figures.py draws figures/fig_pareto.pdf and
figures/fig_space.pdf from.
"""
import csv, json, re
import numpy as np
import common, surrogate as sg
from surrogate import SS, GENES, CARD, NG
from scipy.stats import spearmanr

IMG = 32
THRU = re.compile(r'Throughput:\s*([0-9.]+)')
ARCH = re.compile(r'Evaluating unique arch \(checksum:\s*([0-9a-f]+)')


def cost_of(codes):
    """(parameters in millions, MMACs) of the network a genotype describes.

    Convolutions only, which dominate both counts: a C-column fractal block holds 2^C - 1
    convolutions; the first maps the incoming channels, the rest are channel-preserving.
    Depthwise convolutions are counted as a depthwise kernel plus a pointwise mixer.
    """
    ch = {g: SS[g][int(v)] for g, v in zip(GENES, codes)}
    C, K, nb, nc = ch['base_channels'], ch['kernel_size'], ch['n_blocks'], ch['n_columns']
    dw = ch['conv_type'] == 'Depthwise'
    p = mac = 0.0
    cin, res = 3, IMG
    for b in range(nb):
        cout = C * (2 ** b)
        n_conv = 2 ** nc - 1

        def one(ci, co):
            return (ci * K * K + ci * co) if dw else (ci * co * K * K)
        p_blk = one(cin, cout) + (n_conv - 1) * one(cout, cout)
        p += p_blk
        mac += p_blk * res * res
        cin, res = cout, max(1, res // 2)
    return p / 1e6, mac / 1e6


def measured_throughput():
    """uid -> mean samples/s, from the pods that printed it (frozen CIFAR-100 only)."""
    out = {}
    for r in common.raw_runs('frozen', 'cifar100'):
        uids = {x['uid'] for x in r['e1']}
        pre = {}
        for u in uids:
            pre.setdefault(u[:8], []).append(u)
        cur, buf = None, []

        def close():
            c = pre.get(cur, [])
            if cur and len(c) == 1 and buf:
                out.setdefault(c[0], float(np.mean(buf)))
        for line in common._pod_text(r['path']).split('\n'):
            a = ARCH.search(line)
            if a:
                close(); cur, buf = a.group(1), []
                continue
            t = THRU.search(line)
            if t and cur:
                buf.append(float(t.group(1)))
        close()
    return out


def main():
    out = {}
    thr = measured_throughput()
    rows_out = []
    for ds in common.DATASETS:
        rows = [r for r in common.gene_table()
                if r['dataset'] == ds and r['acc3'] is not None]
        P = np.array([sg.codes_of(r) for r in rows])
        y = np.array([r['acc3'] for r in rows])
        cm = np.array([cost_of(p) for p in P])
        par, mac = cm[:, 0], cm[:, 1]
        for r, p_, m_ in zip(rows, par, mac):
            rows_out.append((r['uid'], ds, r['method'], f'{p_:.4f}', f'{m_:.1f}',
                             f'{r["acc3"]:.2f}'))
        e = {'n': len(y),
             'rho_acc_params': float(spearmanr(par, y).statistic),
             'rho_acc_macs': float(spearmanr(mac, y).statistic),
             'median_params': float(np.median(par)),
             'median_params_top50': float(np.median(par[np.argsort(-y)[:50]]))}

        # --- what a parameter buys, holding the genes fixed
        A = np.hstack([sg.design(P).astype(np.float64), np.log2(par)[:, None]])
        G = A.T @ A
        G[np.diag_indices_from(G)] += 10.0
        w = np.linalg.solve(G, A.T @ (y - y.mean()))
        e['points_per_doubling'] = float(w[-1])

        # --- per-gene: accuracy spread, and the parameter factor it costs
        g = {}
        for gi, gn in enumerate(GENES):
            vals = {}
            for v in range(CARD[gi]):
                m = P[:, gi] == v
                if m.sum() < 30:
                    continue
                vals[str(SS[gn][v])] = {'acc': float(y[m].mean()),
                                        'params': float(np.median(par[m])),
                                        'n': int(m.sum())}
            if len(vals) < 2:
                continue
            best = max(vals, key=lambda k: vals[k]['acc'])
            worst = min(vals, key=lambda k: vals[k]['acc'])
            order = [str(x) for x in SS[gn]]
            pos = order.index(best)
            g[gn] = {'values': vals, 'best': best, 'worst': worst,
                     'spread': vals[best]['acc'] - vals[worst]['acc'],
                     'param_ratio': vals[best]['params'] / vals[worst]['params'],
                     'edge': ('low' if pos == 0 else 'high' if pos == len(order) - 1
                              else 'interior') if len(order) > 2 else 'binary',
                     'n_values': len(order)}
        e['genes'] = g
        e['n_edge'] = sum(1 for v in g.values() if v['edge'] in ('low', 'high'))
        e['n_ordinal'] = sum(1 for v in g.values() if v['edge'] != 'binary')

        # --- Pareto front over (params, accuracy)
        o = np.argsort(par)
        front, bestacc = [], -np.inf
        for i in o:
            if y[i] > bestacc:
                bestacc = y[i]
                front.append({'params': float(par[i]), 'acc': float(y[i]),
                              'macs': float(mac[i])})
        e['pareto'] = front
        cheap = par <= np.quantile(par, 0.25)
        e['best_overall'] = float(y.max())
        e['best_cheapest_quartile'] = float(y[cheap].max())
        e['cheapest_quartile_params'] = float(np.quantile(par, 0.25))
        out[ds] = e

    # --- validation of the cost model against measured throughput
    u = [k for k in thr if k]
    gt = {r['uid']: sg.codes_of(r) for r in common.gene_table() if r['dataset'] == 'cifar100'}
    pair = [(cost_of(gt[k])[1], thr[k]) for k in u if k in gt]
    if pair:
        m_, t_ = np.array([p[0] for p in pair]), np.array([p[1] for p in pair])
        out['cost_model_validation'] = {
            'n': len(pair),
            'rho_macs_vs_throughput': float(spearmanr(m_, t_).statistic),
            'rho_logmacs_vs_logthroughput': float(spearmanr(np.log(m_), np.log(t_)).statistic)}
    json.dump(out, open(f'{common.RES}/space.json', 'w'), indent=1)

    with open(f'{common.DATA}/cost_model.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['uid', 'dataset', 'method', 'params_M', 'mmacs', 'acc_epoch3_true'])
        wr.writerows(rows_out)

    # ---- table: one audit of the space --------------------------------------------
    SHOW = ['n_columns', 'base_channels', 'n_blocks', 'kernel_size', 'conv_type']
    L = [r'\begin{table}[t]', r'\centering',
         r'\caption{Audit of the search space. Top: what each structural gene buys and what it',
         r'costs, as the mean 3-epoch accuracy and the median parameter count over every',
         r'architecture carrying that value. Fractal width is the one gene that costs parameters',
         r"and loses accuracy. Bottom: where in its declared range each non-binary gene's best",
         r'value sits; an optimum on the edge means the grid, not the optimiser, is what caps',
         r'the attainable accuracy.}',
         r'\label{tab:space}', r'\footnotesize', r'\resizebox{\columnwidth}{!}{%', r'\begin{tabular}{llcccc}', r'\toprule',
         r'& & \multicolumn{2}{c}{CIFAR-10} & \multicolumn{2}{c}{CIFAR-100} \\',
         r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
         r'Gene & Value & Acc. & Par.\,(M) & Acc. & Par.\,(M) \\', r'\midrule']
    for gn in SHOW:
        vals = out['cifar10']['genes'][gn]['values']
        for k, v in enumerate(vals):
            a = vals[v]
            b = out['cifar100']['genes'][gn]['values'].get(
                v, {'acc': float('nan'), 'params': float('nan')})
            nm = gn.replace('_', r'\_') if k == 0 else ''
            L.append(rf'{nm} & {v} & {a["acc"]:.1f} & {a["params"]:.3f} & '
                     rf'{b["acc"]:.1f} & {b["params"]:.3f} \\')
        L.append(r'\addlinespace[1.5pt]')
    L += [r'\midrule',
          rf'\multicolumn{{2}}{{l}}{{Points per doubling of parameters}} & '
          rf'\multicolumn{{2}}{{c}}{{{out["cifar10"]["points_per_doubling"]:+.2f}}} & '
          rf'\multicolumn{{2}}{{c}}{{{out["cifar100"]["points_per_doubling"]:+.2f}}} \\',
          r'\midrule',
          r"\multicolumn{6}{l}{\emph{Boundary audit: position of the best value in the gene's range}} \\"]
    MARK = {'low': r'\textbf{edge (low)}', 'high': r'\textbf{edge (high)}',
            'interior': 'interior', 'binary': '---'}
    for gn in GENES:
        a = out['cifar10']['genes'].get(gn)
        b = out['cifar100']['genes'].get(gn)
        if not a or not b or a['edge'] == 'binary':
            continue
        L.append(rf'\multicolumn{{2}}{{l}}{{\quad {gn.replace("_", chr(92)+"_")}}} & '
                 rf'\multicolumn{{2}}{{c}}{{{a["best"]}, {MARK[a["edge"]]}}} & '
                 rf'\multicolumn{{2}}{{c}}{{{b["best"]}, {MARK[b["edge"]]}}} \\')
    L += [rf'\multicolumn{{2}}{{l}}{{\quad \emph{{On an edge}}}} & '
          rf'\multicolumn{{2}}{{c}}{{\textbf{{{out["cifar10"]["n_edge"]} of '
          rf'{out["cifar10"]["n_ordinal"]}}}}} & '
          rf'\multicolumn{{2}}{{c}}{{\textbf{{{out["cifar100"]["n_edge"]} of '
          rf'{out["cifar100"]["n_ordinal"]}}}}} \\',
          r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    open(f'{common.HERE}/../tables/tab_space.tex', 'w').write('\n'.join(L) + '\n')

    v = out.get('cost_model_validation', {})
    print(f'cost model validation: n={v.get("n")} rho(MMACs, throughput)='
          f'{v.get("rho_macs_vs_throughput", float("nan")):+.3f}')
    for ds in common.DATASETS:
        e = out[ds]
        print(f'== {ds} ==')
        print(f'  rho(acc, params)={e["rho_acc_params"]:+.3f}  '
              f'median params all={e["median_params"]:.3f}M top50={e["median_params_top50"]:.3f}M')
        print(f'  points per doubling of parameters = {e["points_per_doubling"]:+.2f}')
        print(f'  best overall {e["best_overall"]:.2f} | best in cheapest quartile '
              f'(<= {e["cheapest_quartile_params"]:.3f}M) {e["best_cheapest_quartile"]:.2f}')
        print(f'  Pareto front: {len(e["pareto"])} points')
        print(f'  ordinal genes with the optimum on an edge: {e["n_edge"]}/{e["n_ordinal"]}')
        nc = e['genes']['n_columns']
        print(f'  n_columns best={nc["best"]} worst={nc["worst"]} spread={nc["spread"]:.2f} '
              f'param ratio best/worst={nc["param_ratio"]:.2f}x')


if __name__ == '__main__':
    main()
