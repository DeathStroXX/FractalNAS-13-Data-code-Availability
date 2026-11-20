"""The acceptance gate: what LLM-written operators are worth once they are tested.

The pipeline accepted a rewrite if it parsed and survived one smoke call on a single
chromosome. s2_operator_audit.py shows what that lets through -- 24 of the 90 accepted rewrites
leave the search space or fail to terminate -- and s20_operator_bench.py shows that the
survivors are not interchangeable: every mutation rewrite beats the handcrafted operator while
crossover rewrites lose. Both facts are about the *gate*, not about the model. A generator whose
proposals are accepted on syntax alone is being asked to be right first time, which no
generator is.

This script replaces that gate and prices the pipeline under each one, over the identical 90
proposals the three 7B models actually produced. Three gates:

  none       -- the pipeline's own rule: parse, one smoke call, splice it in.
  property   -- hundreds of randomised calls against the space, grid membership and chromosome
                completeness checked on every output, a per-call time-out. This is
                s2_operator_audit.py's harness and it costs seconds per rewrite.
  property+  -- the same, and then a replay of the candidate against the operator it would
                replace on FractalNAS-13, accepting only a non-negative mean. The benchmark
                makes that replay cost milliseconds.

A *deployment* is one meta-run: five segments, one proposal per segment in the pipeline's
round-robin order, with the gate deciding whether each is spliced in or rolled back. Every gate
sees the same proposals in the same order, so the comparison isolates the gate.

Two modelling choices keep this honest.

  * The effect of each proposal is the measured one from s20_operator_bench.py (150 replicates
    of the pipeline's own 5x5 schedule, with its rollback behaviour), so nothing here re-runs a
    GA and nothing is a new estimate.
  * The `property+` gate must not be an oracle. Its accept/reject decision is simulated as the
    decision a *cheap* replay would make: an independent estimate drawn from the measured effect
    and its standard error, inflated to the R_GATE replicates the gate is allowed, averaged over
    DRAWS Monte-Carlo repetitions. A gate given 60 replicates therefore makes mistakes, at the
    rate 60 replicates imply.

It also emits the inventory of the 90 proposals -- accepted, in-space, replayed, and the
measured effect of each class -- so that every count the section quotes reconciles in one
place rather than across four paragraphs.

Writes results/gate.json, ../tables/tab_gate.tex, ../tables/tab_inventory.tex,
../figures/fig_gate.pdf.

    python3 s38_gate.py              # price the gates and write both tables and the figure
    python3 s38_gate.py --tex-only   # rewrite both tables from the stored results
"""
import collections, json, os, sys
import numpy as np

import common

R_GATE = 60          # replicates the property+ gate is allowed for its accept/reject call
DRAWS = 2000         # Monte-Carlo repetitions of the gate's own noisy decision
GATES = ['none', 'property', 'property+']
#: the rewrites that never leave the search space, and so can be replayed faithfully
INSPACE = {'modified', 'equivalent'}
OPS = ['mutate_gene', '_create_random_chromosome', 'select_competitor', 'combine_genes']


def load(ds):
    """Join the audit's verdict to the replay's measured effect, grouped by deployment."""
    audit = [a for a in json.load(open(os.path.join(common.RES, 'audit.json')))
             if a['ds'] == ds]
    bench = {r['file']: r for r in
             json.load(open(os.path.join(common.RES, 'operator_bench.json')))[ds]['rows']}
    runs = collections.defaultdict(list)
    for a in audit:
        f = os.path.basename(a['file'])
        b = bench.get(f)
        runs[(a['model'], a['run'])].append({
            'file': f, 'op': a['op'], 'cat': a['cat'], 'it': a['it'],
            'inspace': a['cat'] in INSPACE,
            # A rewrite that leaves the space is reverted at run time, so the segment it was
            # active in runs on the handcrafted operator: no gain, and no loss either. The same
            # holds for one the replay could not measure -- `bench` carries a row without a
            # `delta` when the harness abandoned the operator on its CPU cap -- since a gate
            # cannot act on a number nobody has.
            'delta': (b.get('delta', 0.0) if b else 0.0) if a['cat'] in INSPACE else 0.0,
            'se': (b.get('se', 0.0) if b else 0.0) if a['cat'] in INSPACE else 0.0,
            'replayed': bool(b) and 'delta' in b})
    for k in runs:
        runs[k].sort(key=lambda r: r['it'])
    return dict(runs)


def main():
    rng = np.random.default_rng(0)
    out = {'r_gate': R_GATE, 'draws': DRAWS, 'gates': GATES, 'r_bench': 150}

    for ds in common.DATASETS:
        runs = load(ds)
        props = [p for v in runs.values() for p in v]
        n_prop = len(props)
        n_inspace = sum(p['inspace'] for p in props)

        # the gate's own cheap estimate is noisier than the reported effect by sqrt(R/R_gate)
        infl = np.sqrt(out['r_bench'] / R_GATE)

        res = {'n_proposals': n_prop, 'n_inspace': n_inspace,
               'n_deployments': len(runs), 'gates': {}}
        for gate in GATES:
            per_draw = []
            acc, rej = [], []
            for _ in range(DRAWS):
                tot, a_n, r_n = [], 0, 0
                for key, ps in runs.items():
                    s = 0.0
                    for p in ps:
                        if gate == 'none':
                            ok = True
                        elif gate == 'property':
                            ok = p['inspace']
                        else:
                            est = p['delta'] + rng.normal(0, max(p['se'], 1e-9) * infl)
                            ok = p['inspace'] and est >= 0
                        if ok:
                            s += p['delta']
                            a_n += 1
                        else:
                            r_n += 1
                    tot.append(s / len(ps))
                per_draw.append(float(np.mean(tot)))
                acc.append(a_n)
                rej.append(r_n)
            res['gates'][gate] = {
                'delta_mean': float(np.mean(per_draw)),
                'delta_se': float(np.std(per_draw)),
                'accepted': float(np.mean(acc)),
                'rejected': float(np.mean(rej)),
                'accept_rate': float(np.mean(acc) / (np.mean(acc) + np.mean(rej))),
                'deployable_share': (1.0 if gate != 'none' else n_inspace / n_prop)}

        by_op = collections.defaultdict(list)
        for p in props:
            if p['replayed']:
                by_op[p['op']].append(p['delta'])
        res['by_op'] = {k: {'n': len(v), 'mean': float(np.mean(v)),
                            'se': float(np.std(v) / np.sqrt(len(v))),
                            'share_pos': float(np.mean(np.array(v) > 0))}
                        for k, v in by_op.items()}
        res['gain_property'] = (res['gates']['property']['delta_mean']
                                - res['gates']['none']['delta_mean'])
        res['gain_replay'] = (res['gates']['property+']['delta_mean']
                              - res['gates']['property']['delta_mean'])
        res['gain_total'] = (res['gates']['property+']['delta_mean']
                             - res['gates']['none']['delta_mean'])
        out[ds] = res

    # the audit's own counts, quoted in the section
    audit = json.load(open(os.path.join(common.RES, 'audit.json')))
    cat = collections.Counter(a['cat'] for a in audit)
    out['audit'] = {
        'n': len(audit),
        'inspace': sum(cat[c] for c in INSPACE),
        'inspace_share': sum(cat[c] for c in INSPACE) / len(audit),
        'by_cat': dict(cat),
        'by_op_fail': {op: sum(1 for a in audit
                               if a['op'] == op and a['cat'] not in INSPACE)
                       for op in OPS},
        'by_op_n': {op: sum(1 for a in audit if a['op'] == op) for op in OPS}}

    os.makedirs(common.RES, exist_ok=True)
    json.dump(out, open(os.path.join(common.RES, 'gate.json'), 'w'), indent=1)
    emit_table(out)
    emit_inventory(out)
    emit_figure(out)
    for ds in common.DATASETS:
        r = out[ds]
        print('%s: %d proposals, %d in-space, %d deployments'
              % (ds, r['n_proposals'], r['n_inspace'], r['n_deployments']))
        for g in GATES:
            v = r['gates'][g]
            print('   %-10s accept %4.1f/%d  deployable %3.0f%%  delta %+.3f +- %.3f'
                  % (g, v['accepted'], r['n_proposals'] * 1,
                     100 * v['deployable_share'], v['delta_mean'], v['delta_se']))
        print('   gain: property %+.3f, replay %+.3f, total %+.3f'
              % (r['gain_property'], r['gain_replay'], r['gain_total']))


def emit_table(o):
    A, B = 'cifar10', 'cifar100'
    LAB = {'none': r'Parse $+$ one smoke call (as run)',
           'property': r'$+$ property-based testing',
           'property+': r'$+$ benchmark replay'}
    L = []
    a = L.append
    a(r'\begin{table}[t]')
    a(r'\centering')
    a(r'\caption{The acceptance gate and what an operator-writing LLM is worth. The %d '
      r'rewrites the three 4-bit 7B-class language models produced and the pipeline accepted are passed through three gates; '
      r'\emph{deployable} is the share of accepted rewrites that stay inside the search space '
      r'and terminate, and $\Delta$ is the change in the best architecture found per proposal, '
      r'averaged over the accepted proposals of each deployment, against the handcrafted operators on the identical '
      r'schedule. Effects are the replays described in the text (150 replicates each); the '
      r'replay gate gets only %d replicates for its own decision, so it makes the mistakes '
      r'that budget implies ($\pm$ s.d.\ over %d Monte-Carlo repetitions). Produced by '
      r'\texttt{analysis/s38\_gate.py}.}'
      % (o[A]['n_proposals'] + o[B]['n_proposals'], o['r_gate'], o['draws']))
    a(r'\label{tab:gate}')
    a(r'\footnotesize')
    a(r'\resizebox{\columnwidth}{!}{%')
    a(r'\begin{tabular}{lcccc}')
    a(r'\toprule')
    a(r'& \multicolumn{2}{c}{CIFAR-10} & \multicolumn{2}{c}{CIFAR-100} \\')
    a(r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}')
    a(r'Gate & deployable & $\Delta$ & deployable & $\Delta$ \\')
    a(r'\midrule')
    for g in GATES:
        va, vb = o[A]['gates'][g], o[B]['gates'][g]
        a('%s & %.0f\\%% & $%+.2f\\!\\pm\\!%.2f$ & %.0f\\%% & $%+.2f\\!\\pm\\!%.2f$ \\\\'
          % (LAB[g], 100 * va['deployable_share'], va['delta_mean'], va['delta_se'],
             100 * vb['deployable_share'], vb['delta_mean'], vb['delta_se']))
    a(r'\bottomrule')
    a(r'\end{tabular}}')
    a(r'\end{table}')
    p = os.path.join(common.HERE, '..', 'tables', 'tab_gate.tex')
    open(p, 'w').write('\n'.join(L) + '\n')
    print('wrote', p)


OP_LABEL = {'mutate_gene': 'Mutation', '_create_random_chromosome': 'Initialization',
            'combine_genes': 'Crossover', 'select_competitor': 'Selection'}


def _unreplayed(n):
    if n <= 0:
        return ''
    return ('; the %d in-space rewrite%s not replayed %s abandoned by the harness on its CPU cap'
            % (n, '' if n == 1 else 's', 'was' if n == 1 else 'were'))


def emit_inventory(o):
    """Every count the operator section quotes, reconciled in one table."""
    A, B = 'cifar10', 'cifar100'
    ob = json.load(open(os.path.join(common.RES, 'operator_bench.json')))
    aud = o['audit']
    L = []
    a = L.append
    a(r'\begin{table}[t]')
    a(r'\centering')
    a(r'\caption{The %d accepted rewrites, reconciled. \emph{Accepted} is what the '
      r"pipeline's own gate admitted; \emph{in space} is what never leaves the grid and so "
      r'can be replayed faithfully (%d of the %d do not: %d raise or fail to terminate, %d '
      r'leave the space, %d land off the grid); \emph{replayed} is what '
      r'\texttt{s20\_operator\_bench.py} measured%s. $\Delta$ is the class mean of '
      r'the per-rewrite effect against the operator it replaced in the pipeline, as '
      r'Section~\ref{sec:terms} defines it, with the standard error over that class. '
      r'Produced by \texttt{analysis/s38\_gate.py}.}'
      % (aud['n'], aud['n'] - aud['inspace'], aud['n'],
         aud['by_cat'].get('raises/hangs', 0), aud['by_cat'].get('out-of-space', 0),
         aud['by_cat'].get('off-grid', 0),
         _unreplayed(aud['inspace'] - (ob[A]['n'] + ob[B]['n']))))
    a(r'\label{tab:inventory}')
    a(r'\footnotesize')
    a(r'\resizebox{\columnwidth}{!}{%')
    a(r'\begin{tabular}{l rrr rr}')
    a(r'\toprule')
    a(r'& & & & \multicolumn{2}{c}{$\Delta$ best found (pts)} \\')
    a(r'\cmidrule(lr){5-6}')
    a(r'Operator replaced & accepted & in space & replayed & CIFAR-10 & CIFAR-100 \\')
    a(r'\midrule')
    tot = [0, 0, 0]
    for op in OPS:
        n_acc = aud['by_op_n'][op]
        n_in = n_acc - aud['by_op_fail'].get(op, 0)
        n_rep = ob[A]['by_kind'].get(op, {}).get('n', 0) + ob[B]['by_kind'].get(op, {}).get('n', 0)
        tot = [tot[0] + n_acc, tot[1] + n_in, tot[2] + n_rep]
        cells = []
        for ds in (A, B):
            k = ob[ds]['by_kind'].get(op)
            cells.append('--' if not k else r'$%+.2f\!\pm\!%.2f$' % (k['mean'], k['se']))
        a(r'%s & %d & %d & %d & %s & %s \\'
          % (OP_LABEL[op], n_acc, n_in, n_rep, cells[0], cells[1]))
    a(r'\midrule')
    a(r'\textit{Total} & \textit{%d} & \textit{%d} & \textit{%d} & & \\' % tuple(tot))
    a(r'\bottomrule')
    a(r'\end{tabular}}')
    a(r'\end{table}')
    p = os.path.join(common.HERE, '..', 'tables', 'tab_inventory.tex')
    open(p, 'w').write('\n'.join(L) + '\n')
    print('wrote', p)


def emit_figure(o):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    COL = {'cifar10': '#1b6ca8', 'cifar100': '#c7522a'}
    NAME = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100'}
    LAB = {'none': 'parse + smoke\n(as run)', 'property': '+ property\ntesting',
           'property+': '+ benchmark\nreplay'}
    x = np.arange(len(GATES))
    for i, ds in enumerate(common.DATASETS):
        v = [o[ds]['gates'][g]['delta_mean'] for g in GATES]
        e = [o[ds]['gates'][g]['delta_se'] for g in GATES]
        ax.bar(x + (i - .5) * .36, v, .34, yerr=e, color=COL[ds], label=NAME[ds],
               capsize=2, error_kw={'lw': .9})
    ax.axhline(0, color='k', lw=.8)
    ax.set_xticks(x, [LAB[g] for g in GATES], fontsize=7)
    ax.set_ylabel(r'$\Delta$ best found, per proposal (pts)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=.25, lw=.5, axis='y')
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(common.HERE, '..', 'figures', 'fig_gate.' + ext),
                    dpi=220, bbox_inches='tight')
    print('wrote figures/fig_gate.pdf')


if __name__ == '__main__':
    if '--tex-only' in sys.argv:
        _o = json.load(open(os.path.join(common.RES, 'gate.json')))
        emit_table(_o)
        emit_inventory(_o)
    else:
        main()
