"""The changes this paper recommends, collected into one table a practitioner can act on.

Every row is a difference already measured by another script in this archive; nothing is
recomputed here.  The point of the table is that the recommendations are scattered across
three sections -- the GA's settings in Section V, the acceptance gate in Section VI, the
grid's own boundary in Supplementary Section S6 -- and a reader deciding what to change
tomorrow should not have to reassemble them.

Each row carries what the audited study used, what the benchmark recommends instead, the
difference on each dataset, and the status the paper gives it: a claimed gain, a statement
about the landscape rather than about a real pipeline, or a forecast past the edge of the
grid.  Rows the paper declines to claim -- the elite count -- are not listed.

    python3 analysis/s68_adopt.py        # writes ../tables/tab_adopt.tex

Reads results/levers.json, results/gate.json and results/extension.json.
"""
import json
import os

import common

A, B = 'cifar10', 'cifar100'


def J(n):
    return json.load(open(os.path.join(common.RES, n + '.json')))


def main():
    lv, gt, ex = J('levers'), J('gate'), J('extension')
    s = {d: lv[d]['settings'] for d in (A, B)}

    def lever(pre, best, used):
        return {d: s[d][f'{pre}={best}']['best'] - s[d][f'{pre}={used}']['best'] for d in (A, B)}

    def signal():
        return {d: lv[d]['cells']['long_measured']['best']
                - lv[d]['cells']['long_predictor']['best'] for d in (A, B)}

    def best(pre, values):
        return max(values, key=lambda v: s[A][f'{pre}={v}']['best'] + s[B][f'{pre}={v}']['best'])

    kb = best('k', (2, 3, 5, 7, 10))
    pb = best('pm', (0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4))
    rows = [
        ('tournament size $k$', '3', str(kb), lever('k', kb, 3), 'claimed',
         r'Table~\ref{tab:settings}'),
        ('mutation rate $p_m$ (effective)', '0.08', str(pb), lever('pm', pb, 0.08), 'claimed',
         r'Table~\ref{tab:settings}'),
        ('population size', '20', '10', lever('pop', 10, 20), 'claimed',
         r'Table~\ref{tab:settings}'),
        ('acceptance gate', 'smoke call', 'property test $+$ replay',
         {d: gt[d]['gain_total'] for d in (A, B)}, 'claimed', r'Table~\ref{tab:gate}'),
        ('selection signal', 'LLM predictor', 'measured accuracy', signal(),
         'landscape', r'Fig.~\ref{fig:settings}(f)'),
        ('dropout range', '$\\geq 0.05$', 'extended below $0.05$',
         {d: ex[d]['forecast']['dropout_prob']['pred_gain'] for d in (A, B)},
         'forecast', r'Suppl.\ Section~S6'),
    ]

    status = {'claimed': '', 'landscape': r'$\dagger$', 'forecast': r'$\ddagger$'}

    out = [r'\begin{table}[t]', r'\centering',
           r'\caption{What to change, collected. Every entry is a difference between '
           r'configurations on the same landscape, in accuracy points, taken from the table '
           r'or figure named; none is a predicted accuracy. $\dagger$ a property of the '
           r'landscape rather than a claim about a real pipeline (Section~\ref{sec:limitations}); '
           r'$\ddagger$ a forecast one level past the edge of the declared grid, validated '
           r'leave-one-level-out (Supplementary Section~S6). The elite count is omitted: it is '
           r'the one setting this paper does not claim. Produced by '
           r'\texttt{analysis/s68\_adopt.py}.}',
           r'\label{tab:adopt}', r'\footnotesize',
           r'\resizebox{\columnwidth}{!}{%',
           r'\begin{tabular}{l l l rr l}', r'\toprule',
           r'& & & \multicolumn{2}{c}{$\Delta$ (pts)} & \\',
           r'\cmidrule(lr){4-5}',
           r'Decision & As run & Recommended & C-10 & C-100 & Evidence \\',
           r'\midrule']
    for name, used, rec, d, st, where in rows:
        out.append('%s%s & %s & %s & $%+.2f$ & $%+.2f$ & %s \\\\'
                   % (name, status[st], used, rec, d[A], d[B], where))
    out += [r'\bottomrule', r'\end{tabular}}', r'\end{table}', '']

    p = os.path.join(common.HERE, '..', 'tables', 'tab_adopt.tex')
    with open(p, 'w') as fh:
        fh.write('\n'.join(out))
    print('wrote tables/tab_adopt.tex')


if __name__ == '__main__':
    main()
