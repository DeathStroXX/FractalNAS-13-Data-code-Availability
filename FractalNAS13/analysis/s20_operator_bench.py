"""What the 90 LLM-written operators are actually worth, at 150 replicates each.

The original study could only compare operators between whole search arms (9 runs against 5),
where the smallest attainable p-value is 0.0045 and no operator effect could reach it. On
FractalNAS-13 each operator can be dropped into the GA and run 150 times in seconds, which
turns "no measurable effect" into a per-operator effect size.

Each accepted rewrite replaces its handcrafted counterpart in an otherwise identical GA
(5 segments of 5 generations, population 20, 5 elites: the pipeline's own schedule and GA); a
call that raises or times out falls back to the handcrafted behaviour, as the pipeline's rollback
does. A mutation rewrite is compared with the pipeline's own mutate_gene
(pipeline_mutate_gene.py) run through the same code path and random stream, so a rewrite that
reproduces it scores exactly zero.

Writes results/operator_bench.json, tables/tab_opbench.tex.
"""
import json, os, glob, collections
import numpy as np
import common
import fractalbench as fb
import signal, contextlib

R = 150
CAP = 300          # CPU-seconds allowed per operator before the run is abandoned
#: The replay runs with NO per-call time budget. Only operators the audit has already
#: classified as in-space are replayed, and that classification comes from 1,500 randomised
#: calls per operator under a per-call time-out, so every operator reaching this script is
#: known to terminate. A per-call budget here would add nothing and would cost determinism:
#: one of the selection rewrites draws from the population until it has collected three
#: distinct individuals, a loop whose tail is long enough that on 1.3 million calls a few
#: exceed any fixed budget -- and which of them do depends on how loaded the machine is. The
#: whole-operator CPU cap below still abandons anything pathological, and says so.
TIMER_GUARD = False
HANG = {'raises/hangs'}


class _Cap(Exception):
    pass


fb.FATAL_EXC = ()        # set in main() once _Cap exists


@contextlib.contextmanager
def _cap(seconds):
    """CPU-time cap: a rewrite that loops forever would otherwise cost one time-out per call."""
    def boom(sig, frm):
        raise _Cap()
    old = signal.signal(signal.SIGVTALRM, boom)
    signal.setitimer(signal.ITIMER_VIRTUAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_VIRTUAL, 0)
        signal.signal(signal.SIGVTALRM, old)
# Only the rewrites that never leave the search space can be replayed faithfully: a benchmark
# genotype is a point of the grid, so an operator that emits off-grid or out-of-space values
# would have to be snapped back onto it, which is not what it does in the pipeline. Those 24
# are the ones Section VI's property-based testing rejects before deployment.
INSPACE = {'modified', 'equivalent'}


def main():
    fb.FATAL_EXC = (_Cap,)
    audit = {os.path.basename(a['file']): a
             for a in json.load(open(os.path.join(common.RES, 'audit.json')))}
    files = sorted(glob.glob(os.path.join(common.HERE, 'operators', '*.py')))
    out = {}
    for ds in common.DATASETS:
        b = fb.Bench(ds)
        base = fb.run_ga(b, 5, 5, R=R, seed=1)
        base_reps = [float(x) for x in base['best']]
        bbest = float(base['best'].mean())
        bse = float(base['best'].std() / np.sqrt(R))
        _, own_mutate = fb.load_operator(os.path.join(common.HERE, 'pipeline_mutate_gene.py'))
        base_m = fb.run_ga(b, 5, 5, R=R, seed=1, ops={'mutate_gene': own_mutate},
                           timer_guard=TIMER_GUARD)
        ref = {'mutate_gene': (float(base_m['best'].mean()),
                               float(base_m['best'].std() / np.sqrt(R)))}
        rows = []
        for f in files:
            name = os.path.basename(f)
            a = audit.get(name, {})
            # an operator is benchmarked on the dataset whose run produced it
            if a.get('ds') and a['ds'] != ds:
                continue
            if a.get('cat') not in INSPACE:
                continue
            try:
                kind, fn = fb.load_operator(f)
            except Exception as e:
                rows.append(dict(file=name, error=str(e)[:60])); continue
            try:
                with _cap(CAP):
                    r = fb.run_ga(b, 5, 5, R=R, seed=1, ops={kind: fn},
                                  timer_guard=TIMER_GUARD)
            except _Cap:
                rows.append(dict(file=name, kind=kind, cat=a.get('cat'),
                                 error='non-terminating under the harness'))
                print(f"  {ds} {name:52s}  EXCLUDED (CPU cap)", flush=True)
                continue
            except Exception as e:
                rows.append(dict(file=name, error=type(e).__name__))
                print(f"  {ds} {name:52s}  ERROR {type(e).__name__}", flush=True)
                continue
            m = float(r['best'].mean()); se = float(r['best'].std() / np.sqrt(R))
            rb, rse = ref.get(kind, (bbest, bse))
            rows.append(dict(file=name, kind=kind, model=a.get('model'), run=a.get('run'),
                             it=a.get('it'), cat=a.get('cat'), best=m, se=se,
                             delta=m - rb, z=(m - rb) / max(1e-9, (se**2 + rse**2)**.5),
                             fail_rate=r['op_fail_rate'], calls=r['op_calls'],
                             mean_sampled=float(r['mean_sampled'].mean()),
                             top10=float(r['top10'].mean()),
                             reps=[float(x) for x in r['best']]))
            print(f"  {ds} {name:52s} {rows[-1]['delta']:+6.2f}  z={rows[-1]['z']:+6.1f} "
                  f"[{a.get('cat')}]", flush=True)
        ok = [r for r in rows if 'delta' in r]
        print(f"  -> {ds}: {len(ok)} benchmarked, "
              f"{sum('error' in r for r in rows)} excluded", flush=True)
        by_cat, by_kind, by_model = (collections.defaultdict(list) for _ in range(3))
        for r in ok:
            by_cat[r['cat']].append(r['delta'])
            by_kind[r['kind']].append(r['delta'])
            by_model[r['model']].append(r['delta'])
        agg = lambda d: {k: dict(n=len(v), mean=float(np.mean(v)),
                                 se=float(np.std(v) / max(1, np.sqrt(len(v)))),
                                 best=float(np.max(v)), frac_pos=float(np.mean(np.array(v) > 0)))
                         for k, v in d.items()}
        out[ds] = dict(baseline_best=bbest, baseline_se=bse, baseline_reps=base_reps,
                       mutation_reference_best=ref['mutate_gene'][0],
                       mutation_reference_reps=[float(x) for x in base_m['best']],
                       n=len(ok), n_excluded=sum('delta' not in r for r in rows),
                       excluded=[r['file'] for r in rows if 'delta' not in r], rows=rows,
                       by_cat=agg(by_cat), by_kind=agg(by_kind), by_model=agg(by_model),
                       overall=dict(mean=float(np.mean([r['delta'] for r in ok])),
                                    se=float(np.std([r['delta'] for r in ok]) / np.sqrt(len(ok))),
                                    frac_pos=float(np.mean([r['delta'] > 0 for r in ok])),
                                    n_sig_pos=int(sum(r['z'] > 2 for r in ok)),
                                    n_sig_neg=int(sum(r['z'] < -2 for r in ok)),
                                    best_delta=float(max(r['delta'] for r in ok)),
                                    best_file=max(ok, key=lambda r: r['delta'])['file']))
    json.dump(out, open(os.path.join(common.RES, 'operator_bench.json'), 'w'), indent=1,
              default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    _tex(out)
    for ds, o in out.items():
        print(f"\n== {ds}  handcrafted baseline {o['baseline_best']:.2f}"
              f"+-{o['baseline_se']:.2f}; {o['n']} operators")
        ov = o['overall']
        print(f"   mean delta {ov['mean']:+.3f}+-{ov['se']:.3f}; {ov['frac_pos']*100:.0f}% positive; "
              f"{ov['n_sig_pos']} significantly better, {ov['n_sig_neg']} worse; "
              f"best {ov['best_delta']:+.2f} ({ov['best_file']})")
        for lab, d in (('class', o['by_cat']), ('operator', o['by_kind']), ('model', o['by_model'])):
            print(f"   by {lab}: " + ", ".join(
                f"{k}={v['mean']:+.2f}(n={v['n']})" for k, v in sorted(d.items(), key=lambda x: -x[1]['mean'])))


def _tex(out):
    nice = {'mutate_gene': 'Mutation', '_create_random_chromosome': 'Initialization',
            'select_competitor': 'Selection', 'combine_genes': 'Crossover'}
    L = ['\\begin{tabular}{l rrr rrr}', '\\toprule',
         '& \\multicolumn{3}{c}{CIFAR-10} & \\multicolumn{3}{c}{CIFAR-100} \\\\',
         '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}',
         '\\textbf{Operator rewritten} & $n$ & $\\Delta$ & \\textbf{\\% +} & $n$ & $\\Delta$ '
         '& \\textbf{\\% +} \\\\', '\\midrule']
    for k in ('mutate_gene', '_create_random_chromosome', 'select_competitor', 'combine_genes'):
        a = out['cifar10']['by_kind'].get(k); b = out['cifar100']['by_kind'].get(k)
        if not (a or b):
            continue
        def cell(d):
            if not d:
                return '-- & -- & --'
            return (f"{d['n']} & ${d['mean']:+.2f}\\!\\pm\\!{d['se']:.2f}$ "
                    f"& {d['frac_pos']*100:.0f}")
        L.append(f"{nice[k]} & {cell(a)} & {cell(b)} \\\\")
    L += ['\\midrule']
    for c, lab in (('modified', 'Sound modification'), ('equivalent', 'Behaviourally equivalent')):
        a = out['cifar10']['by_cat'].get(c); b = out['cifar100']['by_cat'].get(c)
        def cell(d):
            if not d:
                return '-- & -- & --'
            return (f"{d['n']} & ${d['mean']:+.2f}\\!\\pm\\!{d['se']:.2f}$ "
                    f"& {d['frac_pos']*100:.0f}")
        L.append(f"\\textit{{{lab}}} & {cell(a)} & {cell(b)} \\\\")
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(common.HERE, '..', 'tables', 'tab_opbench.tex'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
