"""The two levers on one scale, and the GA settings nobody could afford to sweep.

With FractalNAS-13 (fractalbench.py) a 520-evaluation search costs about ten milliseconds, so
the comparisons the original study ran at three replicates can be run at a thousand. Two
questions:

  1. Which lever is larger, the selection signal or the restart schedule?  The signal lever is
     the measured 3-epoch accuracy against an empirical emulator of the pipeline's own LLM
     predictor (same rank fidelity, same tie structure); the schedule lever is one long GA
     against five restarted segments at an identical evaluation budget.
  2. What should the remaining GA settings be?  Tournament size, mutation rate, population
     size and elitism, each swept at a thousand replicates.

Writes results/levers.json, tables/tab_levers.tex, tables/tab_settings.tex.
"""
import json, os
import numpy as np
import fractalbench as fb
import common

R = 1000
BUDGET = 520


def ci(v):
    return float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))


def reps(v):
    """The per-replicate outcomes themselves, which s62_significance.py tests as paired
    blocks: every configuration is run from the same seed, so replicate r of one setting and
    replicate r of another start from the identical population (common random numbers)."""
    return [float(x) for x in np.asarray(v, float)]


def main():
    out = {}
    for ds in common.DATASETS:
        b = fb.Bench(ds)
        emu = b.predictor_emulator()
        rho_emu = _rho_of(b, emu)
        o = {'n_fit': b.n_fit, 'grid': fb.GRID_SIZE, 'emulator_rho': rho_emu, 'cells': {}}

        def cell(name, **kw):
            r = fb.run_ga(b, R=R, seed=1, **kw)
            m, s = ci(r['best'])
            mm, ss = ci(r['mean_sampled'])
            g, gs = ci(r['gain_over_start'])
            o['cells'][name] = dict(best=m, best_se=s, mean=mm, mean_se=ss,
                                    gain_over_start=g, gain_se=gs, reps=reps(r['best']))
            return o['cells'][name]

        # --- the 2x2 of the two levers, equal budget
        for sched, (ns, gp) in [('long', (1, 26)), ('restart', (5, 5))]:
            for sig, fm in [('measured', None), ('predictor', emu)]:
                cell(f'{sched}_{sig}', n_seg=ns, gens=gp, fitness_map=fm, dup_penalty=True)
        c = o['cells']
        o['levers'] = {
            'signal_at_long': c['long_measured']['best'] - c['long_predictor']['best'],
            'signal_at_restart': c['restart_measured']['best'] - c['restart_predictor']['best'],
            'schedule_at_measured': c['restart_measured']['best'] - c['long_measured']['best'],
            'schedule_at_predictor': c['restart_predictor']['best'] - c['long_predictor']['best'],
            'signal_gain_cost_long': c['long_measured']['gain_over_start'] - c['long_predictor']['gain_over_start'],
            'schedule_gain_cost': c['restart_measured']['gain_over_start'] - c['long_measured']['gain_over_start'],
        }
        # --- segment length at equal budget, under both signals
        o['segments'] = {}
        for ns, gp in [(1, 26), (2, 13), (5, 5), (13, 2), (26, 1)]:
            for sig, fm in [('measured', None), ('predictor', emu)]:
                r = fb.run_ga(b, n_seg=ns, gens=gp, R=R, seed=1, fitness_map=fm,
                              dup_penalty=True)
                m, s = ci(r['best'])
                o['segments'][f'{ns}x{gp}_{sig}'] = dict(best=m, se=s,
                                                         reps=reps(r['best']))
        # --- the GA settings, swept under the measured signal on the 1x26 schedule
        o['settings'] = {}
        def setting(name, **kw):
            r = fb.run_ga(b, R=R, seed=1, **kw)
            m, s_ = ci(r['best'])
            o['settings'][name] = dict(best=m, se=s_, reps=reps(r['best']))

        for k in (2, 3, 5, 7, 10):
            setting(f'k={k}', n_seg=1, gens=26, k=k)
        # the mutation rate is swept as the effective per-gene rate; the pipeline's two-stage
        # operator (0.2 x 0.4) sits at 0.08
        for pm in (0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4):
            setting(f'pm={pm}', n_seg=1, gens=26, **fb.setting_kw('pm', pm))
        for pop, gens in [(10, 52), (20, 26), (40, 13), (52, 10)]:
            setting(f'pop={pop}', n_seg=1, gens=gens, pop=pop)
        for el in (0, 1, 2, 4, 5, 8):
            setting(f'elite={el}', n_seg=1, gens=26, elite=el)
        out[ds] = o
    json.dump(out, open(os.path.join(common.RES, 'levers.json'), 'w'), indent=1)
    _tex(out)
    for ds, o in out.items():
        print(f"\n== {ds}  (surrogate fitted on {o['n_fit']} architectures; "
              f"emulator rho={o['emulator_rho']:.3f}; {R} replicates, budget {BUDGET})")
        for n, c in o['cells'].items():
            print(f"   {n:20s} best {c['best']:6.2f}+-{c['best_se']:.2f}  "
                  f"mean {c['mean']:6.2f}  gain-over-start {c['gain_over_start']:+6.2f}")
        print("   levers:", {k: round(v, 2) for k, v in o['levers'].items()})
        print("   segments:", {k: round(v['best'], 2) for k, v in o['segments'].items()})
        print("   settings:", {k: round(v['best'], 2) for k, v in o['settings'].items()})


def _rho_of(b, fm, n=6000):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(5)
    P = rng.integers(0, fb.CARD, (n, fb.NG))
    t = b.score(P, tabular=False)
    return float(spearmanr(fm(t), t).statistic)


def _tex(out):
    a, b = out['cifar10'], out['cifar100']
    L = ['\\begin{tabular}{l rr rr}', '\\toprule',
         '& \\multicolumn{2}{c}{CIFAR-10} & \\multicolumn{2}{c}{CIFAR-100} \\\\',
         '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
         '\\textbf{Configuration} & \\textbf{Best} & \\textbf{Mean} & \\textbf{Best} '
         '& \\textbf{Mean} \\\\', '\\midrule']
    lab = {'long_measured': 'One long GA, measured accuracy as fitness',
           'long_predictor': 'One long GA, LLM-predictor fitness',
           'restart_measured': 'Five restarts, measured accuracy as fitness',
           'restart_predictor': 'Five restarts, LLM-predictor fitness'}
    for k, t in lab.items():
        L.append(f"{t} & {a['cells'][k]['best']:.2f} & {a['cells'][k]['mean']:.2f} & "
                 f"{b['cells'][k]['best']:.2f} & {b['cells'][k]['mean']:.2f} \\\\")
    L += ['\\midrule',
          '\\textit{Signal lever} (measured $-$ predictor, long GA) & '
          f"\\multicolumn{{2}}{{c}}{{${a['levers']['signal_at_long']:+.2f}$}} & "
          f"\\multicolumn{{2}}{{c}}{{${b['levers']['signal_at_long']:+.2f}$}} \\\\",
          '\\textit{Schedule lever} (restart $-$ long, measured) & '
          f"\\multicolumn{{2}}{{c}}{{${a['levers']['schedule_at_measured']:+.2f}$}} & "
          f"\\multicolumn{{2}}{{c}}{{${b['levers']['schedule_at_measured']:+.2f}$}} \\\\",
          '\\bottomrule', '\\end{tabular}']
    open(os.path.join(common.HERE, '..', 'tables', 'tab_levers.tex'), 'w').write('\n'.join(L) + '\n')

    spec = [('tournament size $k$', 'k=3', ['k=2', 'k=3', 'k=5', 'k=7', 'k=10'],
             {'k=2': '2', 'k=3': '3', 'k=5': '5', 'k=7': '7', 'k=10': '10'}),
            ('mutation rate $p_m$', 'pm=0.08',
             ['pm=0.02', 'pm=0.05', 'pm=0.08', 'pm=0.1', 'pm=0.2', 'pm=0.3', 'pm=0.4'],
             {'pm=0.02': '0.02', 'pm=0.05': '0.05', 'pm=0.08': '0.08', 'pm=0.1': '0.1',
              'pm=0.2': '0.2', 'pm=0.3': '0.3', 'pm=0.4': '0.4'}),
            ('population size', 'pop=20', ['pop=10', 'pop=20', 'pop=40', 'pop=52'],
             {'pop=10': '10', 'pop=20': '20', 'pop=40': '40', 'pop=52': '52'}),
            ('elite count', 'elite=5', ['elite=0', 'elite=1', 'elite=2', 'elite=4', 'elite=5',
                                        'elite=8'],
             {'elite=0': '0', 'elite=1': '1', 'elite=2': '2', 'elite=4': '4', 'elite=5': '5',
              'elite=8': '8'})]
    L = ['\\begin{tabular}{l c cc cc}', '\\toprule',
         '& & \\multicolumn{2}{c}{CIFAR-10} & \\multicolumn{2}{c}{CIFAR-100} \\\\',
         '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}',
         '\\textbf{Setting} & \\textbf{Pipeline} & \\textbf{Best} & \\textbf{Gain} '
         '& \\textbf{Best} & \\textbf{Gain} \\\\', '\\midrule']
    for lab, dflt, keys, nm in spec:
        cells = []
        for ds in ('cifar10', 'cifar100'):
            st = out[ds]['settings']
            bk = max(keys, key=lambda k: st[k]['best'])
            cells.append(f"{nm[bk]} & ${st[bk]['best'] - st[dflt]['best']:+.2f}$")
        L.append(f"{lab} & {nm[dflt]} & " + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(common.HERE, '..', 'tables', 'tab_settings.tex'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
