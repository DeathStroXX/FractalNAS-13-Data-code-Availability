"""Score the pre-registered confirming runs against PREREGISTRATION.md.

The point of this file is that it exists *before* the runs do.  It reads the logs of the four
arms, computes the two registered endpoints, and prints the verdict the registration fixed in
advance.  It makes no choices: the arms, the seeds, the endpoint, the half-of-predicted
threshold and the four possible verdicts are all written down in PREREGISTRATION.md and are
mirrored here as constants.

TWO DEFECTS IN THE REGISTERED DOCUMENT, recorded here and in Supplementary Section S9 rather
than repaired in it, because a registration its authors may correct after the fact registers
nothing and the document is hashed:

  * B2's registered prediction (+0.55, +0.35) is the benchmark's CLASS MEAN over the mutation
    rewrites -- by_kind['mutate_gene']['mean'] in results/operator_bench.json -- not its
    estimate for the single c=2 rewrite the claim names.  That estimate is +1.06 +- 0.18 on
    CIFAR-10; on CIFAR-100 the benchmark has none, because each rewrite is replayed only on
    the landscape of the run that produced it and the CIFAR-100 file of the same name is a
    c=3 gate.  The registered numbers still decide the verdict.

  * B2's prediction was computed under a different SCHEDULE from the one its arms will run.
    s20_operator_bench.py replays each rewrite under the pipeline's five-restart schedule --
    5 segments x 5 generations x population 20, i.e. 500 evaluations -- while section 3 of the
    registration defines the B2 arms as `b1-asrun`, a single 26-generation run of 520
    evaluations.  The endpoint itself is not the problem: with one segment, "the best per
    segment, averaged over a run's segments" is the best found in the run, which is exactly
    what the replay reports (fractalbench.run_ga's `best`).  The schedule is: a rewrite worth
    +0.55 under five restarts need not be worth +0.55 under one long run, and the registered
    arms cannot test the schedule the prediction came from.

Both are recorded; neither is repaired; the registered numbers and the registered rule decide
the verdict exactly as written.

    python3 analysis/confirm/score_confirm.py            # -> results/confirm.json

With no logs present it prints "no data" per claim and exits 0, which is the state the released
archive is in.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402

LOGDIR = os.path.join(HERE, 'logs')
SEEDS = [4001, 4002, 4003]
N_BOOT = 10000
HALF = 0.5

#: the endpoint the registration fixes for each claim.  B2's arms are single-segment runs, so
#: its "mean best per segment" is the best found in the run -- the same functional the replay
#: reports; what differs is the schedule the prediction was computed under (see the docstring).
ENDPOINTS = {
    'B1': dict(label='best 3-epoch accuracy at 520 unique architectures'),
    'B2': dict(label='mean best 3-epoch accuracy per segment, which on a single-segment arm '
                     'is the best found in the run'),
}

#: claim -> (treatment arm, control arm, predicted effect per dataset)
CLAIMS = {
    'B1': dict(section='V', table='V', treatment='b1-tuned', control='b1-asrun',
               predicted={'cifar10': 1.43, 'cifar100': 1.85}),
    'B2': dict(section='VI', table='S11', treatment='b2-rewrite', control='b2-handcrafted',
               predicted={'cifar10': 0.55, 'cifar100': 0.35},
               prediction_is='class mean over the mutation rewrites, not this rewrite, '
                             'and measured under the five-restart schedule rather than the '
                             '26-generation schedule these arms run'),
}


def _arm_values(arm, ds):
    """Endpoint value per seed for one arm, or [] when the runs have not been done."""
    path = os.path.join(LOGDIR, ds, arm + '.json')
    if not os.path.exists(path):
        return []
    rec = json.load(open(path))
    return [float(rec[str(s)]) for s in SEEDS if str(s) in rec]


def _verdict(diff, predicted):
    if not len(diff):
        return 'no data'
    m = float(np.mean(diff))
    if m <= 0:
        return 'refuted'
    if m >= HALF * predicted:
        return 'confirmed'
    return 'directionally confirmed, magnitude not'


def main():
    out = {'logdir': LOGDIR, 'seeds': SEEDS, 'half_threshold': HALF, 'n_boot': N_BOOT,
           'rule': 'see analysis/confirm/PREREGISTRATION.md section 2', 'claims': {}}
    rng = np.random.default_rng(0)
    for name, c in CLAIMS.items():
        ep = ENDPOINTS[name]
        rec = {'section': c['section'], 'table': c['table'], 'endpoint': ep['label'],
               'by_dataset': {}}
        if 'prediction_is' in c:
            rec['prediction_is'] = c['prediction_is']
        for ds, pred in c['predicted'].items():
            t, k = _arm_values(c['treatment'], ds), _arm_values(c['control'], ds)
            n = min(len(t), len(k))
            diff = np.array(t[:n]) - np.array(k[:n])          # paired by seed
            cell = {'predicted': pred, 'threshold': HALF * pred, 'seeds': SEEDS[:n],
                    'status': _verdict(diff, pred)}
            if n:
                boot = [float(rng.choice(diff, n, replace=True).mean())
                        for _ in range(N_BOOT)]
                cell.update(mean=float(diff.mean()),
                            ci=[float(np.percentile(boot, 2.5)),
                                float(np.percentile(boot, 97.5))])
            rec['by_dataset'][ds] = cell
        out['claims'][name] = rec

    json.dump(out, open(os.path.join(common.RES, 'confirm.json'), 'w'), indent=1)
    for name, rec in out['claims'].items():
        for ds, cell in rec['by_dataset'].items():
            got = ('%+.2f' % cell['mean']) if 'mean' in cell else '--'
            print('%s (Sec. %s, Table %s) %-9s predicted %+.2f, measured %s: %s'
                  % (name, rec['section'], rec['table'], ds, cell['predicted'], got,
                     cell['status']))
    if any('prediction_is' in r for r in out['claims'].values()):
        print('note: B2\'s registered prediction is a %s.\n'
              '      See the module docstring and Supplementary Section S9.'
              % CLAIMS['B2']['prediction_is'])
    print('wrote analysis/results/confirm.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
