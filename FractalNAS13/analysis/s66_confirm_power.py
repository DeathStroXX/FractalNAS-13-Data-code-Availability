"""Operating characteristics of the pre-registered confirmation, from the record's own spread.

`analysis/confirm/PREREGISTRATION.md` fixes two claims, three seeds per arm, and a rule that
reads a mean difference as confirmation when it has the predicted sign and reaches half the
predicted size.  That document is hashed and time-stamped, so it is not edited here.  What this
script does is state, from data the archive already holds, what such a rule can and cannot
decide -- because a registration whose operating characteristics are unknown is a commitment
rather than a test, and a reader is entitled to both.

Two quantities fix everything:

  the spread of the endpoint across runs of one arm, which the record measures directly
    (`real_arm_sd` in results/bench_profile.json: the run-to-run standard deviation of the
    best architecture found, within each arm of the released study); and

  how much of that spread the registered pairing removes.  (For B2, whose endpoint is a
  per-segment mean rather than a best-of-run, the arm-level spread is an upper bound, because
  averaging a run's segments cannot increase the variance; the figures reported for B2 are
  conservative in that direction and the record holds no per-segment arm to measure instead.)

  How much of that spread the registered pairing removes.  The registered arms share their
  three seeds, so each pair starts from the identical initial population.  The benchmark
  measures what that is worth: results/levers.json keeps all 1,000 per-replicate outcomes of
  every configuration, run under common random numbers, so the correlation between replicate
  r of k=3 and replicate r of k=10 is the pairing's own correlation on this landscape.  It is
  reported here and used as the optimistic bound; zero correlation is the conservative one.

For each claim, dataset and pairing assumption the script reports

  P(confirmed)   under the benchmark's own point prediction and under a true effect of zero;
  P(refuted)     under the benchmark's own point prediction;
  power          of the two-sided paired t-test at alpha = 0.05 that the registration does not
                 require but a reader will look for;
  seeds          per arm that an 80%-powered test of the same effect would need.

The conclusion the paper draws from this is not that the confirmation is worthless: a sign that
comes out right is still evidence, and the runs cost 58 GPU-hours rather than the thousands a
powered test needs.  It is that the confirmation has to be reported as a sign check with known
error rates, and that the impossibility of powering it on real hardware is the paper's own
argument for the benchmark.

Reads results/bench_profile.json, results/levers.json, results/operator_bench.json and
analysis/confirm/PREREGISTRATION.md.  Writes results/confirm_power.json.

    python3 analysis/s66_confirm_power.py
"""
import json
import os
import re

import numpy as np
from scipy import stats

import common

ALPHA = 0.05
TARGET_POWER = 0.80
#: seeds per arm the registration fixes
N_SEEDS = 3
#: the registered rule reads a mean difference as confirmation at this share of the prediction
HALF = 0.5
#: GPU-hours one confirming run costs, as the registration states.  It is well below the 9.6
#: hours the released runs averaged (results/bench_profile.json) because a confirming arm runs
#: no language model: no operator rewriting between segments and no LoRA predictor inference,
#: only the GA and the 3-epoch training the released pipeline also did.  Both costs are
#: reported, since the second is the one a reader will have in mind from Table I.
GPU_HOURS_PER_RUN = 3.2


def _registered_predictions():
    """The two predicted effects, read out of the registered document itself."""
    md = open(os.path.join(common.HERE, 'confirm', 'PREREGISTRATION.md')).read()
    out = {}
    for claim in ('B1', 'B2'):
        row = re.search(r'\*\*%s\*\*.*?\n' % claim, md, re.S)
        if row is None:
            raise SystemExit('%s is not a row of PREREGISTRATION.md' % claim)
        nums = re.findall(r'\*\*\+?(-?\d+\.\d+)\*\* \((C10|C100)\)', row.group(0))
        got = {{'C10': 'cifar10', 'C100': 'cifar100'}[w]: float(v) for v, w in nums}
        if set(got) != set(common.DATASETS):
            raise SystemExit('%s: expected one prediction per dataset, read %s' % (claim, got))
        out[claim] = got
    return out


def _pairing_correlation(lv):
    """Correlation between the two registered B1 arms on the benchmark, under common random
    numbers: replicate r of k=3 and replicate r of k=10 start from the identical population."""
    out = {}
    for ds in common.DATASETS:
        s = lv[ds]['settings']
        a = np.asarray(s['k=3']['reps'], float)
        b = np.asarray(s['k=10']['reps'], float)
        out[ds] = float(np.corrcoef(a, b)[0, 1])
    return out


def _characteristics(sd_run, rho_pair, predicted, n=N_SEEDS, released_hours=9.6):
    """What the registered rule does, at one arm-level spread and one pairing correlation."""
    sd_diff = sd_run * np.sqrt(2.0 * (1.0 - rho_pair))        # one paired difference
    se = sd_diff / np.sqrt(n)                                 # its mean over n seeds
    thr = HALF * predicted
    # the registered rule, read off the sampling distribution of that mean
    p_conf_true = float(1.0 - stats.norm.cdf((thr - predicted) / se))
    p_conf_null = float(1.0 - stats.norm.cdf(thr / se))
    p_ref_true = float(stats.norm.cdf((0.0 - predicted) / se))
    p_dir_true = float(max(0.0, 1.0 - p_conf_true - p_ref_true))
    # the test the registration does not require, for a reader who looks for one
    d = predicted / sd_diff
    nc, dof = d * np.sqrt(n), n - 1
    tcrit = stats.t.ppf(1.0 - ALPHA / 2.0, dof)
    power = float(1.0 - stats.nct.cdf(tcrit, dof, nc) + stats.nct.cdf(-tcrit, dof, nc))
    n80 = _seeds_for_power(d, TARGET_POWER)
    return dict(sd_run=float(sd_run), rho_pair=float(rho_pair), sd_diff=float(sd_diff),
                se_mean=float(se), predicted=float(predicted), threshold=float(thr),
                p_confirm_if_true=p_conf_true, p_confirm_if_null=p_conf_null,
                p_refute_if_true=p_ref_true, p_directional_if_true=p_dir_true,
                cohen_d=float(d), power_paired_t=power, seeds_for_80_power=n80,
                gpu_hours_for_80_power=(None if n80 is None
                                        else round(2 * n80 * GPU_HOURS_PER_RUN)),
                gpu_hours_for_80_power_at_released_rate=(
                    None if n80 is None else round(2 * n80 * released_hours)))


def _seeds_for_power(d, target):
    for n in range(2, 100001):
        dof = n - 1
        nc = d * np.sqrt(n)
        tcrit = stats.t.ppf(1.0 - ALPHA / 2.0, dof)
        if 1.0 - stats.nct.cdf(tcrit, dof, nc) + stats.nct.cdf(-tcrit, dof, nc) >= target:
            return int(n)
    return None


def main():
    prof = json.load(open(f'{common.RES}/bench_profile.json'))
    lv = json.load(open(f'{common.RES}/levers.json'))
    pred = _registered_predictions()
    rho = _pairing_correlation(lv)

    rel = {ds: float(prof[ds]['real_hours_mean']) for ds in common.DATASETS}
    out = {'alpha': ALPHA, 'n_seeds': N_SEEDS, 'half_threshold': HALF,
           'gpu_hours_per_run_registered': GPU_HOURS_PER_RUN,
           'gpu_hours_per_run_released': rel,
           'target_power': TARGET_POWER, 'pairing_rho_on_benchmark': rho,
           'source': 'arm-level run-to-run sd of the best architecture found, '
                     'results/bench_profile.json real_arm_sd',
           'claims': {}}
    for claim in ('B1', 'B2'):
        rec = {}
        for ds in common.DATASETS:
            # the spread a three-run arm of the released study actually showed, pooled over
            # the arms of that dataset: this is what three real seeds deliver
            sds = np.asarray(list(prof[ds]['real_arm_sd'].values()), float)
            sd_pooled = float(np.sqrt(np.mean(sds ** 2)))
            rec[ds] = {
                'sd_arm_pooled': sd_pooled,
                'sd_arm_by_arm': {k: float(v) for k, v in prof[ds]['real_arm_sd'].items()},
                'unpaired': _characteristics(sd_pooled, 0.0, pred[claim][ds],
                                             released_hours=rel[ds]),
                'paired_as_on_benchmark': _characteristics(sd_pooled, rho[ds],
                                                           pred[claim][ds],
                                                           released_hours=rel[ds]),
            }
        out['claims'][claim] = rec

    json.dump(out, open(f'{common.RES}/confirm_power.json', 'w'), indent=1)

    print('pairing correlation between the two B1 arms on the benchmark '
          '(common random numbers): %s'
          % ', '.join('%s %.2f' % (d, rho[d]) for d in common.DATASETS))
    for claim in ('B1', 'B2'):
        print('\n== %s, three seeds per arm ==' % claim)
        for ds in common.DATASETS:
            for mode in ('unpaired', 'paired_as_on_benchmark'):
                c = out['claims'][claim][ds][mode]
                print('  %-9s %-22s predicted %+.2f, s.e. of the 3-seed mean %.2f'
                      % (ds, mode, c['predicted'], c['se_mean']))
                print('            P(confirmed | effect true) %.2f   '
                      'P(confirmed | effect zero) %.2f   P(refuted | effect true) %.2f'
                      % (c['p_confirm_if_true'], c['p_confirm_if_null'],
                         c['p_refute_if_true']))
                print('            power of a paired t-test %.2f   '
                      'seeds for %d%% power %s (%s GPU-h registered, %s released)'
                      % (c['power_paired_t'], round(100 * TARGET_POWER),
                         c['seeds_for_80_power'], c['gpu_hours_for_80_power'],
                         c['gpu_hours_for_80_power_at_released_rate']))
    print('\nwrote analysis/results/confirm_power.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
