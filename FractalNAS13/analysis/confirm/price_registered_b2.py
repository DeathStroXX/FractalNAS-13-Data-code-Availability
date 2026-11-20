"""Price the operator the pre-registration names for B2 on the current benchmark.

PREREGISTRATION.md names `analysis/operators/cifar10_deepseek_r2_it2_mutate_gene.py`, the c=2
mutation gate of the run set the registration was written against. The record was later
re-based on the October 2026 release of the same pipeline's runs, and in that record the same
file name belongs to a different rewrite. The registered code is therefore kept verbatim in
`confirm/registered_b2_mutate_gene.py`, and this script replays exactly that code with the
harness and settings of s20_operator_bench.py (150 replicates, seed 1, the 5x5 schedule).

Writes results/registered_b2.json.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import common
import fractalbench as fb

R = 150
HERE = os.path.dirname(os.path.abspath(__file__))
f = os.path.join(HERE, 'registered_b2_mutate_gene.py')
out = {'file': 'analysis/confirm/registered_b2_mutate_gene.py', 'r': R}
for ds in common.DATASETS:
    b = fb.Bench(ds)
    # the registered rewrite replaces the pipeline's own mutate_gene, so that is the reference
    _, own = fb.load_operator(os.path.join(HERE, '..', 'pipeline_mutate_gene.py'))
    base = fb.run_ga(b, 5, 5, R=R, seed=1, ops={'mutate_gene': own}, timer_guard=False)
    kind, fn = fb.load_operator(f)
    r = fb.run_ga(b, 5, 5, R=R, seed=1, ops={kind: fn}, timer_guard=False)
    m, bm = float(r['best'].mean()), float(base['best'].mean())
    se = float(r['best'].std() / np.sqrt(R))
    out[ds] = dict(delta=m - bm, se=se)
    print(f'{ds}: registered B2 rewrite {m - bm:+.2f} +- {se:.2f}')
json.dump(out, open(os.path.join(common.RES, 'registered_b2.json'), 'w'), indent=1)
