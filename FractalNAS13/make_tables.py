"""Rewrite every LaTeX table of the manuscript from the archived results in analysis/results/.

Nothing is recomputed: each analysis script's own table writer is called on the JSON that
script saved, so a table can only differ from the archived one if the results did. Run from
analysis/ (run_all.sh does this).
"""
import importlib
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import common  # noqa: E402

J = lambda name: json.load(open(os.path.join(common.RES, name + '.json')))

JOBS = [('s16_surrogate', '_tex', 'surrogate'),
        ('s18_design_rules', '_tex', 'design_rules'),
        ('s19_levers', '_tex', 'levers'),
        ('s20_operator_bench', '_tex', 'operator_bench'),
        ('s21_segment_model', '_tex', 'segment_model'),
        ('s26_search_value', '_tex', 'search_value'),
        ('s37_extension', 'emit_table', 'extension'),
        ('s38_gate', 'emit_table', 'gate'),
        ('s38_gate', 'emit_inventory', 'gate'),
        ('s56_bench_profile', '_tex', 'bench_profile'),
        ('s62_significance', '_tex', 'significance'),
        ('s63_real_validation', '_tex', 'real_validation'),
        ('s64_surrogate_family', '_tex', 'surrogate_family'),
        ('s65_holdout_tuning', '_tex', 'holdout_tuning'),
        ('s68_adopt', 'main', None),
        ('s58_sm_tables', 'main', None)]

failed = 0
for mod, fn, res in JOBS:
    try:
        f = getattr(importlib.import_module(mod), fn)
        f() if res is None else f(J(res))
        print('  ok   %s.%s' % (mod, fn))
    except Exception as e:  # report and continue, so one failure does not hide the others
        failed += 1
        print('  FAIL %s.%s: %r' % (mod, fn, e))
sys.exit(1 if failed else 0)
