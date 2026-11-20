"""Price claim B1 exactly as PREREGISTRATION.md words it, in the pipeline's own code.

B1 registers "the GA at k=10, p_m=0.08 against k=3, p_m=0.2 ... otherwise the pipeline's own
code". In that code p_m is the outer rate of a two-stage mutation (the handcrafted mutate_gene
redraws with probability 2 x p_m), so the registered treatment runs at an effective per-gene
rate of 0.08 x 0.16 = 0.0128 and the control at 0.2 x 0.4 = 0.08. The predictions in the
registration were computed with a one-stage operator, under which the same words mean 0.08
against 0.2. Both readings are priced here on the 1 x 26 schedule, 1,000 replicates, seed 1.

Writes results/registered_b1.json.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import common
import fractalbench as fb

R = 1000
out = {'r': R}
for ds in common.DATASETS:
    b = fb.Bench(ds)
    def best(**kw):
        return fb.run_ga(b, 1, 26, R=R, seed=1, **kw)['best']
    ctrl = best(k=3, pm=0.2)                         # the pipeline as run (two-stage, gate 2)
    treat = best(k=10, pm=0.08)                      # the registered words, in the same code
    intended = best(k=10, **fb.setting_kw('pm', 0.08))  # effective 0.08, k=10
    d = treat - ctrl
    out[ds] = dict(as_registered=float(d.mean()), as_registered_se=float(d.std() / np.sqrt(R)),
                   effective_treatment_pm=fb.effective_pm(0.08, fb.PIPELINE['gate']),
                   k10_at_pipeline_rate=float((intended - ctrl).mean()))
    print(f"{ds}: B1 as registered (k=10, outer p_m=0.08 -> effective "
          f"{out[ds]['effective_treatment_pm']:.4f}) {out[ds]['as_registered']:+.2f} "
          f"+- {out[ds]['as_registered_se']:.2f};  k=10 at the pipeline's effective 0.08 "
          f"{out[ds]['k10_at_pipeline_rate']:+.2f}")
json.dump(out, open(os.path.join(common.RES, 'registered_b1.json'), 'w'), indent=1)
