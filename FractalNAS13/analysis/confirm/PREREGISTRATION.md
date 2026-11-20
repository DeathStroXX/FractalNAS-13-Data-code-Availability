# The confirmation, pre-registered

Every number in Sections V and VI of the paper is a difference between configurations on a
*surrogate* landscape. That is the paper's main remaining weakness and it is named as such in
the Limitations. This document fixes, in advance, exactly what the confirming runs will be,
what counts as confirmation and what counts as refutation, so that the result cannot be chosen
after it is seen.

`score_confirm.py` in this directory scores the logs against the predictions below and makes no
further choices. Nothing in the manuscript depends on the outcome: the paper's claims are
stated as differences measured on the benchmark, and this is the test that would overturn
them.

Registered before any confirming run. One GPU; the two claims together are **about 58
GPU-hours** at 3.2 GPU-hours per run.

---

## 1. What is being tested

Two claims, each with a number the paper already commits to.

| # | Claim | Where | Predicted |
|---|---|---|---|
| **B1** | The GA at the settings the benchmark recommends (`k=10`, `p_m=0.08`) beats the GA at the settings the original study used (`k=3`, `p_m=0.2`), at an equal budget of 520 evaluated architectures. | Sec. V, Table V | **+1.43** (C10), **+1.85** (C100) |
| **B2** | The `c=2` mutation rewrite, spliced into the pipeline in place of the handcrafted mutation operator, beats it per segment. | Sec. VI, Suppl. Table S11 | **+0.55** (C10), **+0.35** (C100) |

Both predictions are point estimates from the benchmark at 1,000 replicates. The registered
test is on the **direction and the order of magnitude**, not on the second decimal, because
three real runs per arm cannot resolve a tenth of a point — which is the paper's own
Table I.

## 2. Confirmation and refutation, decided now

For each claim and each dataset, with three seeds per arm:

* **Confirmed** if the mean difference is positive and reaches at least **half** the predicted
  value (≥ +0.72 and +0.93 for B1; ≥ +0.28 and +0.18 for B2).
* **Directionally confirmed, magnitude not** if the mean difference is positive but below half
  the predicted value. This is the outcome we would read as *the benchmark overstates the
  effect*, and it is reported as such, in the abstract.
* **Refuted** if the mean difference is zero or negative on both datasets.
* **Inconclusive** on one dataset and confirmed on the other is reported as a partial
  confirmation, with both numbers, and the paper's claim is weakened to the dataset that
  confirmed it.

No other reading is permitted after the fact. In particular we do not add seeds after seeing
the result, we do not switch endpoint, and we do not drop a dataset.

## 3. The runs

Six runs per claim per dataset is beyond the budget, so each claim gets three seeds per arm on
each dataset: 12 runs for B1 and 12 for B2, 24 in all.

| Arm | Configuration |
|---|---|
| `b1-tuned` | the pipeline's own GA, `k=10`, `p_m=0.08`, population 20, 2 elites, 26 generations |
| `b1-asrun` | the same, `k=3`, `p_m=0.2` — the settings of the released study |
| `b2-rewrite` | `b1-asrun` with `analysis/operators/cifar10_deepseek_r2_it2_mutate_gene.py` (the `c=2` gate) spliced in as `mutate_gene` |
| `b2-handcrafted` | `b1-asrun`, unchanged |

Seeds: 4001, 4002, 4003 for every arm, so the arms share their initial populations pairwise.

Everything else is the released pipeline: the same 13-gene space, the same 3-epoch training
recipe, the same accuracy floor, the same duplicate cache, the same hardware class. The fitness
the GA selects on is the **measured 3-epoch accuracy**, not the LLM predictor, because the
predictor is not the object of either claim and its noise would cost resolution the budget
cannot spare. This is a deviation from the released configuration and it is registered here.

## 4. The endpoint

For B1: the **best measured 3-epoch accuracy** found by a run, at the point where it has
evaluated 520 unique architectures. For B2: the same quantity per segment, averaged over a
run's segments, which is the quantity the benchmark's per-rewrite replay reports.

The mean difference between arms is taken over the three seeds, paired by seed. A 95% bootstrap
interval over seeds is reported alongside, but the decision rule above deliberately does not
depend on it: at three seeds an interval is wide enough that requiring it to exclude zero would
make the test almost unfailable in the direction of "inconclusive".

## 5. What a refutation would mean

B1 refuted would say the settings result does not survive contact with a real landscape — most
plausibly because the surrogate's smoothness flatters a high selection pressure, which is the
first mechanism named in the Limitations. B2 refuted would say the per-rewrite replay is
measuring something about the harness rather than about the operator. Either outcome leaves the
benchmark's *resolution* claim untouched, and both would be reported in the paper's abstract,
not only in a limitations paragraph.

## 6. Deviations

Any deviation from this document is recorded here, dated, with the reason, before the affected
runs are scored.

*(none so far)*
