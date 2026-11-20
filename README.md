# Evaluating LLM-Written Operators in Evolutionary Neural Architecture Search

**Acceptance rules and search-configuration effects, audited from a completed search's own logs**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24261318.svg)](https://doi.org/10.5281/zenodo.24261318)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-blue.svg)](LICENSE-DATA)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)

A language model that rewrites a genetic algorithm's operators inside a running search is a
proposal generator; what its proposals are worth depends on the rule that admits them. This
repository contains the analysis code and the committed numerical results behind a study that
fits a cheap surrogate benchmark, **FractalNAS-13**, to the logs of 34 completed LLM-guided
neural architecture searches over a 13-gene FractalNet space on CIFAR-10 and CIFAR-100, and
uses it to price the search's own decisions: its GA settings, its fitness signal, and the 68
operator rewrites three language models wrote and the pipeline accepted.

Every number, table and figure in the accompanying manuscript and its supplement is produced
by a script in `FractalNAS13/` and stored under `FractalNAS13/analysis/results/`,
`FractalNAS13/tables/` or `FractalNAS13/figures/`. Nothing here requires a GPU: the training
runs are already recorded.

Authors: Dmitry Ignatov, Akashdeep Singh, Sai Revanth Mulukuledu, Radu Timofte
(Computer Vision Lab, CAIDAS & IFI, University of Würzburg).

---

## What is in this repository

```
README.md                         this file
LICENSE                           MIT, for the code
LICENSE-DATA                      CC BY 4.0, for the results, data and pre-registration
PREREGISTRATION.md                pre-registration of the real-training confirmation (unmodified)
fractalnas13_large_input_data.zip pipeline pod logs of all 34 runs (+ README, SHA256SUMS)
fractalnas13_large_input_data.zip.sha256
FractalNAS13/
├── run_all.sh                    driver: figures | tables | stats | all | verify
├── verify_results.py             re-runs figures, tables and statistics and diffs them
├── make_tables.py                calls each analysis script's table writer on its result
├── requirements.txt              the package versions the committed outputs were made with
├── analysis/
│   ├── fractalbench.py           the FractalNAS-13 benchmark and the GA harness
│   ├── surrogate.py              ridge surrogate on gene values and gene-value pairs
│   ├── policies.py               screening, regularized evolution, local search,
│   │                             predictor-guided search
│   ├── common.py                 reading the run logs
│   ├── pipeline_mutate_gene.py   the audited pipeline's own mutation operator
│   ├── statsutil.py, figstyle.py
│   ├── s1_ … s70_*.py            the analysis pipeline, one script per result
│   ├── confirm/                  pre-registration copy, its scorer, and the scripts that
│   │                             price the registered claims
│   └── results/                  committed results, one JSON per analysis step
├── tables/                       the LaTeX tables the manuscript includes
├── figures/                      the figures the manuscript includes (PDF, PNG previews)
└── data/                         derived per-architecture files (cost model, learning
                                  curves, held-out surrogate predictions)
```

The manuscript sources (`main.tex`, `supplementary.tex`) are **not** part of this repository;
the scripts regenerate the tables and figures into `FractalNAS13/tables/` and
`FractalNAS13/figures/`.

---

## Quick start

```bash
git clone https://github.com/DeathStroXX/FractalNAS-13-Data-code-Availability.git
cd FractalNAS-13-Data-code-Availability/FractalNAS13
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

bash run_all.sh figures      # redraws every figure from the committed results  (~10 s)
bash run_all.sh tables       # rewrites every LaTeX table                        (~5 s)
bash run_all.sh stats        # re-runs the statistical analyses                  (~1 min)
python3 verify_results.py    # the three above in a temp copy, diffed against the
                             # committed outputs; exit status 0 when all match
```

`verify_results.py` copies the folder to a temporary directory and compares every output with
the committed one: tables, results and PNG figures byte for byte, PDF figures with their
embedded timestamps ignored. Nothing in your checkout is modified.

> If `./run_all.sh` reports *Permission denied*, use `bash run_all.sh …` or run
> `chmod +x run_all.sh` once.

---

## Which output is which part of the manuscript

| Output (under `FractalNAS13/`) | In the manuscript | Produced by |
|---|---|---|
| `tables/tab_profile.tex`, `figures/fig_bench.pdf` | Table I, Fig. 1 (fidelity, cost, resolution) | `s56_bench_profile.py`, `s57_figures.py` |
| `tables/tab_realval.tex` | Table II (held-out real runs) | `s63_real_validation.py` |
| `tables/tab_transfer.tex` | Table III (model families, run splits) | `s65_holdout_tuning.py`, `s64_surrogate_family.py` |
| `tables/tab_search_value.tex` | Table IV (search policies) | `s26_search_value.py` |
| `tables/tab_settings.tex`, `figures/fig_settings.pdf` | Table V, Fig. 2 (GA settings) | `s19_levers.py`, `s57_figures.py` |
| `tables/tab_inventory.tex`, `tables/tab_gate.tex` | Tables VI and VII (operators, gate) | `s38_gate.py`, `s20_operator_bench.py` |
| `tables/tab_tests.tex` | Table VIII (statistical tests) | `s62_significance.py` |
| remaining `tables/*.tex`, `figures/fig_{gate,ops,space,pareto,extension}.pdf` | Supplementary material | the script named in each file's caption |

---

## Full rebuild

Figures, tables and statistics run from the committed results alone. `run_all.sh all`
recomputes every result from the raw logs of the 34 runs, in dependency order, then redraws
the figures and tables (about 1.5 h on one CPU core). It needs two inputs:

```bash
cd FractalNAS13
# 1. the per-run evaluation and operator logs and the gene table (on request, see below):
#      logs/<arm>/<dataset>/seed<N>_<uuid>/*.jsonl
#      data/gene_diversity_benchmark_updated.csv
# 2. the pipeline pod logs, from this repository, unzipped next to analysis/:
unzip ../fractalnas13_large_input_data.zip
bash run_all.sh all
```

Set `PYTHON=/path/to/venv/bin/python` if the system interpreter lacks the stack. Without the
logs, `figures`, `tables` and `stats` still run; the Pareto-front figure and the equal-budget
contrast, which read the logs directly, are skipped and their committed copies stand.

---

## Data sources

No training is performed by this code; all inputs are recorded runs.

| Input | Location | Used for |
|---|---|---|
| Pipeline pod logs, 34 runs | `fractalnas13_large_input_data.zip` in this repository and on Zenodo, [doi:10.5281/zenodo.24261318](https://doi.org/10.5281/zenodo.24261318) | segment split, learning curves, truncation rates |
| Pre-registration | `PREREGISTRATION.md` (also `FractalNAS13/analysis/confirm/`) | the registered confirmation |
| Per-run evaluation and operator logs, gene table, 68 extracted operators | on request from the corresponding author (dmytro.ignatov@uni-wuerzburg.de) | full rebuild |
| Source code of the audited pipeline | [`ABrain-One/nn-gpt`](https://github.com/ABrain-One/nn-gpt/tree/main/ab/gpt/brute/ga/meta_evolution) | the GA and operators the benchmark models |

---

## Reproducibility

**What is guaranteed.** Every random quantity is seeded, and the scripts re-execute themselves
with `PYTHONHASHSEED=0`, so a re-run reproduces the committed outputs exactly for the package
versions in `requirements.txt`; `verify_results.py` checks this.

**The one machine-dependent number.** The wall-clock cost of a benchmark search
(about 8 ms on two CPU cores) depends on the machine; the manuscript quotes it approximately.

**The benchmark models the pipeline as it ran.** Population 20, tournament size 3,
single-point crossover, five elites, and a two-stage mutation (each gene offered with
probability 0.2, redrawn with probability 0.4: an effective per-gene rate of 0.08), as read
from the pipeline's code and confirmed in its pod logs. The audited GA selected on an LLM
predictor's estimate; the benchmark's GA selects on the measured 3-epoch accuracy, and the
difference between the two signals is priced separately (`s19_levers.py`).

**Environment used for the committed outputs.** Python 3.10.6, numpy 1.26.4, scipy 1.15.3,
scikit-learn 1.7.2, matplotlib 3.10.8, statsmodels 0.15.0.

**Statistical conventions.** Configurations within a family share random seeds and are
compared as paired blocks: Friedman tests per family, Wilcoxon signed-rank tests for pairs,
Mann–Whitney *U* for policies that share no randomness, Holm–Bonferroni within each family,
an effect size beside every corrected *p*-value. Operator classes use a dataset-stratified
sign-flip permutation test; bracketed quantities are 95% bootstrap percentile intervals.

---

## Scope of the evidence

34 completed searches, 10,082 per-architecture records (1- and 3-epoch accuracy):

| Arm | CIFAR-10 runs | CIFAR-100 runs |
|---|---|---|
| Baseline GA (one 26-generation search) | 3 | 3 |
| Frozen handcrafted operators, five restart segments | 5 | 5 |
| LLM-written operators (Qwen2.5-Coder-7B, DeepSeek-Coder-6.7B, Mistral-7B; 4-bit) | 9 | 9 |

The search space is a 13-gene FractalNet grid of about 4.2 × 10⁷ architectures. All reported
gains are differences between configurations on the fitted landscape, measured against the
3-epoch accuracy the runs recorded; they are not improvements in fully trained accuracy, and
they are specific to this search space, pipeline and pair of datasets.

---

## Citation

If you use this code or the committed results, please cite the archived release:

```bibtex
@software{ignatov_fractalnas13_artifacts,
  author  = {Ignatov, Dmitry and Singh, Akashdeep and Mulukuledu, Sai Revanth and
             Timofte, Radu},
  title   = {Evaluating LLM-Written Operators in Evolutionary Neural Architecture
             Search: analysis code and results},
  year    = {2026},
  doi     = {10.5281/zenodo.24261318},
  url     = {https://github.com/DeathStroXX/FractalNAS-13-Data-code-Availability}
}
```

The DOI above resolves to the concept record, which always points at the newest version; each
release also receives its own version DOI. Please also cite the accompanying article once it
appears.

---

## License

Code in `FractalNAS13/` is released under the MIT License (`LICENSE`). The committed results,
tables, figures and data under `FractalNAS13/`, the pod logs and the pre-registration are
released under CC BY 4.0 (`LICENSE-DATA`).
