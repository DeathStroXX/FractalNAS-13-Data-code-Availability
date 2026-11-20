#!/usr/bin/env bash
# Reproduce the figures, tables and statistics of the manuscript.
#
#   bash run_all.sh figures   redraw every figure from the archived results        (~10 s)
#   bash run_all.sh tables    rewrite every LaTeX table from the archived results  (~5 s)
#   bash run_all.sh stats     re-run the statistical analyses                     (~1 min)
#   bash run_all.sh all       rebuild every result from the raw logs, then all of
#                             the above                                (~1.5 h, one core)
#   bash run_all.sh verify    run figures, tables and stats in a temporary copy and
#                             report whether every output is identical to the archived one
#
# figures, tables and stats need only this folder. `all` and the Pareto-front figure also need
# the run logs in logs/ and the gene table in data/ (available on request), and three of the
# steps need the pod logs: run `unzip ../fractalnas13_large_input_data.zip` here, next to analysis/.
# Set PYTHON=/path/to/python to use another interpreter.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
HAVE_LOGS=0
[ -d logs ] && [ -f data/gene_diversity_benchmark_updated.csv ] && HAVE_LOGS=1

figures() {
  echo "== figures"
  local which="bench settings space ops"
  [ "$HAVE_LOGS" = 1 ] && which="$which pareto" \
    || echo "   (fig_pareto skipped: it needs logs/ and the gene table; the archived copy is in figures/)"
  (cd analysis && $PY s57_figures.py $which)
  (cd analysis && $PY -c "import json, s37_extension as m; m.emit_figure(json.load(open('results/extension.json')))")
  (cd analysis && $PY -c "import json, s38_gate as m; m.emit_figure(json.load(open('results/gate.json')))")
}

tables() {
  echo "== tables"
  (cd analysis && $PY ../make_tables.py)
}

stats() {
  echo "== statistics"
  (cd analysis && $PY s62_significance.py)      # paired tests, Holm, effect sizes, bootstrap CIs
  (cd analysis && $PY s66_confirm_power.py)     # what the registered confirmation can decide
  if [ "$HAVE_LOGS" = 1 ]; then
    (cd analysis && $PY s70_equal_budget.py)    # real-run contrasts at an equal budget
  else
    echo "   (s70_equal_budget skipped: it reads logs/; its archived result is results/equal_budget.json)"
  fi
}

all() {
  if [ "$HAVE_LOGS" != 1 ]; then
    echo "run_all.sh all needs logs/ and data/gene_diversity_benchmark_updated.csv (on request)" >&2
    exit 1
  fi
  echo "== full rebuild from the logs (dependency order)"
  for s in s1_statistics s16_surrogate s2_operator_audit s19_levers s26_search_value \
           s56_bench_profile s63_real_validation s70_equal_budget s64_surrogate_family \
           s65_holdout_tuning s28_space s18_design_rules s37_extension s20_operator_bench \
           s21_segment_model s38_gate s68_adopt s62_significance s66_confirm_power \
           s14_learning_curves s69_censoring confirm/price_registered_b2 \
           confirm/price_registered_b1; do
    echo "-- $s"; (cd analysis && $PY $s.py)
  done
  figures; tables
}

case "${1:-}" in
  figures) figures ;;
  tables)  tables ;;
  stats)   stats ;;
  all)     all ;;
  verify)  $PY verify_results.py ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
