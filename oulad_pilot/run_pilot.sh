#!/usr/bin/env bash
# Regenerates every pre-seal output (stages 0-7 and 9) and the rehearsal, then runs the tests.
# The sealed evaluation is NOT run here; after supervisor sign-off run:  (cd src && python s08_sealed_evaluation.py --sealed)
set -euo pipefail
cd "$(dirname "$0")/src"
rm -f ../outputs/run_log.txt
for s in s00_data_manifest s01_eligibility s02_split_manifest s03_features s04_feature_checks s05_baseline_lasso s06_xgboost_inner s06b_selection_outputs s07_protocol_lock s09_cohortB_audit; do
  python -W ignore "$s.py"
done
python -W ignore s08_sealed_evaluation.py --rehearsal
python s07_protocol_lock.py
cd .. && python -m pytest -q tests | tee outputs/pytest_log.txt
echo "PRE-SEAL PIPELINE COMPLETE - sealed period (P4) labels not read"
