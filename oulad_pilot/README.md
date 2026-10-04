# oulad_pilot v1.1 — MIT252503 pre-seal pipeline

Status: **Prototype/Pilot** until re-run on the student's workstation and committed (WP0).
The sealed test period (P4, 2014J) has NOT been evaluated under this protocol.

## Reproduce (Week 1, WP0)
1. Put the 7 OULAD CSVs (UCI mirror) in `data/raw/oulad/` and the 3 KDD processed files in `data/kdd/`.
2. `pip install -r requirements.txt`
3. `bash run_pilot.sh`  → regenerates every pre-seal output, the P3 rehearsal, the lock and the tests (~3 min).
4. Compare with Chapter 6 Tables 6.7–6.18; commit code, config and outputs; `git tag v1.1-lock`.

## After the supervisor signs Appendix V
Edit `outputs/protocol_lock.json` → `supervisor_approval`: set `appendix_v_signed: true`, name and date, commit, then run once:
`cd src && python s08_sealed_evaluation.py --sealed`
The script refuses to run if config/code changed since the lock or if it has already been run.

## Cohort B raw-log features (workstation only)
`python src/s09b_kdd_shared_features.py --raw path/to/kddcup2015`

Parts of this code were drafted with an AI assistant; the student reviews, executes and is responsible for it (MIT AI-use disclosure).
