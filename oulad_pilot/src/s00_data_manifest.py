"""Stage 0: raw-data manifest and environment snapshot (Gate G1 smoke test)."""
from core import *
log = get_logger("s00")
PUBLISHED = {"studentInfo": 32593, "studentRegistration": 32593, "studentVle": 10655280, "vle": 6364,
             "studentAssessment": 173912, "assessments": 206, "courses": 22}
log.info("STAGE 0: raw-data manifest and environment snapshot (Gate G1 smoke test)")
env = env_snapshot()
for k, v in env.items(): log.info(f"ENV {k} = {v}")
rows, ok = [], True
for t, n in PUBLISHED.items():
    p = ROOT / CFG["data"]["raw_dir"] / f"{t}.csv"
    if not p.exists():
        ok = False; log.info(f"MISSING {t}"); continue
    df = read(t); good = len(df) == n; ok &= good
    rows.append({"table": t, "rows": len(df), "published_rows": n, "match": good, "missing_cells": int(df.isna().sum().sum()),
                 "bytes": p.stat().st_size, "sha256": sha256(p)})
    log.info(f"DATA {t}: rows={len(df):,} published={n:,} match={good} sha256={rows[-1]['sha256'][:16]}...")
write_json({"environment": env, "tables": rows, "g1_smoke_test": "PASS" if ok else "FAIL"}, "data_manifest.json")
log.info(f"G1 smoke test {'PASS' if ok else 'FAIL'}: all 7 tables present, row counts match published counts, hashes recorded")
if not ok: raise SystemExit(1)
