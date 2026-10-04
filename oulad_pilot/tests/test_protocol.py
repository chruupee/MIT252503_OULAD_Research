"""Unit tests for the decision rule, the paired student-clustered bootstrap and the pipeline outputs. Run: pytest -q"""
import json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from core import decision, paired_cluster_bootstrap, CFG
OUT = ROOT / "outputs"
M = CFG["evaluation"]["practical_margin"]

def test_decision_branches():
    assert decision(0.030, 0.012, 0.048, M) == "MEANINGFUL_IMPROVEMENT"
    assert decision(0.015, 0.004, 0.026, M) == "DISTINGUISHABLE_BELOW_MARGIN"
    assert decision(0.010, -0.004, 0.024, M) == "NOT_DISTINGUISHABLE_PREFER_LASSO"
    assert decision(0.025, -0.001, 0.051, M) == "NOT_DISTINGUISHABLE_PREFER_LASSO"
    assert decision(-0.009, -0.013, -0.006, M) == "BASELINE_BETTER"

def test_bootstrap_identity_and_known_better():
    r = np.random.default_rng(0); n = 3000; y = r.integers(0, 2, n); g = r.integers(0, 2000, n); p = r.random(n) + 0.3 * y
    d, *_ = paired_cluster_bootstrap(y, p, p, g, 200, 42); assert np.allclose(d, 0)
    d, *_ = paired_cluster_bootstrap(y, p, p + 0.8 * y, g, 200, 42); assert np.percentile(d, 2.5) > 0

def test_all_feature_and_window_checks_pass():
    c = json.loads((OUT / "feature_checks.json").read_text()); assert c["passed"] == c["total"] >= 21
    ids = {x["id"] for x in c["checks"]}; assert {"T16", "T17", "T18"} <= ids

def test_folds_forward_and_sealed_free():
    f = json.loads((OUT / "fold_definition.json").read_text())["folds"]
    assert all(v["strictly_forward"] for v in f.values()); assert not any(v["sealed_in_fold"] for v in f.values())
    assert f["FINAL"]["label_latency_safe"] and not f["F3"]["label_latency_safe"]

def test_sealed_labels_never_written():
    e = pd.read_csv(OUT / "registrations_eligibility.csv"); assert e[e.period == "P4"].at_risk.isna().all()
    assert "at_risk" not in pd.read_csv(OUT / "features_sealed_nolabels.csv", nrows=5).columns

def test_sealed_run_refused_without_approval():
    lock = json.loads((OUT / "protocol_lock.json").read_text())
    if lock["supervisor_approval"]["appendix_v_signed"]:
        return
    r = subprocess.run([sys.executable, str(ROOT / "src" / "s08_sealed_evaluation.py"), "--sealed"], capture_output=True, text=True, cwd=ROOT / "src")
    assert r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr)
