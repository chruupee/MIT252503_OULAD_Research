"""Stage 7: write the protocol lock. Records hashes of config, code, feature files and every training-derived
decision. The supervisor approval block must be completed (Appendix V) before s08 will run."""
from core import *
log = get_logger("s07")
src = sorted((ROOT / "src").glob("*.py"))
lock = {"protocol_version": CFG["protocol_version"], "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_sha256": sha256(CFG_PATH), "code_sha256": {p.name: sha256(p) for p in src},
        "feature_files_sha256": {n: sha256(OUT / n) for n in ["features_nonsealed.csv", "features_sealed_nolabels.csv"]},
        "frozen_decisions": {"lasso": json.loads((OUT / "lasso_inner_results.json").read_text())["selected_C"],
                             "xgboost": json.loads((OUT / "xgb_inner_results.json").read_text()),
                             "selection": json.loads((OUT / "selection_outputs.json").read_text())},
        "pre_specified": {"practical_margin": CFG["evaluation"]["practical_margin"], "bootstrap": CFG["evaluation"]["bootstrap"],
                          "decision_rule": "four-branch paired rule (Section 6.8.2)", "calibration": CFG["calibration"]["rule"],
                          "primary_class_treatment": CFG["preprocessing"]["class_treatment"], "sensitivity_only": ["F3", "SMOTE (S5)", "S7"]},
        "supervisor_approval": {"appendix_v_signed": False, "approved_by": "", "date": "", "note": "complete before running s08"}}
p = OUT / "protocol_lock.json"
if p.exists():
    old = json.loads(p.read_text()); lock["supervisor_approval"] = old.get("supervisor_approval", lock["supervisor_approval"])
write_json(lock, "protocol_lock.json")
log.info(f"PROTOCOL LOCK written: config {lock['config_sha256'][:12]}..., {len(src)} code files hashed; approval signed = {lock['supervisor_approval']['appendix_v_signed']}")
