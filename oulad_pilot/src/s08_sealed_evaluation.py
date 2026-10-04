"""Stage 8: ONE-TIME sealed evaluation on P4 (2014J), exactly as frozen in outputs/protocol_lock.json.

Usage:
  python s08_sealed_evaluation.py --rehearsal   # code test only: trains on P1+P2, evaluates on P3 (no P4 access)
  python s08_sealed_evaluation.py --sealed      # the single thesis evaluation; refuses to run unless
                                                # protocol_lock.json carries the supervisor approval and all hashes match
Outputs are written with prefix 'R_' (sealed) or 'rehearsal_' (rehearsal).
"""
from core import *
import argparse, xgboost as xgbm
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, f1_score

ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--rehearsal", action="store_true"); g.add_argument("--sealed", action="store_true")
args = ap.parse_args()
log = get_logger("s08")
lock = json.loads((OUT / "protocol_lock.json").read_text())
pref = "R_" if args.sealed else "rehearsal_"

if args.sealed:
    if not lock["supervisor_approval"]["appendix_v_signed"]:
        raise SystemExit("REFUSED: supervisor approval not recorded in protocol_lock.json (Appendix V).")
    bad = [n for n, h in lock["code_sha256"].items() if sha256(ROOT / "src" / n) != h]
    if sha256(CFG_PATH) != lock["config_sha256"] or bad:
        raise SystemExit(f"REFUSED: config or code changed since the lock: {bad or 'config.yaml'}")
    if (OUT / "R_results.json").exists():
        raise SystemExit("REFUSED: the sealed evaluation has already been run (R_results.json exists). It is run once only.")
    log.info("SEALED EVALUATION AUTHORISED - P4 labels will now be read for the first and only time")

fd = lock["frozen_decisions"]; C = fd["lasso"]; P = fd["xgboost"]["selected_params"]; NT = fd["xgboost"]["final_n_estimators"]
SEL = fd["selection"]; margin = CFG["evaluation"]["practical_margin"]; B = CFG["evaluation"]["bootstrap"]["resamples"]; seed = CFG["seed"]
X = pd.read_csv(OUT / "features_nonsealed.csv")
if args.sealed:
    info = read("studentInfo")
    XS = pd.read_csv(OUT / "features_sealed_nolabels.csv").merge(info[KEY + ["final_result"]], on=KEY, how="left", validate="1:1")
    XS["at_risk"] = XS.final_result.isin(CFG["labels"]["primary"]["positive"]).astype(int)
    XS["withdrawn"] = XS.final_result.isin(CFG["labels"]["secondary"]["positive"]).astype(int)
    train, test = X[X.period.isin(CFG["final"]["train"])], XS
    dup_tr = pd.read_csv(OUT / "features_nonsealed_dup_retained.csv")
    dup_te = pd.read_csv(OUT / "features_sealed_nolabels_dup_retained.csv").merge(XS[KEY + ["at_risk"]], on=KEY)
    bx = pd.read_csv(OUT / "features_boundary_extra.csv"); bx_tr = bx[bx.period != "P4"]
    bx_te = bx[bx.period == "P4"].drop(columns=["at_risk", "withdrawn"]).merge(info[KEY + ["final_result"]], on=KEY)
    bx_te["at_risk"] = bx_te.final_result.isin(CFG["labels"]["primary"]["positive"]).astype(int); bx_te = bx_te.drop(columns="final_result")
else:
    train, test = X[X.period.isin(["P1", "P2"])], X[X.period == "P3"]
    dup = pd.read_csv(OUT / "features_nonsealed_dup_retained.csv"); dup_tr, dup_te = dup[dup.period.isin(["P1", "P2"])], dup[dup.period == "P3"]
    bx = pd.read_csv(OUT / "features_boundary_extra.csv"); bx_tr, bx_te = bx[bx.period.isin(["P1", "P2"])], bx[bx.period == "P3"]
info = read("studentInfo"); test = test.merge(info[KEY + ["gender", "age_band", "imd_band"]], on=KEY, how="left")
log.info(f"{pref}: train n={len(train):,} test n={len(test):,} students={test.id_student.nunique():,}")


def fit_both(tr, label="at_risk", smote=False, nt=NT):
    Xt, yt = tr[FEATURES], tr[label].to_numpy()
    if smote:
        Xt, yt = smote_resample(Xt, yt, seed)
        ml = lasso(C); ml.set_params(lasso__class_weight=None); ml.fit(Xt, yt)
        mx = xgb(P, np.zeros(1), n_estimators=nt); mx.set_params(scale_pos_weight=1.0); mx.fit(Xt, yt)
    else:
        ml = lasso(C).fit(Xt, yt); mx = xgb(P, yt, n_estimators=nt).fit(Xt, yt)
    return ml, mx


ml, mx = fit_both(train)
y = test.at_risk.to_numpy(); pl = ml.predict_proba(test[FEATURES])[:, 1]; px = mx.predict_proba(test[FEATURES])[:, 1]
for m, flag in [("lasso", SEL["lasso"]["isotonic_recalibration_triggered"]), ("xgb", SEL["xgb"]["isotonic_recalibration_triggered"])]:
    if flag:
        iso = pd.read_csv(OUT / f"isotonic_{m}.csv")
        cal = np.interp(pl if m == "lasso" else px, iso.x, iso.y)
        test[f"p_{m}_calibrated"] = cal
test = test.assign(p_lasso=pl, p_xgb=px)
test[KEY + ["at_risk", "withdrawn", "p_lasso", "p_xgb"]].to_csv(OUT / f"{pref}test_predictions.csv", index=False)

# ---- R3: discrimination, paired difference, decision
d, a_b, x_b, ns = paired_cluster_bootstrap(y, pl, px, test.id_student.to_numpy(), B, seed)
auc_l, auc_x = roc_auc_score(y, pl), roc_auc_score(y, px); delta = auc_x - auc_l
lo, hi = np.percentile(d, [2.5, 97.5]); branch = decision(delta, lo, hi, margin)
R = {"mode": "sealed" if args.sealed else "rehearsal (P1+P2 -> P3; NOT a thesis result)", "n_test": int(len(test)), "students": ns,
     "base_rate": float(y.mean()), "auc": {"lasso": auc_l, "lasso_ci": list(np.percentile(a_b, [2.5, 97.5])), "xgb": auc_x, "xgb_ci": list(np.percentile(x_b, [2.5, 97.5]))},
     "delta_auc": delta, "delta_ci": [float(lo), float(hi)], "bootstrap_valid_resamples": int(len(d)), "decision": branch, "practical_margin": margin}
log.info(f"{pref}AUC LASSO {auc_l:.4f} XGB {auc_x:.4f}; Delta {delta:+.4f} [{lo:+.4f}, {hi:+.4f}] -> {branch}")

# ---- calibration
cal = {}
for m, p in [("lasso", pl), ("xgb", px)]:
    q = pd.qcut(pd.Series(p).rank(method="first"), CFG["calibration"]["reliability_bins"], labels=False)
    rel = pd.DataFrame({"bin": q, "p": p, "y": y}).groupby("bin").agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()
    rel.to_csv(OUT / f"{pref}reliability_{m}.csv", index=False)
    cal[m] = {"brier": float(brier_score_loss(y, p)), "brier_constant": brier_constant(y), "recalibrated": f"p_{m}_calibrated" in test}
    if f"p_{m}_calibrated" in test: cal[m]["brier_after_isotonic"] = float(brier_score_loss(y, test[f"p_{m}_calibrated"]))
R["calibration"] = cal

# ---- thresholds frozen from training OOF
ops = []
for m, p in [("lasso", pl), ("xgb", px)]:
    for tname, t in [("youden", SEL[m]["youden_threshold"]), ("top20", SEL[m]["top20_threshold"])]:
        yh = (p >= t).astype(int); tp = int(((yh == 1) & (y == 1)).sum()); fp = int(((yh == 1) & (y == 0)).sum())
        ops.append({"model": m, "threshold_rule": tname, "threshold": t, "tpr": tp / max((y == 1).sum(), 1), "fpr": fp / max((y == 0).sum(), 1),
                    "precision": precision_score(y, yh, zero_division=0), "f1": f1_score(y, yh, zero_division=0), "flag_rate": float(yh.mean())})
pd.DataFrame(ops).to_csv(OUT / f"{pref}operating_points.csv", index=False); R["operating_points"] = ops

# ---- AUC by module
mods = [{"module": k, "n": len(g), "base_rate": float(g.at_risk.mean()), "auc_lasso": roc_auc_score(g.at_risk, g.p_lasso),
         "auc_xgb": roc_auc_score(g.at_risk, g.p_xgb)} for k, g in test.groupby("code_module") if g.at_risk.nunique() == 2]
pd.DataFrame(mods).to_csv(OUT / f"{pref}auc_by_module.csv", index=False)

# ---- R5 fairness at frozen thresholds, Wilson CIs and student-clustered bootstrap CI for EOD
imd = {v: k for k, vs in CFG["evaluation"]["fairness"]["imd_groups"].items() for v in vs}
test["imd_group"] = test.imd_band.map(imd)
rng = np.random.default_rng(seed); codes, uniq = pd.factorize(test.id_student)
order = np.argsort(codes, kind="stable"); starts = np.searchsorted(codes[order], np.arange(len(uniq))); ends = np.append(starts[1:], len(order))
boot_ix = [np.concatenate([order[starts[k]:ends[k]] for k in rng.integers(0, len(uniq), len(uniq))]) for _ in range(B)]
fair, eods = [], []
for m in ["lasso", "xgb"]:
    t = SEL[m]["youden_threshold"]; yh = (test[f"p_{m}"].to_numpy() >= t)
    for attr in CFG["evaluation"]["fairness"]["attributes"]:
        grp = test[attr].to_numpy(); ok = pd.notna(grp)
        groups = sorted(pd.unique(grp[ok]))
        for gname in groups:
            s = grp == gname; n1 = int((s & (y == 1)).sum()); n0 = int((s & (y == 0)).sum())
            tp = int((s & yh & (y == 1)).sum()); fp = int((s & yh & (y == 0)).sum())
            fair.append({"model": m, "attribute": attr, "group": gname, "n": int(s.sum()), "positives": n1, "tpr": tp / max(n1, 1), "tpr_ci": wilson(tp, n1),
                         "fpr": fp / max(n0, 1), "fpr_ci": wilson(fp, n0), "small_group_flag": bool(s.sum() < CFG["evaluation"]["fairness"]["small_group_flag"])})
        def gaps(ix):
            yy, hh, gg = y[ix], yh[ix], grp[ix]; tprs, fprs = [], []
            for gname in groups:
                s = gg == gname
                if (s & (yy == 1)).sum() == 0 or (s & (yy == 0)).sum() == 0: return None
                tprs.append((s & hh & (yy == 1)).sum() / (s & (yy == 1)).sum()); fprs.append((s & hh & (yy == 0)).sum() / (s & (yy == 0)).sum())
            return max(tprs) - min(tprs), max(fprs) - min(fprs)
        full = gaps(np.arange(len(y))); bs = [r for r in (gaps(ix) for ix in boot_ix) if r is not None]
        bt, bf = np.array([r[0] for r in bs]), np.array([r[1] for r in bs])
        eods.append({"model": m, "attribute": attr, "tpr_gap": full[0], "tpr_gap_ci": list(np.percentile(bt, [2.5, 97.5])), "fpr_gap": full[1],
                     "fpr_gap_ci": list(np.percentile(bf, [2.5, 97.5])), "eod": max(full), "screening_flag": max(full) > CFG["evaluation"]["fairness"]["eod_screening_flag"],
                     "n_excluded_missing": int((~ok).sum())})
pd.DataFrame(fair).to_csv(OUT / f"{pref}fairness_subgroups.csv", index=False); pd.DataFrame(eods).to_csv(OUT / f"{pref}fairness_eod.csv", index=False)
R["fairness_eod"] = eods

# ---- R4 SHAP on the frozen XGBoost for test rows
contrib = mx.get_booster().predict(xgbm.DMatrix(test[FEATURES]), pred_contribs=True)[:, :-1]
shap = pd.DataFrame({"feature": FEATURES, "name": [FEATURE_NAMES[f] for f in FEATURES], "mean_abs_shap": np.abs(contrib).mean(0),
                     "mean_shap": contrib.mean(0)}).sort_values("mean_abs_shap", ascending=False)
shap.to_csv(OUT / f"{pref}shap_importance.csv", index=False)
coef = pd.DataFrame({"term": ml.named_steps["impute"].get_feature_names_out(FEATURES), "std_coef": ml.named_steps["lasso"].coef_[0]})
coef["name"] = coef.term.map(lambda t: FEATURE_NAMES.get(t, t)); coef.sort_values("std_coef", key=np.abs, ascending=False).to_csv(OUT / f"{pref}lasso_coefficients.csv", index=False)
R["shap_top5"] = shap.name.head(5).tolist(); R["lasso_nonzero"] = int((coef.std_coef != 0).sum())

# ---- pre-specified sensitivities on the same test period
sens = [{"analysis": "Primary", "n": len(test), "auc_lasso": auc_l, "auc_xgb": auc_x, "delta": delta, "ci": [lo, hi], "decision": branch}]
def paired(name, yy, a, b, g):
    dd, _, _, _ = paired_cluster_bootstrap(yy, a, b, g, B, seed); dl = roc_auc_score(yy, b) - roc_auc_score(yy, a); l2, h2 = np.percentile(dd, [2.5, 97.5])
    sens.append({"analysis": name, "n": len(yy), "auc_lasso": roc_auc_score(yy, a), "auc_xgb": roc_auc_score(yy, b), "delta": dl, "ci": [l2, h2], "decision": decision(dl, l2, h2, margin)})
    log.info(f"{pref}{name}: LASSO {roc_auc_score(yy, a):.4f} XGB {roc_auc_score(yy, b):.4f} Delta {dl:+.4f} [{l2:+.4f}, {h2:+.4f}]")
m1, m2 = fit_both(train, "withdrawn"); paired("S1 Withdrawn-only label", test.withdrawn.to_numpy(), m1.predict_proba(test[FEATURES])[:, 1], m2.predict_proba(test[FEATURES])[:, 1], test.id_student.to_numpy())
m1, m2 = fit_both(dup_tr); paired("S2 duplicates retained", dup_te.at_risk.to_numpy(), m1.predict_proba(dup_te[FEATURES])[:, 1], m2.predict_proba(dup_te[FEATURES])[:, 1], dup_te.id_student.to_numpy())
seen = test.id_student.isin(set(train.id_student)); t2 = test[~seen]
paired("S3 repeat students removed", t2.at_risk.to_numpy(), t2.p_lasso.to_numpy(), t2.p_xgb.to_numpy(), t2.id_student.to_numpy())
trb, teb = pd.concat([train, bx_tr], ignore_index=True), pd.concat([test[train.columns.intersection(bx_te.columns)], bx_te], ignore_index=True)
m1, m2 = fit_both(trb); paired("S4 day-28 unregistrations included", teb.at_risk.to_numpy(), m1.predict_proba(teb[FEATURES])[:, 1], m2.predict_proba(teb[FEATURES])[:, 1], teb.id_student.to_numpy())
m1, m2 = fit_both(train, smote=True); paired("S5 SMOTE instead of class weighting", y, m1.predict_proba(test[FEATURES])[:, 1], m2.predict_proba(test[FEATURES])[:, 1], test.id_student.to_numpy())
_, m2 = fit_both(train, nt=SEL["S7_n_estimators_from_F3"]); paired(f"S7 XGBoost trees set on F3 ({SEL['S7_n_estimators_from_F3']})", y, pl, m2.predict_proba(test[FEATURES])[:, 1], test.id_student.to_numpy())
pd.DataFrame(sens).to_csv(OUT / f"{pref}sensitivities.csv", index=False); R["sensitivities"] = sens
R["n_repeat_student_registrations"] = int(seen.sum())
write_json(R, f"{pref}results.json")
log.info(f"{pref}EVALUATION COMPLETE")
