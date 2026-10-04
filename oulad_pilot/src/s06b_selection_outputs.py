"""Stage 6b: training-only decisions frozen before the sealed evaluation: thresholds, calibration rule,
SHAP fold stability, inner-fold paired comparison (selection evidence only) and inner-fold sensitivities."""
from core import *
import xgboost as xgbm
from sklearn.metrics import roc_auc_score, brier_score_loss
log = get_logger("s06b")
F = CFG["folds"]; X = pd.read_csv(OUT / "features_nonsealed.csv")
L = json.loads((OUT / "lasso_inner_results.json").read_text()); G = json.loads((OUT / "xgb_inner_results.json").read_text())
C, P, NT = L["selected_C"], G["selected_params"], G["final_n_estimators"]
ol = pd.read_csv(OUT / "oof_lasso.csv"); ox = pd.read_csv(OUT / "oof_xgb.csv")
oof = ol.merge(ox[KEY + ["fold", "p_xgb"]], on=KEY + ["fold"])
pr = oof[oof.fold.isin(["F1", "F2"])]
res = {"pooled_oof_n": len(pr), "pooled_oof_students": int(pr.id_student.nunique())}
# thresholds (training partition only)
k = CFG["thresholds"]["top_k_share"]
for m, col in [("lasso", "p_lasso"), ("xgb", "p_xgb")]:
    y, p = pr.y.to_numpy(), pr[col].to_numpy()
    ty = youden_threshold(y, p); tk = float(np.quantile(p, 1 - k))
    def rates(t):
        yh = p >= t; return {"tpr": float((yh & (y == 1)).sum() / (y == 1).sum()), "fpr": float((yh & (y == 0)).sum() / (y == 0).sum()), "flag_rate": float(yh.mean())}
    brier, const = brier_score_loss(y, p), brier_constant(y)
    recal = bool(brier > const)
    res[m] = {"youden_threshold": ty, "youden_on_training_oof": rates(ty), "top20_threshold": tk, "top20_on_training_oof": rates(tk),
              "pooled_oof_auc": float(roc_auc_score(y, p)), "pooled_oof_brier": float(brier), "pooled_constant_brier": float(const),
              "isotonic_recalibration_triggered": recal}
    log.info(f"{m}: Youden t={ty:.4f} (TPR {rates(ty)['tpr']:.3f}, FPR {rates(ty)['fpr']:.3f}); top-20% t={tk:.4f}; pooled OOF Brier {brier:.4f} vs constant {const:.4f} -> isotonic {'TRIGGERED' if recal else 'not triggered'}")
    if recal:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        pd.DataFrame({"x": iso.X_thresholds_, "y": iso.y_thresholds_}).to_csv(OUT / f"isotonic_{m}.csv", index=False)
# inner paired comparison (selection evidence only; NOT a thesis result)
d, a, b, ns = paired_cluster_bootstrap(pr.y.to_numpy(), pr.p_lasso.to_numpy(), pr.p_xgb.to_numpy(), pr.id_student.to_numpy(), CFG["evaluation"]["bootstrap"]["resamples"], CFG["seed"])
delta = float(roc_auc_score(pr.y, pr.p_xgb) - roc_auc_score(pr.y, pr.p_lasso)); lo, hi = np.percentile(d, [2.5, 97.5])
res["inner_paired_comparison"] = {"delta_auc": delta, "ci": [float(lo), float(hi)], "student_clusters": ns, "branch_if_this_were_test": decision(delta, lo, hi, CFG["evaluation"]["practical_margin"]),
                                  "note": "pooled F1+F2 validation rows; selection evidence only; not an estimate of generalisation"}
log.info(f"inner paired Delta-AUC (XGB - LASSO) = {delta:.4f} [{lo:.4f}, {hi:.4f}], {ns:,} students (selection evidence only)")
# SHAP stability across primary folds
s1 = pd.read_csv(OUT / "shap_inner_F1.csv"); s2 = pd.read_csv(OUT / "shap_inner_F2.csv"); kk = CFG["evaluation"]["shap"]["top_k"]
t1, t2 = set(s1.feature[:kk]), set(s2.feature[:kk]); jac = len(t1 & t2) / len(t1 | t2)
res["shap_top5_jaccard_F1_F2"] = {"jaccard": jac, "top5_F1": [FEATURE_NAMES[f] for f in s1.feature[:kk]], "top5_F2": [FEATURE_NAMES[f] for f in s2.feature[:kk]]}
# F1 and F2 both train on P1 with identical settings, so their top-5 agreement is expected by construction.
tr3 = X[X.period.isin(F["F3"]["train"])]; va3 = X[X.period == F["F3"]["validate"]]
m3s = xgb(P, tr3.at_risk.to_numpy(), n_estimators=NT).fit(tr3[FEATURES], tr3.at_risk)
c3 = m3s.get_booster().predict(xgbm.DMatrix(va3[FEATURES]), pred_contribs=True)[:, :-1]
s3 = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(c3).mean(0)}).sort_values("mean_abs_shap", ascending=False)
s3.to_csv(OUT / "shap_inner_F3.csv", index=False)
t3 = set(s3.feature[:kk]); res["shap_top5_jaccard_F2_F3"] = {"jaccard": len(t2 & t3) / len(t2 | t3), "top5_F3": [FEATURE_NAMES[f] for f in s3.feature[:kk]],
    "note": "F1 and F2 share the P1 training set, so F1-F2 agreement is by construction; F2 (P1 model) vs F3 (P1+P2 model) on the same P3 rows is the informative comparison"}
log.info(f"SHAP top-5 Jaccard F2 vs F3 = {res['shap_top5_jaccard_F2_F3']['jaccard']:.3f}; F3 {res['shap_top5_jaccard_F2_F3']['top5_F3']}")
log.info(f"SHAP top-5 Jaccard F1 vs F2 = {jac:.3f}; F1 {res['shap_top5_jaccard_F1_F2']['top5_F1']}; F2 {res['shap_top5_jaccard_F1_F2']['top5_F2']}")
# inner-fold sensitivities with the selected hyper-parameters fixed
def fit_auc(data, label, folds=("F1", "F2"), smote=False, nt=NT):
    out = {}
    for f in folds:
        tr = data[data.period.isin(F[f]["train"])]; va = data[data.period == F[f]["validate"]]; yt = tr[label].to_numpy()
        Xt = tr[FEATURES]
        if smote:
            Xt, yt = smote_resample(Xt, yt, CFG["seed"])
            ml = lasso(C); ml.set_params(lasso__class_weight=None); ml.fit(Xt, yt)
            mx = xgb(P, np.zeros(1), n_estimators=nt); mx.set_params(scale_pos_weight=1.0); mx.fit(Xt, yt)
        else:
            ml = lasso(C).fit(Xt, yt); mx = xgb(P, yt, n_estimators=nt).fit(Xt, yt)
        out[f] = (roc_auc_score(va[label], ml.predict_proba(va[FEATURES])[:, 1]), roc_auc_score(va[label], mx.predict_proba(va[FEATURES])[:, 1]))
    return out
sens = []
def add(name, desc, r):
    for f, (al, ax) in r.items():
        sens.append({"analysis": name, "description": desc, "fold": f, "auc_lasso": al, "auc_xgb": ax, "delta": ax - al})
    log.info(f"SENS {name}: " + "; ".join(f"{f} LASSO {al:.3f} XGB {ax:.3f}" for f, (al, ax) in r.items()))
add("Primary", "at_risk; duplicates collapsed; class weighting", fit_auc(X, "at_risk", ("F1", "F2", "F3")))
add("S1", "secondary label: Withdrawn only", fit_auc(X, "withdrawn"))
add("S2", "exact duplicate VLE rows retained", fit_auc(pd.read_csv(OUT / "features_nonsealed_dup_retained.csv"), "at_risk"))
xb = pd.read_csv(OUT / "features_boundary_extra.csv"); xb = xb[xb.period != CFG["sealed_period"]]
add("S4", f"day-28 unregistrations included ({len(xb)} training-period rows)", fit_auc(pd.concat([X, xb], ignore_index=True), "at_risk"))
add("S5", "SMOTE within training rows instead of class weighting", fit_auc(X, "at_risk", smote=True))
# S7 (tuning-limit check): number of trees set by early stopping on F3, other parameters unchanged
tr3 = X[X.period.isin(F["F3"]["train"])]; va3 = X[X.period == F["F3"]["validate"]]
m3 = xgb(P, tr3.at_risk.to_numpy(), early=True).fit(tr3[FEATURES], tr3.at_risk, eval_set=[(va3[FEATURES], va3.at_risk)], verbose=False)
nt3 = int(m3.best_iteration + 1); res["S7_n_estimators_from_F3"] = nt3
add("S7", f"XGBoost trees set on F3 ({nt3} instead of {NT})", fit_auc(X, "at_risk", nt=nt3))
pd.DataFrame(sens).to_csv(OUT / "inner_sensitivities.csv", index=False)
write_json(res, "selection_outputs.json")
log.info("SELECTION OUTPUTS COMPLETE - sealed period not accessed")
