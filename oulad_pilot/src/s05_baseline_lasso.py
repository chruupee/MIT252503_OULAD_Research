"""Stage 5: LASSO (L1) logistic regression tuned on the primary inner folds F1, F2 (training periods only)."""
from core import *
from sklearn.metrics import roc_auc_score, brier_score_loss
log = get_logger("s05")
X = pd.read_csv(OUT / "features_nonsealed.csv"); F = CFG["folds"]
def fold(k, data=X, label="at_risk"):
    f = F[k]; tr = data[data.period.isin(f["train"])]; va = data[data.period == f["validate"]]
    return tr[FEATURES], tr[label].to_numpy(), va, va[label].to_numpy()
rows = []
for C in CFG["lasso"]["C_grid"]:
    r = {"C": C}
    for k in F:
        Xt, yt, va, yv = fold(k); m = lasso(C).fit(Xt, yt); p = m.predict_proba(va[FEATURES])[:, 1]
        r[f"auc_{k}"] = roc_auc_score(yv, p); r[f"nonzero_{k}"] = int((m.named_steps["lasso"].coef_ != 0).sum())
    r["mean_auc_primary"] = (r["auc_F1"] + r["auc_F2"]) / 2; rows.append(r)
    log.info(f"C={C}: F1={r['auc_F1']:.4f} F2={r['auc_F2']:.4f} (F3 sens {r['auc_F3']:.4f}) mean={r['mean_auc_primary']:.4f}")
grid = pd.DataFrame(rows); grid.to_csv(OUT / "lasso_C_grid_inner_auc.csv", index=False)
best = grid.sort_values(["mean_auc_primary", "C"], ascending=[False, True]).iloc[0]; C = float(best.C)
oof, fold_rows = [], []
for k in F:
    Xt, yt, va, yv = fold(k); m = lasso(C).fit(Xt, yt); p = m.predict_proba(va[FEATURES])[:, 1]
    fold_rows.append({"fold": k, "role": F[k]["role"], "train_n": len(yt), "val_n": len(yv), "val_rate": float(yv.mean()), "auc": roc_auc_score(yv, p),
                      "brier": brier_score_loss(yv, p), "brier_constant": brier_constant(yv), "nonzero": int((m.named_steps["lasso"].coef_ != 0).sum()),
                      "inputs": int(m.named_steps["lasso"].coef_.shape[1])})
    oof.append(va[KEY + ["period"]].assign(fold=k, y=yv, p_lasso=p))
pd.concat(oof).to_csv(OUT / "oof_lasso.csv", index=False)
write_json({"selected_C": C, "folds": fold_rows}, "lasso_inner_results.json")
log.info(f"selected C (training-partition inner folds only) = {C}; " + "; ".join(f"{r['fold']} AUC={r['auc']:.3f} Brier={r['brier']:.3f} (const {r['brier_constant']:.3f})" for r in fold_rows))
