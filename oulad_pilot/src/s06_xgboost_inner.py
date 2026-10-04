"""Stage 6: XGBoost tuned with Optuna (TPE, 60 trials) on F1 and F2; early stopping on the inner validation fold."""
from core import *
import optuna
from sklearn.metrics import roc_auc_score, brier_score_loss
log = get_logger("s06"); optuna.logging.set_verbosity(optuna.logging.WARNING)
X = pd.read_csv(OUT / "features_nonsealed.csv"); F = CFG["folds"]; S = CFG["xgboost"]["search_space"]
def fold(k, data=X, label="at_risk"):
    f = F[k]; tr = data[data.period.isin(f["train"])]; va = data[data.period == f["validate"]]
    return tr[FEATURES], tr[label].to_numpy(), va, va[label].to_numpy()
def params_from(t):
    return {"max_depth": t.suggest_int("max_depth", *S["max_depth"]), "learning_rate": t.suggest_float("learning_rate", *S["learning_rate"], log=True),
            "min_child_weight": t.suggest_float("min_child_weight", *S["min_child_weight"]), "subsample": t.suggest_float("subsample", *S["subsample"]),
            "colsample_bytree": t.suggest_float("colsample_bytree", *S["colsample_bytree"]), "reg_lambda": t.suggest_float("reg_lambda", *S["reg_lambda"], log=True),
            "gamma": t.suggest_float("gamma", *S["gamma"])}
trials = []
def objective(t):
    p = params_from(t); aucs, its = {}, {}
    for k in ["F1", "F2"]:
        Xt, yt, va, yv = fold(k)
        m = xgb(p, yt, early=True).fit(Xt, yt, eval_set=[(va[FEATURES], yv)], verbose=False)
        aucs[k] = roc_auc_score(yv, m.predict_proba(va[FEATURES], iteration_range=(0, m.best_iteration + 1))[:, 1]); its[k] = m.best_iteration + 1
    trials.append({"trial": t.number, **p, "auc_F1": aucs["F1"], "auc_F2": aucs["F2"], "best_iter_F1": its["F1"], "best_iter_F2": its["F2"],
                   "mean_auc": (aucs["F1"] + aucs["F2"]) / 2})
    log.info(f"trial {t.number:02d}: F1={aucs['F1']:.4f} F2={aucs['F2']:.4f} iters=({its['F1']},{its['F2']})")
    return (aucs["F1"] + aucs["F2"]) / 2
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=CFG["seed"]))
study.optimize(objective, n_trials=CFG["xgboost"]["optuna_trials"])
tr = pd.DataFrame(trials); tr.to_csv(OUT / "xgb_optuna_trials.csv", index=False)
b = tr.loc[tr.mean_auc.idxmax()]
best = {k: (int(b[k]) if k == "max_depth" else float(b[k])) for k in ["max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda", "gamma"]}
n_final = int(round((b.best_iter_F1 + b.best_iter_F2) / 2))
log.info(f"SELECTED XGBoost (inner folds only): {best}; final n_estimators={n_final}; mean inner AUC={b.mean_auc:.4f}")
oof, rows = [], []
for k in F:
    Xt, yt, va, yv = fold(k)
    m = xgb(best, yt, n_estimators=n_final).fit(Xt, yt); p = m.predict_proba(va[FEATURES])[:, 1]
    rows.append({"fold": k, "role": F[k]["role"], "auc": roc_auc_score(yv, p), "brier": brier_score_loss(yv, p), "brier_constant": brier_constant(yv)})
    oof.append(va[KEY + ["period"]].assign(fold=k, y=yv, p_xgb=p))
    if k in ("F1", "F2"):
        contrib = m.get_booster().predict(__import__("xgboost").DMatrix(va[FEATURES]), pred_contribs=True)[:, :-1]
        pd.DataFrame({"feature": FEATURES, "name": [FEATURE_NAMES[f] for f in FEATURES], "mean_abs_shap": np.abs(contrib).mean(0)}).sort_values(
            "mean_abs_shap", ascending=False).to_csv(OUT / f"shap_inner_{k}.csv", index=False)
pd.concat(oof).to_csv(OUT / "oof_xgb.csv", index=False)
write_json({"selected_params": best, "final_n_estimators": n_final, "best_trial": int(b.trial), "mean_inner_auc_with_early_stopping": float(b.mean_auc),
            "folds_fixed_n_estimators": rows}, "xgb_inner_results.json")
log.info("; ".join(f"{r['fold']} AUC={r['auc']:.3f} Brier={r['brier']:.3f} (const {r['brier_constant']:.3f})" for r in rows))
