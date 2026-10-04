"""Shared library for the MIT252503 OULAD pipeline (protocol v1.1).

All protocol constants come from config/config.yaml. Functions here are used by the
stage scripts s00-s08 and by the window-boundary tests, so that the tests exercise the
same code that produces the thesis numbers.
"""
from __future__ import annotations

import hashlib, json, logging, platform, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
CFG_PATH = ROOT / "config" / "config.yaml"
CFG = yaml.safe_load(CFG_PATH.read_text())
KEY = ["code_module", "code_presentation", "id_student"]
FEATURES = [f"f{i:02d}" for i in range(1, 29)]
FEATURE_NAMES = {
    "f01": "total_clicks", "f02": "active_days", "f03": "mean_clicks_per_active_day", "f04": "days_since_last_click",
    "f05": "days_to_first_click", "f06": "weekly_decay_ratio", "f07": "cv_daily_clicks", "f08": "forum_clicks",
    "f09": "quiz_clicks", "f10": "unique_resources", "f11": "unique_activity_types", "f12": "active_weeks",
    "f13": "oucontent_clicks", "f14": "resource_clicks", "f15": "homepage_clicks", "f16": "pre_course_clicks",
    "f17": "log_growth_w2_w1", "f18": "log_growth_w3_w2", "f19": "log_growth_w4_w3", "f20": "n_assessments_submitted",
    "f21": "submission_rate", "f22": "mean_score", "f23": "days_to_first_submission", "f24": "prop_submitted_on_time",
    "f25": "highest_education", "f26": "num_of_prev_attempts", "f27": "studied_credits", "f28": "registration_earliness"}
ASSESS_NULLABLE = ["f21", "f22", "f23", "f24"]
PERIOD_OF = {v: k for k, v in CFG["periods"].items()}


# ------------------------------------------------------------------ logging / provenance
def get_logger(stage: str):
    log = logging.getLogger(stage)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        for h in (logging.FileHandler(OUT / "run_log.txt", mode="a"), logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt); log.addHandler(h)
    return log


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def env_snapshot() -> dict:
    import sklearn, xgboost, scipy
    d = {"python": platform.python_version(), "platform": platform.platform(), "pandas": pd.__version__,
         "numpy": np.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
         "scipy": scipy.__version__, "config_sha256": sha256(CFG_PATH), "seed": CFG["seed"]}
    try:
        import optuna; d["optuna"] = optuna.__version__
    except ImportError:
        pass
    return d


def write_json(obj, name):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# ------------------------------------------------------------------ loading
def read(table: str, **kw) -> pd.DataFrame:
    return pd.read_csv(ROOT / CFG["data"]["raw_dir"] / f"{table}.csv", na_values=CFG["data"]["na_values"], **kw)


# ------------------------------------------------------------------ eligibility and labels
def eligibility(info: pd.DataFrame, reg: pd.DataFrame, unreg_after: int | None = None) -> pd.DataFrame:
    """Rules E1-E3 (Section 6.5.2). Returns all registrations with flags and reasons."""
    t0 = CFG["prediction"]["t0_day"]
    unreg_after = CFG["eligibility"]["unregistration_strictly_after"] if unreg_after is None else unreg_after
    m = info.merge(reg, on=KEY, how="outer", indicator=True, validate="1:1")
    m["E1"] = m["_merge"] == "both"
    m["E2"] = m["date_registration"].notna() & (m["date_registration"] < t0)
    m["E3"] = m["date_unregistration"].isna() | (m["date_unregistration"] > unreg_after)
    m["eligible"] = m["E1"] & m["E2"] & m["E3"]
    m["already_left"] = m["date_unregistration"].notna() & (m["date_unregistration"] <= unreg_after)
    m["period"] = m["code_presentation"].map(PERIOD_OF)
    pos1 = CFG["labels"]["primary"]["positive"]; pos2 = CFG["labels"]["secondary"]["positive"]
    m["at_risk"] = m["final_result"].isin(pos1).astype(int)
    m["withdrawn"] = m["final_result"].isin(pos2).astype(int)
    return m.drop(columns="_merge")


# ------------------------------------------------------------------ features (28)
def _log_ratio(a, b):
    return np.log((b + 1.0) / (a + 1.0))


def build_features(elig: pd.DataFrame, info, reg, sv, vle, sa, asm, collapse_duplicates=None) -> tuple[pd.DataFrame, dict]:
    """Computes f01-f28 for the registrations in `elig` using only records dated <= window end."""
    lo, hi = CFG["prediction"]["window"]; plo, phi = CFG["prediction"]["pre_course_window"]
    collapse = CFG["vle"]["collapse_exact_duplicates"] if collapse_duplicates is None else collapse_duplicates
    keys = elig[KEY].drop_duplicates()
    audit = {}
    v = sv[(sv["date"] >= plo) & (sv["date"] <= hi)].merge(keys, on=KEY, how="inner")
    audit["vle_rows_in_range"] = int(len(v)); audit["vle_exact_duplicates_in_range"] = int(v.duplicated().sum())
    if collapse:
        v = v.drop_duplicates()
    v = v.merge(vle[["id_site", "code_module", "code_presentation", "activity_type"]],
                on=["id_site", "code_module", "code_presentation"], how="left")
    pre = v[v["date"] < lo]; w = v[v["date"] >= lo]
    audit["max_vle_date_used_in_window_features"] = int(w["date"].max()) if len(w) else None
    audit["min_vle_date_used"] = int(v["date"].min()) if len(v) else None

    daily = w.groupby(KEY + ["date"], as_index=False)["sum_click"].sum()
    g = daily.groupby(KEY)
    f = pd.DataFrame({"f01": g["sum_click"].sum(), "f02": g["date"].nunique(),
                      "_first": g["date"].min(), "_last": g["date"].max()})
    ndays = hi - lo + 1
    s2 = daily.assign(sq=daily["sum_click"].astype(float) ** 2).groupby(KEY)["sq"].sum()
    mean = f["f01"] / ndays
    sd = np.sqrt((s2 / ndays - mean ** 2).clip(lower=0))
    f["f07"] = sd / mean.replace(0, np.nan)
    h1 = daily[daily["date"] <= 13].groupby(KEY)["sum_click"].sum().reindex(f.index, fill_value=0)
    h2 = daily[daily["date"] >= 14].groupby(KEY)["sum_click"].sum().reindex(f.index, fill_value=0)
    f["f06"] = (h2 + 1.0) / (h1 + 1.0)
    wk = pd.Series(0, index=daily.index)
    for i, (name, (a, b)) in enumerate(CFG["prediction"]["weeks"].items()):
        wk[(daily["date"] >= a) & (daily["date"] <= b)] = i + 1
    weekly = daily.assign(week=wk).pivot_table(index=KEY, columns="week", values="sum_click", aggfunc="sum", fill_value=0)
    weekly = weekly.reindex(columns=[1, 2, 3, 4], fill_value=0)
    f["f12"] = (weekly > 0).sum(axis=1)
    f["f17"] = _log_ratio(weekly[1], weekly[2]); f["f18"] = _log_ratio(weekly[2], weekly[3]); f["f19"] = _log_ratio(weekly[3], weekly[4])
    bt = w.pivot_table(index=KEY, columns="activity_type", values="sum_click", aggfunc="sum", fill_value=0)
    for fid, at in [("f08", "forumng"), ("f09", "quiz"), ("f13", "oucontent"), ("f14", "resource"), ("f15", "homepage")]:
        f[fid] = bt[at] if at in bt.columns else 0
    f["f10"] = w.groupby(KEY)["id_site"].nunique()
    f["f11"] = (bt > 0).sum(axis=1)
    f["f16_tmp"] = 0
    f = f.reset_index()
    out = keys.merge(f, on=KEY, how="left")
    out = out.merge(pre.groupby(KEY)["sum_click"].sum().rename("f16").reset_index(), on=KEY, how="left")
    for c in ["f01", "f02", "f08", "f09", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19"]:
        out[c] = out[c].fillna(0)
    out["f03"] = np.where(out["f02"] > 0, out["f01"] / out["f02"].replace(0, np.nan), 0.0)
    out["f04"] = np.where(out["_last"].notna(), hi - out["_last"], hi + 1)
    out["f05"] = np.where(out["_first"].notna(), out["_first"], hi + 1)
    out["f06"] = out["f06"].fillna(1.0)
    out["f07"] = out["f07"].fillna(0.0)
    out = out.drop(columns=["_first", "_last", "f16_tmp"])

    # assessments: submitted within window, not banked
    s = sa.merge(asm[["id_assessment", "code_module", "code_presentation", "date", "assessment_type"]],
                 on="id_assessment", how="left", validate="m:1")
    s = s[(s["is_banked"] == 0) & (s["date_submitted"] <= hi)].merge(keys, on=KEY, how="inner")
    audit["max_submission_date_used"] = int(s["date_submitted"].max()) if len(s) else None
    s["on_time"] = (s["date_submitted"] <= s["date"]).astype(float)
    gs = s.groupby(KEY)
    a = pd.DataFrame({"f20": gs.size(), "f22": gs["score"].mean(), "f23": gs["date_submitted"].min(),
                      "f24": gs["on_time"].mean()}).reset_index()
    due = asm[(asm["date"] <= hi) & (asm["assessment_type"] != "Exam")].groupby(["code_module", "code_presentation"]).size().rename("n_due").reset_index()
    # f21 (v1.1 correction): share of assessments DUE on or before day 27 that were submitted by day 27
    due_ids = asm[(asm["date"] <= hi) & (asm["assessment_type"] != "Exam")][["id_assessment"]]
    done_due = s.merge(due_ids, on="id_assessment").groupby(KEY)["id_assessment"].nunique().rename("n_due_done").reset_index()
    out = out.merge(a, on=KEY, how="left").merge(due, on=["code_module", "code_presentation"], how="left")
    out = out.merge(done_due, on=KEY, how="left")
    out["f20"] = out["f20"].fillna(0)
    out["f21"] = np.where(out["n_due"].fillna(0) > 0, out["n_due_done"].fillna(0) / out["n_due"], np.nan)
    out = out.drop(columns="n_due_done")
    out.loc[out["f20"] == 0, ["f22", "f23", "f24"]] = np.nan
    out = out.drop(columns="n_due")

    edu = {"No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2, "HE Qualification": 3, "Post Graduate Qualification": 4}
    b = keys.merge(info[KEY + ["highest_education", "num_of_prev_attempts", "studied_credits"]], on=KEY, how="left")
    b = b.merge(reg[KEY + ["date_registration"]], on=KEY, how="left")
    b["f25"] = b["highest_education"].map(edu); b["f26"] = b["num_of_prev_attempts"]; b["f27"] = b["studied_credits"]
    b["f28"] = -b["date_registration"]
    out = out.merge(b[KEY + ["f25", "f26", "f27", "f28"]], on=KEY, how="left", validate="1:1")
    return out[KEY + FEATURES], audit


def load_all():
    return dict(info=read("studentInfo"), reg=read("studentRegistration"), courses=read("courses"),
                sv=read("studentVle"), vle=read("vle"), sa=read("studentAssessment"), asm=read("assessments"))


# ------------------------------------------------------------------ models
def lasso(C):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    L = CFG["lasso"]
    return Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler()),
                     ("lasso", LogisticRegression(l1_ratio=1.0, solver=L["solver"], C=C,
                                                  class_weight=L["class_weight"], max_iter=5000, random_state=CFG["seed"]))])


def xgb(params, y_train, n_estimators=None, early=False):
    from xgboost import XGBClassifier
    spw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
    kw = dict(params); kw.pop("n_estimators", None)
    return XGBClassifier(n_estimators=n_estimators or CFG["xgboost"]["max_estimators"], scale_pos_weight=spw,
                         tree_method="hist", eval_metric="auc", random_state=CFG["seed"], n_jobs=4, verbosity=0,
                         early_stopping_rounds=CFG["xgboost"]["early_stopping_rounds"] if early else None, **kw)


def smote_resample(X, y, seed):
    """SMOTE inside training rows only (sensitivity S5). Median-imputes first (fitted on the same rows)."""
    from sklearn.neighbors import NearestNeighbors
    Xf = X.copy(); med = Xf.median(); Xf = Xf.fillna(med)
    rng = np.random.default_rng(seed)
    Xp = Xf[y == 1].to_numpy(); n_new = int((y == 0).sum() - (y == 1).sum())
    if n_new <= 0:
        return Xf, y
    mu, sd = Xp.mean(0), Xp.std(0) + 1e-9
    nn = NearestNeighbors(n_neighbors=6).fit((Xp - mu) / sd)
    idx = rng.integers(0, len(Xp), n_new)
    nbr = nn.kneighbors((Xp[idx] - mu) / sd, return_distance=False)[:, 1:]
    pick = nbr[np.arange(n_new), rng.integers(0, 5, n_new)]
    lam = rng.random((n_new, 1))
    syn = Xp[idx] + lam * (Xp[pick] - Xp[idx])
    Xs = pd.concat([Xf, pd.DataFrame(syn, columns=Xf.columns)], ignore_index=True)
    ys = np.concatenate([y, np.ones(n_new, dtype=int)])
    return Xs, ys


# ------------------------------------------------------------------ metrics and decision rule
def youden_threshold(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.argmax(tpr - fpr)); return float(thr[i])


def brier_constant(y):
    b = float(np.mean(y)); return b * (1 - b)


def decision(delta, lo, hi, margin):
    if hi < 0:
        return "BASELINE_BETTER"
    if lo > 0 and delta >= margin:
        return "MEANINGFUL_IMPROVEMENT"
    if lo > 0:
        return "DISTINGUISHABLE_BELOW_MARGIN"
    return "NOT_DISTINGUISHABLE_PREFER_LASSO"


def paired_cluster_bootstrap(y, p_a, p_b, groups, B, seed, extra=None):
    """Student-clustered paired bootstrap of AUC(b) - AUC(a). Same resample for both models."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(pd.Series(groups))
    order = np.argsort(codes, kind="stable"); starts = np.searchsorted(codes[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(order))
    deltas, aa, bb = [], [], []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        ix = np.concatenate([order[starts[k]:ends[k]] for k in pick])
        yb = y[ix]
        if yb.min() == yb.max():
            continue
        a = roc_auc_score(yb, p_a[ix]); b = roc_auc_score(yb, p_b[ix])
        aa.append(a); bb.append(b); deltas.append(b - a)
    return np.array(deltas), np.array(aa), np.array(bb), len(uniq)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)
