"""Stage 9b (run on the student's workstation, which holds the raw KDD Cup files):
nine shared features f01-f09 for KDD Cup 2015 enrolments over course days 0-27, using the same
definitions and null conventions as the OULAD registry (Table 6.10). Usage:
  python s09b_kdd_shared_features.py --raw path/to/kddcup2015  (expects train/log_train.csv,
  train/enrollment_train.csv, train/truth_train.csv, date.csv)"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

def shared_features(log: pd.DataFrame, enrol: pd.DataFrame, dates: pd.DataFrame, lo=0, hi=27) -> pd.DataFrame:
    d = log.merge(enrol[["enrollment_id", "course_id"]], on="enrollment_id").merge(dates, on="course_id")
    d["day"] = (pd.to_datetime(d["time"]).dt.normalize() - pd.to_datetime(d["from"]).dt.normalize()).dt.days
    w = d[(d["day"] >= lo) & (d["day"] <= hi)]
    daily = w.groupby(["enrollment_id", "day"]).size().rename("n").reset_index(); g = daily.groupby("enrollment_id")
    f = pd.DataFrame({"f01": g["n"].sum(), "f02": g["day"].nunique(), "first": g["day"].min(), "last": g["day"].max()})
    nd = hi - lo + 1; s2 = daily.assign(q=daily.n.astype(float) ** 2).groupby("enrollment_id").q.sum(); mu = f.f01 / nd
    f["f07"] = np.sqrt((s2 / nd - mu ** 2).clip(lower=0)) / mu.replace(0, np.nan)
    h1 = daily[daily.day <= 13].groupby("enrollment_id").n.sum().reindex(f.index, fill_value=0)
    h2 = daily[daily.day >= 14].groupby("enrollment_id").n.sum().reindex(f.index, fill_value=0)
    f["f06"] = (h2 + 1.0) / (h1 + 1.0)
    f["f08"] = w[w.event == "discussion"].groupby("enrollment_id").size().reindex(f.index, fill_value=0)
    f["f09"] = w[w.event == "problem"].groupby("enrollment_id").size().reindex(f.index, fill_value=0)
    out = enrol[["enrollment_id"]].merge(f.reset_index(), on="enrollment_id", how="left")
    for c in ["f01", "f02", "f08", "f09"]: out[c] = out[c].fillna(0)
    out["f03"] = np.where(out.f02 > 0, out.f01 / out.f02.replace(0, np.nan), 0.0)
    out["f04"] = np.where(out["last"].notna(), hi - out["last"], hi + 1)
    out["f05"] = np.where(out["first"].notna(), out["first"], hi + 1)
    out["f06"] = out.f06.fillna(1.0); out["f07"] = out.f07.fillna(0.0)
    return out[["enrollment_id"] + [f"f0{i}" for i in range(1, 10)]]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True); a = ap.parse_args(); R = Path(a.raw)
    enrol = pd.read_csv(R / "train" / "enrollment_train.csv"); dates = pd.read_csv(R / "date.csv")
    truth = pd.read_csv(R / "train" / "truth_train.csv", header=None, names=["enrollment_id", "dropout"])
    log = pd.read_csv(R / "train" / "log_train.csv", usecols=["enrollment_id", "time", "event"])
    out = shared_features(log, enrol, dates).merge(truth, on="enrollment_id", how="left")
    Path("outputs").mkdir(exist_ok=True); out.to_csv("outputs/kdd_shared_features_days0_27.csv", index=False)
    print(out.shape, out.dropout.mean())
