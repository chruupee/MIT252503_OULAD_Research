"""Stage 4: join, range, null, leakage, chronology and feature-window boundary checks; 10-registration raw recomputation."""
from core import *
log = get_logger("s04")
D = load_all(); el = pd.read_csv(OUT / "registrations_eligibility.csv")
X = pd.read_csv(OUT / "features_nonsealed.csv"); XS = pd.read_csv(OUT / "features_sealed_nolabels.csv")
folds = json.loads((OUT / "fold_definition.json").read_text())["folds"]; man = json.loads((OUT / "feature_manifest.json").read_text())
lo, hi = CFG["prediction"]["window"]; sealed = CFG["sealed_period"]; t0 = CFG["prediction"]["t0_day"]
checks = []
def chk(cid, name, ok, detail=""):
    checks.append({"id": cid, "check": name, "result": "PASS" if ok else "FAIL", "detail": detail}); log.info(f"{cid} {'PASS' if ok else 'FAIL'}: {name} {detail}")
info, reg = D["info"], D["reg"]
chk("T01", "Registration key unique", not info[KEY].duplicated().any(), f"n={len(info):,}")
j = info.merge(reg, on=KEY, how="outer", indicator=True)
chk("T02", "studentInfo-studentRegistration join is 1:1 with no orphans", (j._merge == "both").all() and len(j) == len(info))
re = (el.date_registration.notna() & (el.date_registration < t0)) & (el.date_unregistration.isna() | (el.date_unregistration > CFG["eligibility"]["unregistration_strictly_after"]))
chk("T03", "Eligibility flag reproduces E2 and E3 by independent recomputation", bool((re == el.eligible).all()))
mod = el[el.eligible]
chk("T04", "No modelling row has date_unregistration on or before day 28", not (mod.date_unregistration <= CFG["eligibility"]["unregistration_strictly_after"]).any())
chk("T05", "Sealed period absent from the modelling file", not (X.period == sealed).any())
chk("T06", "Sealed labels blank in written tables", el[el.period == sealed][["at_risk", "withdrawn", "final_result"]].isna().all().all() and "at_risk" not in XS.columns)
lab = info.merge(X[KEY + ["at_risk"]], on=KEY)
chk("T07", "Label rule: at_risk = 1 iff Fail or Withdrawn", bool((lab.at_risk == lab.final_result.isin(["Fail", "Withdrawn"]).astype(int)).all()))
chk("T08", "Exactly 28 features", all(f in X.columns for f in FEATURES) and len([c for c in X.columns if c.startswith("f")]) == 28)
nn = [f for f in FEATURES if X[f].isna().any()]
chk("T09", "Missing values only in the four declared assessment features", set(nn) <= set(ASSESS_NULLABLE), f"nullable present: {nn}")
rng_ok = (X.f02.between(0, hi - lo + 1).all() and X.f12.between(0, 4).all() and X.f04.between(0, hi + 1).all() and X.f05.between(0, hi + 1).all()
          and X.f24.dropna().between(0, 1).all() and X.f21.dropna().between(0, 1).all() and (X[["f01", "f08", "f09", "f10", "f13", "f14", "f15", "f16", "f20"]] >= 0).all().all())
chk("T10", "Ranges valid (f02 0-28 active days within window 0-27, f12 0-4, f04/f05 0-28, f21/f24 in 0-1, counts >= 0)", bool(rng_ok),
    f"f21 max = {X.f21.max():.2f} (v1.1 definition: due-and-submitted / due)")
z = X[X.f01 == 0]
chk("T11", "Zero-activity conventions hold", bool((z.f02 == 0).all() and (z.f04 == hi + 1).all() and (z.f05 == hi + 1).all() and (z.f06 == 1).all() and (z.f07 == 0).all()),
    f"{len(z):,} rows ({len(z)/len(X):.1%}) had no clicks in the window")
# T12 raw recomputation with separate code
sv = D["sv"]; sa = D["sa"].merge(D["asm"], on="id_assessment")
r = np.random.default_rng(CFG["seed"]); spot = []
for i in r.choice(len(X), 10, replace=False):
    k = X.iloc[i]
    rows = sv[(sv.code_module == k.code_module) & (sv.code_presentation == k.code_presentation) & (sv.id_student == k.id_student)].drop_duplicates()
    w = rows[(rows.date >= lo) & (rows.date <= hi)]
    s = sa[(sa.code_module == k.code_module) & (sa.code_presentation == k.code_presentation) & (sa.id_student == k.id_student) & (sa.is_banked == 0) & (sa.date_submitted <= hi)]
    rec = {"key": f"{k.code_module}/{k.code_presentation}/{k.id_student}", "f01_raw": int(w.sum_click.sum()), "f01": int(k.f01),
           "f02_raw": int(w.date.nunique()), "f02": int(k.f02), "f20_raw": len(s), "f20": int(k.f20)}
    rec["match"] = rec["f01_raw"] == rec["f01"] and rec["f02_raw"] == rec["f02"] and rec["f20_raw"] == rec["f20"]; spot.append(rec)
pd.DataFrame(spot).to_csv(OUT / "spot_check_10_registrations.csv", index=False)
chk("T12", "Ten random registrations recomputed from raw rows match pipeline features", all(x["match"] for x in spot), f"{sum(x['match'] for x in spot)}/10")
man_s = pd.read_csv(OUT / "split_manifest.csv")
chk("T13", "No presentation in two partitions", not man_s.duplicated(["code_module", "code_presentation"]).any())
order = {p: i for i, p in enumerate(CFG["periods"])}
for k, f in folds.items():
    if k == "FINAL": continue
    chk(f"T14-{k}", f"Inner fold {k} strictly forward and sealed-free", f["strictly_forward"] and not f["sealed_in_fold"])
chk("T15", "Final split strictly forward (P1-P3 -> P4)", folds["FINAL"]["strictly_forward"])
# ---- new window-boundary tests (Rubric Priority 1, item 2)
au = man["audit"]
chk("T16", "No VLE record dated after day 27 enters any window feature; pre-course records only from -25..-1",
    au["max_vle_date_used_in_window_features"] <= hi and au["min_vle_date_used"] >= CFG["prediction"]["pre_course_window"][0], f"max={au['max_vle_date_used_in_window_features']}, min={au['min_vle_date_used']}")
chk("T17", "No submission dated after day 27 enters any feature", au["max_submission_date_used"] <= hi, f"max={au['max_submission_date_used']}")
# T18 injection test: add synthetic rows dated 28 and 30 (and submissions dated 28); features must not change; a row dated 27 must change f01
samp = X.sample(25, random_state=CFG["seed"])[KEY]
e25 = el.merge(samp, on=KEY)
sv_s = sv.merge(samp, on=KEY); sa_s = D["sa"].merge(samp[["id_student"]].drop_duplicates(), on="id_student")
base, _ = build_features(e25, info, reg, sv_s, D["vle"], sa_s, D["asm"])
site = D["vle"].groupby(["code_module", "code_presentation"]).id_site.first().reset_index()
inj = samp.merge(site, on=["code_module", "code_presentation"])
late = pd.concat([inj.assign(date=28, sum_click=50), inj.assign(date=30, sum_click=50)])[sv.columns]
asm_any = D["asm"].groupby(["code_module", "code_presentation"]).id_assessment.first().reset_index()
sub_late = samp.merge(asm_any, on=["code_module", "code_presentation"]).assign(date_submitted=28, is_banked=0, score=100.0)[D["sa"].columns]
after, _ = build_features(e25, info, reg, pd.concat([sv_s, late]), D["vle"], pd.concat([sa_s, sub_late]), D["asm"])
same = np.allclose(base[FEATURES].fillna(-999).to_numpy(float), after[FEATURES].fillna(-999).to_numpy(float))
early = inj.assign(date=27, sum_click=50)[sv.columns]
moved, _ = build_features(e25, info, reg, pd.concat([sv_s, early]), D["vle"], sa_s, D["asm"])
changed = bool((moved.f01.to_numpy() == base.f01.to_numpy() + 50).all() and (moved.f04 == 0).all())
chk("T18", "Injection test: records dated 28 or 30 leave all 28 features unchanged; a record dated 27 changes f01 and sets f04 = 0", same and changed)
chk("T19", "Sealed-period features computed with identical code and window (no labels)", set(FEATURES) <= set(XS.columns) and len(XS) == folds["FINAL"]["validate_eligible"], f"n={len(XS):,}")
n_pass = sum(c["result"] == "PASS" for c in checks)
write_json({"checks": checks, "passed": n_pass, "total": len(checks)}, "feature_checks.json")
log.info(f"SUMMARY: {n_pass}/{len(checks)} checks passed")
if n_pass != len(checks): raise SystemExit(1)
