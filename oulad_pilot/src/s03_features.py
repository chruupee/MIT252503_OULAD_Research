"""Stage 3: 28-feature registry for eligible registrations. Sealed-period features are written to a separate file without labels."""
from core import *
log = get_logger("s03")
D = load_all(); el = pd.read_csv(OUT / "registrations_eligibility.csv")
e = el[el.eligible].copy(); sealed = CFG["sealed_period"]
for variant, collapse in [("", True), ("_dup_retained", False)]:
    X, audit = build_features(e, D["info"], D["reg"], D["sv"], D["vle"], D["sa"], D["asm"], collapse_duplicates=collapse)
    X = X.merge(e[KEY + ["period", "at_risk", "withdrawn"]], on=KEY, how="left", validate="1:1")
    ns, s = X[X.period != sealed], X[X.period == sealed].drop(columns=["at_risk", "withdrawn"])
    ns.to_csv(OUT / f"features_nonsealed{variant}.csv", index=False); s.to_csv(OUT / f"features_sealed_nolabels{variant}.csv", index=False)
    log.info(f"features{variant}: non-sealed {ns.shape}, sealed (no labels) {s.shape}; audit={audit}")
    if variant == "":
        write_json({"version": "1.1", "window": CFG["prediction"]["window"], "features": FEATURE_NAMES, "audit": audit,
                    "nullable": ASSESS_NULLABLE, "fairness_only_not_predictors": ["gender", "age_band", "imd_band"]}, "feature_manifest.json")
# boundary sensitivity population (S4): includes the day-28 unregistrations
eb = eligibility(D["info"], D["reg"], CFG["eligibility"]["sensitivity_boundary_unregistration_strictly_after"])
extra = eb[eb.eligible & ~eb.set_index(KEY).index.isin(e.set_index(KEY).index)]
Xb, _ = build_features(extra, D["info"], D["reg"], D["sv"], D["vle"], D["sa"], D["asm"])
Xb = Xb.merge(extra[KEY + ["period", "at_risk", "withdrawn"]], on=KEY)
Xb.loc[Xb.period == sealed, ["at_risk", "withdrawn"]] = np.nan
Xb.to_csv(OUT / "features_boundary_extra.csv", index=False)
log.info(f"boundary-sensitivity extra registrations (unregistered on day 28): {len(Xb)}")
