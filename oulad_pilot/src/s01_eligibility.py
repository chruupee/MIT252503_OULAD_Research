"""Stage 1: 1:1 join, rules E1-E3, labels, already-left log; sealed-period labels blanked in written files."""
from core import *
log = get_logger("s01")
info, reg = read("studentInfo"), read("studentRegistration")
m = eligibility(info, reg)
sealed = CFG["periods"][CFG["sealed_period"]]
rows = []
for per, code in CFG["periods"].items():
    d = m[m.code_presentation == code]; e = d[d.eligible]
    r = {"period": per, "presentation": code, "registrations": len(d), "already_left_by_t0": int(d.already_left.sum()),
         "left_before_day0": int((d.date_unregistration < 0).sum()), "no_registration_date": int(d.date_registration.isna().sum()),
         "registered_on_or_after_t0": int((d.date_registration >= CFG["prediction"]["t0_day"]).sum()),
         "unregistered_on_day_28": int((d.date_unregistration == 28).sum()), "eligible": len(e)}
    if code != sealed:
        r.update(eligible_at_risk_rate=round(e.at_risk.mean(), 4), eligible_withdrawn_rate=round(e.withdrawn.mean(), 4),
                 all_reg_withdrawn_rate=round(d.withdrawn.mean(), 4),
                 already_left_withdrawn=f"{int((d.already_left & (d.final_result=='Withdrawn')).sum())}/{int(d.already_left.sum())}")
    rows.append(r); log.info(f"{per} {code}: registrations={len(d):,} already_left={r['already_left_by_t0']:,} eligible={len(e):,}")
tot = {"registrations": len(m), "already_left_by_t0": int(m.already_left.sum()), "eligible": int(m.eligible.sum()),
       "any_reason_excluded": int((~m.eligible).sum()), "left_before_day0": int((m.date_unregistration < 0).sum()),
       "unregistered_on_day_28": int((m.date_unregistration == 28).sum()),
       "eligible_if_day28_unregistrations_included": int(eligibility(info, reg, CFG['eligibility']['sensitivity_boundary_unregistration_strictly_after']).eligible.sum())}
tr = m[m.eligible & (m.code_presentation != sealed)]
tot["training_eligible_at_risk_rate"] = round(tr.at_risk.mean(), 4); tot["training_eligible_withdrawn_rate"] = round(tr.withdrawn.mean(), 4)
write_json({"by_period": rows, "total": tot}, "eligibility_counts.json")
out = m[KEY + ["period", "date_registration", "date_unregistration", "E1", "E2", "E3", "eligible", "already_left", "final_result", "at_risk", "withdrawn"]].copy()
out.loc[out.code_presentation == sealed, ["final_result", "at_risk", "withdrawn"]] = np.nan   # blank sealed labels
out.to_csv(OUT / "registrations_eligibility.csv", index=False)
m[m.already_left].assign(**{"final_result": lambda d: np.where(d.code_presentation == sealed, "SEALED", d.final_result)})[
    KEY + ["period", "date_unregistration", "final_result"]].to_csv(OUT / "already_left_log.csv", index=False)
log.info(f"ELIGIBLE: {tot['eligible']:,} of {tot['registrations']:,} ({tot['eligible']/tot['registrations']:.1%}) | left by t0: {tot['already_left_by_t0']:,} | day-28 unregistrations: {tot['unregistered_on_day_28']}")
