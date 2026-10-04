"""Stage 2: presentation chronology, fold feasibility, label-latency flags."""
from core import *
log = get_logger("s02")
courses = read("courses"); el = pd.read_csv(OUT / "registrations_eligibility.csv")
order = {p: i for i, p in enumerate(CFG["periods"])}
start_month = {"P1": 0, "P2": 8, "P3": 12, "P4": 20}       # months after Feb 2013 (B ~ Feb, J ~ Oct)
c = courses.assign(period=courses.code_presentation.map(PERIOD_OF))
c["start_m"] = c.period.map(start_month); c["end_m"] = c.start_m + c.module_presentation_length / 30.44
n = el.groupby(["code_module", "code_presentation"]).agg(registrations=("id_student", "size"), eligible=("eligible", "sum")).reset_index()
man = c.merge(n, on=["code_module", "code_presentation"]).sort_values(["start_m", "code_module"]).reset_index(drop=True)
man.insert(0, "id", range(1, len(man) + 1))
man["role"] = np.where(man.period == CFG["sealed_period"], "TEST (sealed)", "train")
man[["id", "code_module", "code_presentation", "period", "module_presentation_length", "registrations", "eligible", "role"]].to_csv(OUT / "split_manifest.csv", index=False)
folds = {}
allf = {**CFG["folds"], "FINAL": {"train": CFG["final"]["train"], "validate": CFG["final"]["test"], "role": "final"}}
for k, f in allf.items():
    tr, va = f["train"], f["validate"]
    forward = max(order[p] for p in tr) < order[va]
    t0_m = start_month[va] + 28 / 30.44
    latest_end = man[man.period.isin(tr)].end_m.max()
    safe = bool(latest_end <= t0_m)
    trn = int(el[el.period.isin(tr) & el.eligible].shape[0]); van = int(el[(el.period == va) & el.eligible].shape[0])
    new_mods = sorted(set(man[man.period == va].code_module) - set(man[man.period.isin(tr)].code_module))
    folds[k] = {"train": tr, "validate": va, "role": f["role"], "train_eligible": trn, "validate_eligible": van,
                "strictly_forward": forward, "label_latency_safe": safe, "latest_train_end_month": round(latest_end, 2),
                "validation_t0_month": round(t0_m, 2), "modules_without_history": new_mods,
                "sealed_in_fold": CFG["sealed_period"] in tr or (va == CFG["sealed_period"] and k != "FINAL")}
    log.info(f"{k}: train {tr} ({trn:,} eligible) -> {va} ({van:,}); forward={forward}; label-latency-safe={safe}")
assert all(f["strictly_forward"] for f in folds.values()), "a fold is not strictly forward"
assert not any(f["sealed_in_fold"] for f in folds.values()), "sealed period appears in an inner fold"
write_json({"folds": folds, "assumption": "B presentations start ~February, J ~October; lengths from courses.csv"}, "fold_definition.json")
log.info("PASS: all folds strictly forward; sealed period absent from every inner fold")
