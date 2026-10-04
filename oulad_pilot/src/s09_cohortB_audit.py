"""Stage 9: Cohort B (KDD Cup 2015) audit for Objective 3: file hashes, recomputed counts and label distribution,
and a mapping test of the nine shared features against what the held files can support."""
from core import *
log = get_logger("s09")
K = ROOT / "data" / "kdd"
files = {"kddcup2015_train_labelled.csv": "enrolment_id, username, course_id, dropout (from enrollment_train + truth_train)",
         "kdd_analytical_4week.csv": "first-4-week aggregates per enrolment", "KDDCup2015_baseline_features.csv": "30-day event-type aggregates"}
man = [{"file": f, "content": c, "rows": int(sum(1 for _ in open(K / f)) - 1), "sha256": sha256(K / f)} for f, c in files.items() if (K / f).exists()]
lab = pd.read_csv(K / "kddcup2015_train_labelled.csv")
counts = {"labelled_enrolments": int(len(lab)), "unique_enrolment_ids": int(lab.enrollment_id.nunique()), "users": int(lab.username.nunique()),
          "courses": int(lab.course_id.nunique()), "dropout_1": int((lab.dropout == 1).sum()), "dropout_0": int((lab.dropout == 0).sum()),
          "dropout_rate": float(lab.dropout.mean()), "missing_labels": int(lab.dropout.isna().sum()),
          "enrolments_per_user_max": int(lab.groupby("username").size().max())}
per_course = lab.groupby("course_id").dropout.agg(["size", "mean"]).describe().round(3).to_dict()
log.info(f"COHORT B: {counts}")
bf = pd.read_csv(K / "KDDCup2015_baseline_features.csv")
mapping = [
 {"id": "f01", "oulad": "total_clicks (sum_click)", "kdd_source": "total_interactions (event count)", "computable_from_held_processed_files": True, "semantic_match": "partial: events, not clicks"},
 {"id": "f02", "oulad": "active_days", "kdd_source": "active_days", "computable_from_held_processed_files": True, "semantic_match": "yes (window 30 vs 28 days)"},
 {"id": "f03", "oulad": "mean_clicks_per_active_day", "kdd_source": "f01/f02", "computable_from_held_processed_files": True, "semantic_match": "partial"},
 {"id": "f04", "oulad": "days_since_last_click", "kdd_source": "log_train timestamps", "computable_from_held_processed_files": False, "semantic_match": "yes, needs raw log"},
 {"id": "f05", "oulad": "days_to_first_click", "kdd_source": "log_train timestamps + date.csv course start", "computable_from_held_processed_files": False, "semantic_match": "yes, needs raw log"},
 {"id": "f06", "oulad": "weekly_decay_ratio", "kdd_source": "log_train daily counts", "computable_from_held_processed_files": False, "semantic_match": "yes, needs raw log"},
 {"id": "f07", "oulad": "cv_daily_clicks", "kdd_source": "log_train daily counts", "computable_from_held_processed_files": False, "semantic_match": "yes, needs raw log"},
 {"id": "f08", "oulad": "forum_clicks (forumng)", "kdd_source": "discussion_events", "computable_from_held_processed_files": True, "semantic_match": "partial"},
 {"id": "f09", "oulad": "quiz_clicks (quiz)", "kdd_source": "problem_events", "computable_from_held_processed_files": True, "semantic_match": "partial"}]
assert set(["total_interactions", "active_days", "discussion_events", "problem_events"]) <= set(bf.columns)
assert len(bf) == counts["labelled_enrolments"]
harm = {"outcome": "OULAD at_risk = Fail or Withdrawn at presentation end; KDD dropout = no activity in the 10 days after a 30-day window. "
                   "The two labels are not the same construct; Objective 1's 'common definition' can only be an aligned disengagement proxy.",
        "window": "OULAD days 0-27 from presentation start; KDD 30 days from course start (date.csv). Truncation to days 0-27 requires the raw log.",
        "unit": "OULAD sum_click per site-day; KDD one row per logged event (access, navigate, problem, video, discussion, wiki, page_close).",
        "population": "KDD has no registration/unregistration dates, so the day-28 eligibility rule cannot be applied; no demographics (no fairness analysis)."}
write_json({"manifest": man, "counts": counts, "per_course_summary": per_course, "nine_feature_mapping": mapping,
            "computable_now": [m["id"] for m in mapping if m["computable_from_held_processed_files"]], "harmonisation_limits": harm,
            "raw_files_held_by_student_not_in_repository": ["enrollment_train.csv", "log_train.csv (589.63 MB)", "truth_train.csv",
                                                             "enrollment_test.csv", "log_test.csv (389.51 MB)", "truth_test.csv", "date.csv", "object.csv"]},
           "cohortB_audit.json")
log.info(f"nine-feature mapping: {sum(m['computable_from_held_processed_files'] for m in mapping)}/9 computable from held processed files; 4 need the raw log")
