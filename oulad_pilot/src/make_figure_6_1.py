"""Figure 6.1: numbered end-to-end framework with evidence status matching Table 6.13 (status register)."""
from core import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
E1 = json.loads((OUT / "eligibility_counts.json").read_text())["total"]; L = json.loads((OUT / "lasso_inner_results.json").read_text())
G = json.loads((OUT / "xgb_inner_results.json").read_text()); S = json.loads((OUT / "selection_outputs.json").read_text())
K = json.loads((OUT / "cohortB_audit.json").read_text())["counts"]; FC = json.loads((OUT / "feature_checks.json").read_text())
GREEN, AMBER, GREY, BLUE = "#d5ead8", "#fdf0c8", "#e6e6e6", "#eaf2fb"
EDGE = {GREEN: "#2e6fb0", AMBER: "#2e6fb0", GREY: "#7a7a7a", BLUE: "#2e6fb0"}
fig, ax = plt.subplots(figsize=(10, 11.3)); ax.set_xlim(0, 100); ax.set_ylim(13, 126); ax.axis("off")
def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=EDGE[fc], lw=1.6))
    ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=10.2, fontweight="bold", color="#1f3864")
    ax.text(x + w / 2, y + h - 5.2, body, ha="center", va="top", fontsize=8.3, linespacing=1.35)
def arr(x1, y1, x2, y2, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.3, color="#7a7a7a" if dashed else "#1f3864", linestyle="--" if dashed else "-"))
ax.text(50, 125.5, "Figure 6.1  End-to-end framework: data joins, components, baseline,\nprimary model and evaluation pathway (status at 4 October 2026)", ha="center", va="top", fontsize=13, fontweight="bold", color="#1f3864")
for i, (fc, lab) in enumerate([(GREEN, "Executed pre-seal; evidence in outputs/ (Prototype/Pilot until re-run on workstation)"),
                               (AMBER, "Scripted and rehearsed; awaiting authorisation or raw data (In progress)"), (GREY, "Planned for MR604")]):
    yy = 117.5 - i * 2.6
    ax.add_patch(FancyBboxPatch((3, yy - 0.9), 2.4, 1.8, boxstyle="round,pad=0.1", fc=fc, ec="#555")); ax.text(6.5, yy, lab, va="center", fontsize=8.6)
LW = 61; LX = 3; RX = 67; RW = 30
box(LX, 98, LW, 11.5, "D1  OULAD raw tables (Cohort A, CC BY 4.0)", "studentInfo 32,593 | studentRegistration 32,593 | studentVle 10,655,280\nvle 6,364 | studentAssessment 173,912 | assessments 206 | courses 22\nJoin key: (code_module, code_presentation, id_student)", BLUE)
box(LX, 87.5, LW, 8.5, "C1  Audit and provenance", "SHA-256 + row counts (G1 smoke test PASS); '?' read as missing;\nexact VLE duplicates collapsed (primary) or kept (sensitivity S2)", GREEN)
box(LX, 74.5, LW, 11.5, "C2  Join and day-28 eligibility", f"1:1 join; t0 = start of day 28; E2 registered before t0;\nE3 not unregistered on or before day 28 (15 day-28 cases: sensitivity S4)\nEligible {E1['eligible']:,}; already-left log {E1['already_left_by_t0']:,}", GREEN)
box(LX, 63.5, 29.5, 9, "C3  Label  [Obj 1]", "at_risk = Fail or Withdrawn\nsecondary: Withdrawn only (S1)", GREEN)
box(LX + 31.5, 63.5, 29.5, 9, "C4  Chronological split", "P1 2013B | P2 2013J | P3 2014B\nP4 2014J sealed; F1, F2 primary", GREEN)
box(LX, 51.5, LW, 10, "C5  Feature extraction (28 features)", f"Window days 0-27 only; pre-course (-25..-1) kept apart (f16)\nf21 corrected; {FC['passed']}/{FC['total']} checks pass, incl. window-injection test", GREEN)
box(LX, 41, LW, 8.5, "C6  Train-fold-only preprocessing (frozen)", "Median imputation + indicators fitted on training rows; scaling (LASSO);\nclass weighting primary; SMOTE sensitivity only (S5)", GREEN)
box(LX, 28.5, 29.5, 10.5, "C7a  Baseline: LASSO", f"L1 logistic regression\nC = {L['selected_C']} by mean F1/F2 AUC\ninner AUC {L['folds'][0]['auc']:.3f} / {L['folds'][1]['auc']:.3f}", GREEN)
box(LX + 31.5, 28.5, 29.5, 10.5, "C7b  Primary: XGBoost", f"Optuna 60 trials on F1, F2\ndepth {G['selected_params']['max_depth']}, {G['final_n_estimators']} trees\ninner AUC {G['folds_fixed_n_estimators'][0]['auc']:.3f} / {G['folds_fixed_n_estimators'][1]['auc']:.3f}", GREEN)
box(LX, 15, LW, 11.5, "C8  One-time sealed evaluation on P4", "Protocol locked (protocol_lock.json); code rehearsed on P1+P2 -> P3;\nruns only after Appendix V is signed. Paired student-cluster bootstrap of\nDelta-AUC vs margin 0.02; Youden and top-20% thresholds from training OOF", AMBER)
box(RX, 98, RW, 11.5, "D2  KDD Cup 2015 (Cohort B)", f"Audited: {K['labelled_enrolments']:,} labelled, {K['users']:,} users,\n{K['courses']} courses, {K['dropout_rate']:.1%} dropout.\nRaw log on student workstation", AMBER)
box(RX, 82, RW, 12, "C11  Cross-platform  [Obj 3]", "9 shared features: 5 computable from\nheld files, 4 need raw log (script\ntested on synthetic data); transfer\npending CC-3 decision", AMBER)
box(RX, 66, RW, 12, "C9  SHAP  [Obj 4a]", f"Inner folds done: top-5 Jaccard\nF2 vs F3 = {S['shap_top5_jaccard_F2_F3']['jaccard']:.2f}\n(F1 = F2 by construction);\ntest-set SHAP pending C8", AMBER)
box(RX, 50, RW, 12, "C10  Fairness  [Obj 4b]", "TPR/FPR by gender, age band, IMD;\nEOD with Wilson + bootstrap CIs\nat training-derived threshold;\nscripted, rehearsed, pending C8", AMBER)
box(RX, 15, RW, 31, "R  Required thesis outputs", "R1 data-flow counts        DONE\nR2 fold-count table          DONE\nR7 at-risk definition record  DONE\nR3 Delta-AUC + CI (RQ2)    after C8\nR4 SHAP + stability (RQ4)  inner only\nR5 TPR/FPR/EOD + CI (RQ4)  after C8\nR6 cross-platform (RQ3)    after CC-3\n\nObj 5 (intervention): deferred,\nChapter 7 future work", BLUE)
cx = LX + LW / 2
arr(cx, 98, cx, 96.2); arr(cx, 88, cx, 86.2); arr(LX + 14.75, 74.5, LX + 14.75, 72.7); arr(LX + 46.25, 74.5, LX + 46.25, 72.7)
arr(LX + 14.75, 63.5, LX + 14.75, 61.7); arr(LX + 46.25, 63.5, LX + 46.25, 61.7); arr(cx, 51.5, cx, 49.7)
arr(LX + 14.75, 41, LX + 14.75, 39.2); arr(LX + 46.25, 41, LX + 46.25, 39.2); arr(LX + 14.75, 28.5, LX + 14.75, 26.7); arr(LX + 46.25, 28.5, LX + 46.25, 26.7)
arr(RX + RW / 2, 98, RX + RW / 2, 94.2)
ax.plot([65.5, 65.5], [20.5, 88], ls="--", color="#7a7a7a", lw=1.2)
arr(LX + LW, 20.5, RX, 20.5, dashed=True)
for yy in (88, 72, 56):
    arr(65.5, yy, RX, yy, dashed=True)
fig.savefig(OUT / "figures" / "fig_6_1_framework.png", dpi=200, bbox_inches="tight"); print("ok")
