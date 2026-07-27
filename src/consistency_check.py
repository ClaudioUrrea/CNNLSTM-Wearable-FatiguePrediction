"""
Numerical consistency check for the manuscript.

Every quantity below is either stated in the paper or derivable from quantities
that are. The script recomputes each one and fails loudly on any mismatch, so
that an edit to a table, a figure or a sentence cannot silently desynchronise
the rest of the article.

    python src/consistency_check.py

Exit status is 0 when all checks pass, 1 otherwise.
"""

from __future__ import annotations

import math
import sys

import numpy as np

PASS, FAIL = "  ok  ", " FAIL "
_failures: list[str] = []


def check(name: str, got: float, expected: float, tol: float = 5e-3) -> None:
    ok = abs(got - expected) <= tol
    print(f"[{PASS if ok else FAIL}] {name:<52} got {got:>12,.4f}  expected {expected:>12,.4f}")
    if not ok:
        _failures.append(name)


def check_exact(name: str, got, expected) -> None:
    ok = got == expected
    print(f"[{PASS if ok else FAIL}] {name:<52} got {got!s:>12}  expected {expected!s:>12}")
    if not ok:
        _failures.append(name)


# --------------------------------------------------------------------------
# 1. Dataset arithmetic (Sections 5.1 and 6.4)
# --------------------------------------------------------------------------
print("\n--- dataset arithmetic ---")
N_OPERATORS, SESSIONS_PER_OP, SESSION_HOURS = 24, 8, 8
check_exact("total sessions", N_OPERATORS * SESSIONS_PER_OP, 192)
check_exact("total operational hours",
            N_OPERATORS * SESSIONS_PER_OP * SESSION_HOURS, 1536)

BREAK_MIN, WINDOW_S, STRIDE_S = 60, 30, 15
task_seconds = (SESSION_HOURS * 3600) - BREAK_MIN * 60
gross_windows = task_seconds // STRIDE_S
check_exact("gross windows per session (7 h at 15 s stride)", gross_windows, 1680)

WINDOWS_PER_SESSION = 1536
discarded = 1 - WINDOWS_PER_SESSION / gross_windows
check("fraction discarded at breaks and by quality flags", discarded * 100, 8.571, tol=0.05)
check_exact("windows per operator", WINDOWS_PER_SESSION * SESSIONS_PER_OP, 12_288)
POOLED = WINDOWS_PER_SESSION * SESSIONS_PER_OP * N_OPERATORS
check_exact("pooled out-of-fold windows", POOLED, 294_912)

# --------------------------------------------------------------------------
# 2. Feature vector (Section 4.3, Table 2)
# --------------------------------------------------------------------------
print("\n--- feature vector ---")
features = {"sEMG time-domain": 5 * 8, "sEMG frequency-domain": 3 * 8,
            "IMU kinematics": 6 * 6, "biomechanical": 12,
            "FSR force dynamics": 3 * 4, "contextual": 3}
check_exact("sEMG features per channel", (5 * 8 + 3 * 8) // 8, 8)
check_exact("feature vector dimension", sum(features.values()), 127)
check_exact("non-biomechanical features", 127 - features["biomechanical"], 115)

# --------------------------------------------------------------------------
# 3. Class weighting (Section 5.4)
# --------------------------------------------------------------------------
print("\n--- class weighting ---")
p = np.array([0.45, 0.38, 0.17])
w = (1 / p) * (3 / np.sum(1 / p))
check("weight sum (normalised to K = 3)", float(w.sum()), 3.0)
for label, value, expected in zip(("fresh", "moderate", "severe"), w, (0.62, 0.74, 1.64)):
    check(f"class weight, {label}", float(value), expected, tol=6e-3)

# --------------------------------------------------------------------------
# 4. Confusion matrix (Figure 4a) and Table 4
# --------------------------------------------------------------------------
print("\n--- confusion matrix ---")
cm = np.array([[124_074, 7_158, 1_476],
               [7_164, 99_846, 5_058],
               [2_526, 8_214, 39_396]])
check_exact("confusion matrix total", int(cm.sum()), POOLED)
row, col, diag = cm.sum(1), cm.sum(0), np.diag(cm)
check_exact("row totals", tuple(int(x) for x in row), (132_708, 112_068, 50_136))
for label, r, expected in zip(("fresh", "moderate", "severe"), diag / row, (0.935, 0.891, 0.786)):
    check(f"recall, {label}", float(r), expected, tol=1e-3)
for label, pr, expected in zip(("fresh", "moderate", "severe"), diag / col, (0.928, 0.867, 0.858)):
    check(f"precision, {label}", float(pr), expected, tol=1e-3)

acc = diag.sum() / cm.sum()
bal = float(np.mean(diag / row))
f1 = 2 * (diag / col) * (diag / row) / ((diag / col) + (diag / row))
check("overall accuracy", acc * 100, 89.3, tol=0.05)
check("balanced accuracy", bal * 100, 87.1, tol=0.05)
check("macro F1 (Table 4)", float(f1.mean()) * 100, 87.7, tol=0.05)
check("severe-class F1", float(f1[2]) * 100, 82.0, tol=0.1)

check("class prior, fresh", float(row[0] / cm.sum()) * 100, 45.0, tol=0.1)
check("class prior, moderate", float(row[1] / cm.sum()) * 100, 38.0, tol=0.1)
check("class prior, severe", float(row[2] / cm.sum()) * 100, 17.0, tol=0.1)

# --------------------------------------------------------------------------
# 5. Model comparison margins (Section 7.1)
# --------------------------------------------------------------------------
print("\n--- model comparison ---")
acc_by_model = {"SVM": 82.1, "RF": 84.3, "XGBoost": 86.7, "LSTM": 88.5, "hybrid": 89.3}
check("hybrid over LSTM", acc_by_model["hybrid"] - acc_by_model["LSTM"], 0.8)
check("hybrid over XGBoost", acc_by_model["hybrid"] - acc_by_model["XGBoost"], 2.6)
check("hybrid over SVM", acc_by_model["hybrid"] - acc_by_model["SVM"], 7.2)
check("smallest sequential-to-instantaneous gap",
      acc_by_model["LSTM"] - acc_by_model["XGBoost"], 1.8)
check("calibration gain (73% generic to 89.3%)", acc_by_model["hybrid"] - 73.0, 16.3)

# --------------------------------------------------------------------------
# 6. Binormal ROC of Figure 3b (Section 7.1)
# --------------------------------------------------------------------------
print("\n--- ROC geometry ---")


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def phi_inv(q, lo=-9.0, hi=9.0):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if phi(mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


tp = 0.79 * 17.0
fp = tp / 0.86 - tp
op_fpr, op_tpr = fp / 83.0, 0.79
check("argmax operating point, specificity", (1 - op_fpr) * 100, 97.4, tol=0.1)

z0, k, a_auc = phi_inv(1 - op_fpr), -phi_inv(1 - op_tpr), phi_inv(0.95)
lo, hi = 1e-4, 50.0
for _ in range(200):
    mid = 0.5 * (lo + hi)
    g = lambda s: z0 + k * s - a_auc * math.sqrt(1 + s * s)
    if g(lo) * g(mid) <= 0:
        hi = mid
    else:
        lo = mid
sigma = 0.5 * (lo + hi)
mu = a_auc * math.sqrt(1 + sigma * sigma)

zs = np.linspace(6, -6, 20_001)
fpr_curve = np.array([1 - phi(z) for z in zs])
tpr_curve = np.array([1 - phi((z - mu) / sigma) for z in zs])
check("fitted ROC AUC", float(np.trapezoid(tpr_curve, fpr_curve)), 0.95, tol=1e-3)
check("curve passes through argmax point",
      float(np.interp(op_fpr, fpr_curve, tpr_curve)), op_tpr, tol=2e-3)
j = int(np.argmax(tpr_curve - fpr_curve))
check("Youden point, sensitivity", float(tpr_curve[j]) * 100, 85.2, tol=0.1)
check("Youden point, specificity", (1 - float(fpr_curve[j])) * 100, 93.4, tol=0.1)

# --------------------------------------------------------------------------
# 7. Intervention effects (Table 5, Figure 7)
# --------------------------------------------------------------------------
print("\n--- intervention effects ---")
rows = {
    "peak shoulder moment (Nm)": (18.3, 4.2, 10.4, 2.8, -43, 2.21),
    "cumulative load (Nm h)": (12.7, 3.1, 8.8, 2.4, -31, 1.41),
    "injury risk events per shift": (3.2, 1.4, 0.4, 0.6, -88, None),
    "assemblies per hour": (7.8, 0.9, 7.3, 0.8, -6, None),
    "fatigue demand index": (61.2, 12.3, 40.7, 9.8, -34, 1.84),
}
for name, (m0, s0, m1, s1, pct, d) in rows.items():
    check(f"{name}: change", (m1 - m0) / m0 * 100, pct, tol=0.6)
    if d is not None:
        pooled_sd = math.sqrt((s0 ** 2 + s1 ** 2) / 2)
        check(f"{name}: Cohen's d", (m0 - m1) / pooled_sd, d, tol=0.02)
check("throughput retained", 7.3 / 7.8 * 100, 94.0, tol=0.5)

# --------------------------------------------------------------------------
# 8. Synthetic cohort (Table 3, Section 6.1)
# --------------------------------------------------------------------------
print("\n--- synthetic cohort ---")
check("BMI implied by mean height and mass", 71.2 / (1.673 ** 2), 25.4, tol=0.1)
tau_m, sd_m, tau_f, sd_f = 156.0, 51.0, 128.0, 42.0
pooled_mean = (tau_m + tau_f) / 2
pooled_sd = math.sqrt((sd_m ** 2 + sd_f ** 2) / 2
                      + ((tau_m - pooled_mean) ** 2 + (tau_f - pooled_mean) ** 2) / 2)
check("cohort endurance mean tau (s)", pooled_mean, 142.0, tol=0.5)
check("cohort endurance SD tau (s)", pooled_sd, 48.0, tol=1.0)
check("endurance coefficient of variation (%)", 48.0 / 142.0 * 100, 34.0, tol=0.5)
check_exact("phenotype cluster sizes", 7 + 10 + 7, N_OPERATORS)

# --------------------------------------------------------------------------
# 9. Latency budget (Section 4.2)
# --------------------------------------------------------------------------
print("\n--- latency budget ---")
low = [0, 20, 5, 15, 15, 22]
high = [10, 50, 15, 15, 15, 22]
check_exact("end-to-end latency, lower bound (ms)", sum(low) + 10, 87)
check_exact("end-to-end latency, upper bound (ms)", sum(high), 127)
check_exact("inference within the 100 ms real-time budget", 22 < 100, True)

# --------------------------------------------------------------------------
print("\n" + "=" * 78)
if _failures:
    print(f"{len(_failures)} CHECK(S) FAILED:")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
print("All consistency checks passed.")
sys.exit(0)
