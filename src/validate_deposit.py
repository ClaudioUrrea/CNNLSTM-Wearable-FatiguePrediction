"""
Deposit validator.

Three of the files in the Figshare deposit are experimental output and cannot be
reconstructed from the manuscript: the per-operator accuracies, the trained
weights, and the feature array. This script does not generate them. It checks
the ones you supply against the aggregates the article publishes, so that a
transcription slip or a stale file is caught before the archive is minted with
a DOI.

    python src/validate_deposit.py --data-dir data

Exit status 0 if every supplied file agrees with the paper, 1 otherwise. Files
that are absent are reported as skipped, not as failures, so the script is
usable while the deposit is still being assembled.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# Values as published. Edit here only if the article itself changes.
PUBLISHED = {
    "n_operators": 24,
    "windows_per_operator": 12_288,
    "windows_pooled": 294_912,
    "accuracy": 89.3,
    "balanced_accuracy": 87.1,
    "macro_f1": 87.7,
    "ci": (87.8, 90.8),
    "phenotype_counts": {"fast": 7, "average": 10, "resistant": 7},
    "phenotype_accuracy": {"fast": 85.1, "resistant": 92.8},
    "tau_bounds": {"fast": (None, 100), "average": (100, 170), "resistant": (170, None)},
    "cohort": {
        "age_years": (38.4, 9.2, 24, 56),
        "height_cm": (167.3, 8.4, 152, 184),
        "mass_kg": (71.2, 11.8, 54, 96),
        "bmi": (25.4, 3.1, 20.3, 31.2),
        "tau_s": (142.0, 48.0, 85, 218),
    },
    "sex_counts": {"M": 12, "F": 12},
}

problems: list[str] = []
skipped: list[str] = []


def ok(msg: str) -> None:
    print(f"[  ok  ] {msg}")


def bad(msg: str) -> None:
    print(f"[ FAIL ] {msg}")
    problems.append(msg)


def skip(msg: str) -> None:
    print(f"[ skip ] {msg}")
    skipped.append(msg)


def near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


# ---------------------------------------------------------------- cohort ----
def check_cohort(path: Path) -> None:
    if not path.exists():
        skip(f"{path.name} not present")
        return
    rows = list(csv.DictReader(path.open()))
    n = len(rows)
    if n != PUBLISHED["n_operators"]:
        bad(f"{path.name}: {n} rows, expected {PUBLISHED['n_operators']}")
        return
    ok(f"{path.name}: {n} rows")

    sexes = {}
    for r in rows:
        sexes[r["sex"]] = sexes.get(r["sex"], 0) + 1
    if sexes != PUBLISHED["sex_counts"]:
        bad(f"{path.name}: sex balance {sexes}, expected {PUBLISHED['sex_counts']}")
    else:
        ok(f"{path.name}: sex balance {sexes}")

    for col, (m, s, lo, hi) in PUBLISHED["cohort"].items():
        try:
            vals = [float(r[col]) for r in rows]
        except (KeyError, ValueError):
            bad(f"{path.name}: column '{col}' missing or non-numeric")
            continue
        gm, gs, gmin, gmax = mean(vals), sd(vals), min(vals), max(vals)
        tol_m = max(0.15, abs(m) * 0.01)
        if not near(gm, m, tol_m):
            bad(f"{path.name}: {col} mean {gm:.2f}, article states {m}")
        elif not near(gs, s, max(0.5, s * 0.08)):
            bad(f"{path.name}: {col} SD {gs:.2f}, article states {s}")
        elif gmin < lo - 1e-6 or gmax > hi + 1e-6:
            bad(f"{path.name}: {col} range [{gmin:g}, {gmax:g}] outside [{lo}, {hi}]")
        else:
            ok(f"{path.name}: {col} {gm:.1f} +/- {gs:.1f}, range [{gmin:g}, {gmax:g}]")

    # BMI must be internally consistent with height and mass.
    worst = 0.0
    for r in rows:
        try:
            implied = float(r["mass_kg"]) / (float(r["height_cm"]) / 100) ** 2
            worst = max(worst, abs(implied - float(r["bmi"])))
        except (KeyError, ValueError, ZeroDivisionError):
            worst = float("inf")
            break
    if worst > 0.15:
        bad(f"{path.name}: BMI inconsistent with height and mass (max error {worst:.2f})")
    else:
        ok(f"{path.name}: BMI consistent with height and mass")

    # Phenotype labels must follow the tau thresholds and the published counts.
    counts = {"fast": 0, "average": 0, "resistant": 0}
    mislabelled = 0
    for r in rows:
        try:
            tau, ph = float(r["tau_s"]), r["phenotype"].strip().lower()
        except (KeyError, ValueError):
            bad(f"{path.name}: tau_s or phenotype missing")
            return
        expected = "fast" if tau < 100 else ("average" if tau <= 170 else "resistant")
        if ph != expected:
            mislabelled += 1
        counts[ph] = counts.get(ph, 0) + 1
    if mislabelled:
        bad(f"{path.name}: {mislabelled} phenotype labels disagree with tau")
    elif counts != PUBLISHED["phenotype_counts"]:
        bad(f"{path.name}: phenotype counts {counts}, "
            f"article states {PUBLISHED['phenotype_counts']}")
    else:
        ok(f"{path.name}: phenotype counts {counts}")


# ------------------------------------------------- per-operator accuracy ----
def check_accuracy(path: Path, cohort: Path) -> None:
    if not path.exists():
        skip(f"{path.name} not present")
        return
    rows = list(csv.DictReader(path.open()))
    if len(rows) != PUBLISHED["n_operators"]:
        bad(f"{path.name}: {len(rows)} rows, expected {PUBLISHED['n_operators']}")
        return
    ok(f"{path.name}: {len(rows)} rows")

    try:
        acc = {int(r["operator_id"]): float(r["accuracy"]) for r in rows}
        nwin = {int(r["operator_id"]): int(r["n_windows"]) for r in rows}
    except (KeyError, ValueError):
        bad(f"{path.name}: needs columns operator_id, accuracy, n_windows")
        return

    if set(nwin.values()) != {PUBLISHED["windows_per_operator"]}:
        bad(f"{path.name}: n_windows should be {PUBLISHED['windows_per_operator']} "
            f"for every operator")
    else:
        ok(f"{path.name}: {PUBLISHED['windows_per_operator']:,} windows per operator")

    total = sum(nwin.values())
    if total != PUBLISHED["windows_pooled"]:
        bad(f"{path.name}: windows total {total:,}, expected "
            f"{PUBLISHED['windows_pooled']:,}")
    else:
        ok(f"{path.name}: pooled windows {total:,}")

    pooled = sum(acc[i] * nwin[i] for i in acc) / total
    if not near(pooled, PUBLISHED["accuracy"], 0.15):
        bad(f"{path.name}: pooled accuracy {pooled:.2f}%, article states "
            f"{PUBLISHED['accuracy']}%")
    else:
        ok(f"{path.name}: pooled accuracy {pooled:.2f}%")

    lo, hi = min(acc.values()), max(acc.values())
    if not near(lo, PUBLISHED["phenotype_accuracy"]["fast"], 1.0):
        bad(f"{path.name}: lowest operator accuracy {lo:.1f}%, article implies "
            f"about {PUBLISHED['phenotype_accuracy']['fast']}%")
    if not near(hi, PUBLISHED["phenotype_accuracy"]["resistant"], 1.0):
        bad(f"{path.name}: highest operator accuracy {hi:.1f}%, article implies "
            f"about {PUBLISHED['phenotype_accuracy']['resistant']}%")
    if near(lo, PUBLISHED["phenotype_accuracy"]["fast"], 1.0) and \
       near(hi, PUBLISHED["phenotype_accuracy"]["resistant"], 1.0):
        ok(f"{path.name}: accuracy spread [{lo:.1f}, {hi:.1f}]%")

    # Cross-check the phenotype ordering reported in Section 7.3.
    if cohort.exists():
        ph = {int(r["operator_id"]): r["phenotype"].strip().lower()
              for r in csv.DictReader(cohort.open())}
        by_ph = {}
        for i, a in acc.items():
            by_ph.setdefault(ph.get(i, "?"), []).append(a)
        means = {k: mean(v) for k, v in by_ph.items() if v}
        if "fast" in means and "resistant" in means:
            if means["fast"] < means.get("average", means["fast"]) <= means["resistant"]:
                ok("phenotype ordering fast < average < resistant, as reported")
            else:
                bad(f"phenotype ordering violated: {means}")


# ------------------------------------------------------------- artefacts ----
def check_models(models_dir: Path) -> None:
    if not models_dir.exists():
        skip("models/ not present")
        return
    files = list(models_dir.rglob("*"))
    weights = [f for f in files if f.suffix in {".pt", ".pth", ".json",
                                                ".joblib", ".pkl", ".ubj"}]
    expected = 5 * 6  # five architectures over six folds
    if len(weights) < expected:
        bad(f"models/: {len(weights)} weight files, expected at least {expected} "
            f"(5 architectures x 6 folds)")
    else:
        ok(f"models/: {len(weights)} weight files")


def check_features(path: Path) -> None:
    if not path.exists():
        skip(f"{path.name} not present (optional; regeneration is the default)")
        return
    try:
        import pandas as pd
    except ImportError:
        skip(f"{path.name} present but pandas unavailable to inspect it")
        return
    df = pd.read_parquet(path)
    if len(df) != PUBLISHED["windows_pooled"]:
        bad(f"{path.name}: {len(df):,} rows, expected "
            f"{PUBLISHED['windows_pooled']:,}")
    else:
        ok(f"{path.name}: {len(df):,} rows")
    feature_cols = [c for c in df.columns if c.startswith("f")]
    if len(feature_cols) != 127:
        bad(f"{path.name}: {len(feature_cols)} feature columns, expected 127")
    else:
        ok(f"{path.name}: 127 feature columns")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data", type=Path)
    args = ap.parse_args()
    d = args.data_dir

    print("--- cohort ---")
    check_cohort(d / "operator_cohort.csv")
    print("\n--- per-operator accuracy ---")
    check_accuracy(d / "per_operator_accuracy.csv", d / "operator_cohort.csv")
    print("\n--- trained models ---")
    check_models(d / "models")
    print("\n--- feature array (optional) ---")
    check_features(d / "feature_windows.parquet")

    print("\n" + "=" * 74)
    if skipped:
        print(f"{len(skipped)} file(s) not yet supplied:")
        for s in skipped:
            print(f"  - {s}")
    if problems:
        print(f"\n{len(problems)} DISAGREEMENT(S) WITH THE ARTICLE:")
        for p in problems:
            print(f"  - {p}")
        print("\nResolve these before minting the DOI.")
        return 1
    print("\nEverything supplied agrees with the published values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
