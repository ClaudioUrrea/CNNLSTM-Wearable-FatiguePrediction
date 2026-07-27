# Data directory

The 294,912-window feature array is not stored here. It is a deterministic
function of the five inputs below, so depositing those inputs is both smaller and
more useful than depositing the array: a reader who regenerates can also perturb.

Files marked **REQUIRED** must be added before the archive is released. Each is
listed with the exact filename the README and `docs/REPRODUCE.md` refer to, so
nothing else needs editing once they are in place.

---

## 1. `operator_cohort.csv` — REQUIRED

One row per synthetic operator, 24 rows. Column names and units:

| column | unit | description |
|---|---|---|
| `operator_id` | — | integer 1–24 |
| `sex` | — | `M` or `F` (12 of each) |
| `age_years` | years | |
| `height_cm` | cm | |
| `mass_kg` | kg | |
| `bmi` | kg/m² | must equal `mass_kg / (height_cm/100)²` |
| `tau_s` | s | endurance time constant of Equation (5) |
| `phenotype` | — | `fast` (τ<100), `average` (100≤τ≤170), `resistant` (τ>170) |

Summary statistics the file must reproduce (Table 3): age 38.4 ± 9.2 (24–56);
height 167.3 ± 8.4 (152–184); mass 71.2 ± 11.8 (54–96); BMI 25.4 ± 3.1
(20.3–31.2); τ 142 ± 48 (85–218), with τ_male 156 ± 51 and τ_female 128 ± 42.
Phenotype counts: 7 fast, 10 average, 7 resistant.

A template with the correct header is provided as `operator_cohort_template.csv`.

## 2. `simulation_config.yaml` — REQUIRED

Everything needed to rerun the digital twin: workspace dimensions, the three UR5e
poses, assembly cycle definition (part masses, reach distances, fastener torque
and count), the eight part orientations and eight bin positions, break schedule,
Bullet Physics contact parameters, and the Latin square session ordering.

## 3. `random_seeds.json` — REQUIRED

Seeds governing every stochastic component, so the regeneration is exact:

```json
{
  "cohort_sampling": 42,
  "latin_square_assignment": 42,
  "semg_noise": 42,
  "imu_noise": 42,
  "fsr_noise": 42,
  "packet_dropout": 42,
  "wireless_latency": 42,
  "fold_rotation": 42,
  "model_init": [0, 1, 2, 3, 4, 5],
  "bootstrap": 42
}
```

`model_init` carries one seed per fold of the rotation described in
`src/lso_rotation.py`.

## 4. `coppeliasim/` — REQUIRED

The scene file and the Lua scripts driving the 23-DOF kinematic chain. These are
the author's own work and are released under CC BY 4.0. The simulator itself is
not redistributed; obtain it from Coppelia Robotics.

## 5. `models/` — REQUIRED

Trained weights for the five architectures, one set per fold. Suggested naming:

```
models/hybrid_fold0.pt   models/lstm_fold0.pt   models/xgboost_fold0.json
models/hybrid_fold1.pt   ...                    models/rf_fold0.joblib
                                                models/svm_fold0.joblib
```

Also deposit `per_operator_accuracy.csv` (columns `operator_id`,
`accuracy`, `balanced_accuracy`, `n_windows`), which is what
`src/cluster_bootstrap.py` consumes to produce the published intervals.

---

## Optional

`feature_windows.npz` — the full 294,912 × 127 array plus labels. Regeneration is
preferred, but archiving it removes any dependence on the simulator being
available in future. Roughly 300 MB uncompressed; well within Figshare's per-file
limit.
