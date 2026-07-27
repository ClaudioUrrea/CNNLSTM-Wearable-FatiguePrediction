# Reproduction and deposit procedure

## Part 1 — What runs today

With only the files in this repository:

```bash
pip install -r requirements.txt
python src/consistency_check.py     # ~70 numerical checks against the article
python src/lso_rotation.py          # fold table and invariants
python src/cluster_bootstrap.py     # cluster vs naive interval, side by side
python figures/p5_figures.py        # Figures 1–7 into figures_p5_sensors/
```

All four are deterministic. `consistency_check.py` exits 1 on any disagreement
and is the fastest way to detect a table, figure and text drifting apart after an
edit.

## Part 2 — What must be added before deposit

Five items, all listed with exact filenames in [`../data/README.md`](../data/README.md):

1. `data/operator_cohort.csv` — the 24-row cohort
2. `data/simulation_config.yaml` — the digital twin configuration
3. `data/random_seeds.json` — every stochastic seed
4. `data/coppeliasim/` — scene file and Lua scripts
5. `data/models/` — trained weights per fold, plus `per_operator_accuracy.csv`

Once these are present the full pipeline regenerates from scratch, and
`cluster_bootstrap.py` reproduces the published confidence intervals from real
per-operator accuracies instead of the illustrative values it currently uses.

## Part 3 — Depositing

### GitHub

```bash
git init && git add -A
git commit -m "Materials for Urrea (2026), Sensors"
git branch -M main
git remote add origin https://github.com/ClaudioUrrea/CNNLSTM-Wearable-FatiguePrediction.git
git push -u origin main
git tag -a v1.0.0 -m "Version submitted to Sensors"
git push origin v1.0.0
```

The workflow in `.github/workflows/checks.yml` runs the audit on every push, so a
broken commit is visible immediately.

### Figshare

Figshare mints a DOI on publication and versions cleanly, which matters because
the article will be revised.

1. Create the item, choosing **Software** as the type.
2. Upload the tagged release archive together with `data/` and `figures/`.
3. Paste the fields from `figshare_metadata.json` into the metadata form.
4. Reserve the DOI **before** publishing. Figshare shows the reserved DOI
   immediately, which lets you place it in the manuscript prior to acceptance.
5. Publish once the article is accepted, then link the article DOI back.

Zenodo is an equally good target and can mint a DOI automatically from a GitHub
release; `.zenodo.json` is included for that route. Use one or the other, not
both, so there is a single canonical archive.

### Update the manuscript

Replace the Data Availability Statement with:

```latex
\dataavailability{All code and simulation resources supporting this study are
publicly available, in both cases under CC-BY-4.0, to permit independent
replication and to comply with the FAIR (Findable, Accessible, Interoperable,
Reusable) principles. Source code is hosted on GitHub at
\url{https://github.com/ClaudioUrrea/CNNLSTM-Wearable-FatiguePrediction}
(accessed on DD Month 2026), and the archived release, cohort specification and
trained models are deposited on Figshare at
\url{https://doi.org/10.6084/m9.figshare.XXXXXXXX}. ...}
```

The full wording is already in the manuscript; keep the two in sync when the
DOI is minted.

Add the archive to the reference list as well, so it is citable independently of
the article.

## Part 4 — A note on the seeds

`random_seeds.json` uses 42 throughout, matching the value already hard-coded in
`figures/p5_figures.py` and used across the companion papers in this series.
Consistency across the series is worth more than seed diversity here: a reader
comparing P1 and P5 should not have to wonder whether a discrepancy came from the
method or from the seed.
