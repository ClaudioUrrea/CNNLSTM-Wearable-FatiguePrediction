# Real-Time Physiological Fatigue Prediction in Human–Robot Collaborative Manufacturing

Code and materials accompanying:

> Urrea, C. (2026). Real-Time Physiological Fatigue Prediction Using Wearable Sensor
> Fusion and Hybrid DeepLearning: An In Silico Digital Twin Study.
> *Sensors*, 26(XX), XXXXX. https://doi.org/10.3390/sXXXXXXXX

[![DOI](https://img.shields.io/badge/DOI-pending-blue.svg)](https://doi.org/10.6084/m9.figshare.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA.md)

---

## What this study is, and is not

Every result in the paper comes from simulation. Twenty-four synthetic operator
models, each carrying a 23-degree-of-freedom musculoskeletal model scaled to its
own anthropometry, work 8-hour shifts inside a CoppeliaSim cell with three UR5e
manipulators. Their sEMG, IMU and force signals are generated from the underlying
biomechanics; no human participant was recorded at any stage.

That design buys exact ground truth and permits loading regimes no ethics
committee would approve for a person. It costs behavioural realism. The synthetic
operators have no pain perception, no learned pacing strategy, no motivation, and
no memory of the previous shift. The 89.3% accuracy and the 43% load reduction
are properties of this simulation and are not predictions about a factory floor.
Prospective validation with human participants is the necessary next step and has
not been performed.

## Reproducing the numbers

```bash
git clone https://github.com/ClaudioUrrea/CNNLSTM-Wearable-FatiguePrediction.git
cd CNNLSTM-Wearable-FatiguePrediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/consistency_check.py     # verifies every figure quoted in the paper
python src/lso_rotation.py          # prints the six-fold operator rotation
python src/cluster_bootstrap.py     # operator-level confidence intervals
python figures/p5_figures.py        # regenerates all seven figures
```

`consistency_check.py` recomputes more than fifty quantities that appear in the
manuscript — window arithmetic, class weights, every cell of the confusion matrix,
the ROC geometry of Figure 3b, all effect sizes in Table 5, the latency budget —
and exits non-zero if any of them disagrees with what is printed. It is wired into
CI so that an edit to a table cannot silently desynchronise a figure.

## Layout

```
├── figures/p5_figures.py        generates Figures 1–7 as PDF and PNG
├── src/
│   ├── lso_rotation.py          six-fold operator rotation (Section 5.4)
│   ├── cluster_bootstrap.py     operator-level bootstrap CIs (Section 5.5)
│   └── consistency_check.py     numerical audit of the manuscript
├── data/                        cohort specification and simulation inputs
└── docs/REPRODUCE.md            full reproduction procedure
```

## Why the dataset is not deposited directly

The analysis rests on 294,912 feature windows of 127 dimensions. That array is
not archived here, because it is not a measurement: it is a deterministic function
of the cohort specification, the simulation configuration and the random seeds,
all of which are. Regenerating it is preferable to downloading it, since a reader
who regenerates can also perturb — change a seed, widen the endurance
distribution, drop a sensor — and see what happens to the conclusions.

`data/README.md` lists exactly which inputs are required and in what format.

## Evaluation protocol

The leave-subjects-out partition is rotated over six disjoint folds of four
operators. Each operator is held out for testing exactly once, used for
validation exactly once, and never scored by a model that has seen its own data.
All reported metrics are pooled over the six out-of-fold test sets.

The rotation is forced by the design of the study rather than adopted for rigour
alone. The intervention arm reallocates tasks on the strength of a predicted
fatigue state, so it needs an estimate for all 24 operators. Under a single fixed
16/4/4 split, 20 of them would either go unscored or be scored by a model trained
on their own windows. `src/lso_rotation.py` implements the rotation and asserts
these invariants.

Confidence intervals come from a cluster bootstrap over operators. Resampling
windows instead would treat 294,912 correlated observations as independent and
return an interval roughly an order of magnitude too narrow; `cluster_bootstrap.py`
prints both so the difference is visible.

## Citing

See [CITATION.cff](CITATION.cff), or use the archived release DOI for the exact
version of the code you ran.

## Licence

Code under the MIT Licence. Data, figures and documentation under CC BY 4.0.
CoppeliaSim itself is not redistributed here; the scene and Lua scripts are the
author's own work and are released under the terms above. Obtain the simulator
from Coppelia Robotics under whichever licence applies to you.

## Contact

Claudio Urrea — claudio.urrea@usach.cl
Electrical Engineering Department, University of Santiago of Chile
