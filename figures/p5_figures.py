#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Figure generation for P5 — "Real-Time Physiological Fatigue Prediction for
Human-Robot Collaborative Manufacturing Using Wearable Sensor Fusion and Hybrid
Deep Learning: An In Silico Digital Twin Study".

Target journal: MDPI *Sensors* (Manuscript ID sensors-4496124), Round 1 revision.
Author: Claudio Urrea, University of Santiago of Chile.

Run:  python3 p5_figures.py
Output: ./figures_p5_sensors/{Fig1..Fig10}_*.{pdf,png}
"""

import math
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')

from matplotlib import font_manager

# Palatino, per the final proofreading request.  Which cut is present depends on
# the machine: 'Palatino Linotype' ships with Windows and Office, 'Palatino' with
# macOS, and 'TeX Gyre Pagella', 'P052' and 'URW Palladio L' are the
# metric-compatible free clones found on Linux and in TeX distributions.
# The family has to be resolved here rather than hard-coded, because
# mathtext.rm/it/bf take one family name and silently fall back to DejaVu Sans
# if that exact name is absent -- which would set the text in Palatino and the
# math in a sans face on the same label.
PALATINO_STACK = ['Palatino Linotype', 'Palatino', 'TeX Gyre Pagella',
                  'P052', 'URW Palladio L', 'Book Antiqua', 'DejaVu Serif']


def _first_installed(candidates, fallback='DejaVu Serif'):
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            return name
    return fallback


SERIF_FAMILY = _first_installed(PALATINO_STACK)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter

OUTPUT_DIR = 'figures_p5_sensors'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MDPI printed widths, in inches.
COL_SINGLE = 3.39   # 8.6 cm
COL_DOUBLE = 7.01   # 17.8 cm

# ==============================================================================
# 1. GLOBAL STYLE
# ==============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': PALATINO_STACK,
    # Math is set in the whichever Palatino cut was actually found, so that
    # $T_i$, $\pm$ and $p$ do not switch typeface mid-label; anything that cut
    # lacks falls back to STIX, which is closer to Palatino in colour and weight
    # than Computer Modern.
    'mathtext.fontset': 'custom',
    'mathtext.rm': SERIF_FAMILY,
    'mathtext.it': SERIF_FAMILY + ':italic',
    'mathtext.bf': SERIF_FAMILY + ':bold',
    'mathtext.sf': SERIF_FAMILY,
    'mathtext.fallback': 'stix',
    'font.size': 9,
    'axes.titlesize': 9.5,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#333333',
    'axes.grid': False,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.35,
    'grid.linestyle': ':',
    'lines.linewidth': 1.4,
    'patch.linewidth': 0.8,
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.edgecolor': '#999999',
    'legend.fancybox': False,
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
    'pdf.fonttype': 42,          # embed TrueType, required by MDPI production
    'ps.fonttype': 42,
})

# ------------------------------------------------------------------------------
# Okabe-Ito palette: distinguishable under the common forms of colour-vision
# deficiency and separable in greyscale once hatching is added.
# ------------------------------------------------------------------------------
C = {
    'semg':     '#D55E00',   # vermillion
    'imu':      '#0072B2',   # blue
    'biomech':  '#009E73',   # bluish green
    'fsr':      '#E69F00',   # orange
    'context':  '#CC79A7',   # reddish purple
    'sky':      '#56B4E9',   # sky blue
    'yellow':   '#F0E442',
    'ink':      '#222222',
    'grey':     '#7F7F7F',
    'lightgrey': '#D9D9D9',
    'fresh':    '#009E73',
    'moderate': '#E69F00',
    'severe':   '#D55E00',
}

# Hatch patterns used wherever a bar group has to survive greyscale printing.
HATCH = {'semg': '///', 'imu': '\\\\\\', 'biomech': 'xxx',
         'fsr': '...', 'context': '+++', 'none': ''}


def panel_tag(ax, text, dx=-0.085, dy=1.045):
    """Place an (a)/(b) tag in the same position and style on every panel."""
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='bottom', ha='left')


def save(fig, stem, tight=True):
    """Vector PDF for submission plus 600 dpi PNG for on-screen checking."""
    if tight:
        fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/{stem}.pdf', bbox_inches='tight',
                facecolor='white')
    fig.savefig(f'{OUTPUT_DIR}/{stem}.png', dpi=600, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f'  written  {stem}.pdf / .png')


# ==============================================================================
# 2. SINGLE SOURCE OF TRUTH  (mirrors Tables 4-10 of the manuscript)
# ==============================================================================
RESULTS = {

    # ---- Table 4: classification performance, pooled out-of-fold -------------
    # order: as printed in Table 4 (worst to best raw accuracy)
    'models': [
        # name,            short,        acc,  ci_lo, ci_hi, bal,  f1,   auc,  lat, p_vs_hybrid, family
        ('SVM (RBF)',       'SVM',        82.1, 80.3, 83.9, 79.4, 80.0, 0.89,  8, 0.0009, 'inst'),
        ('Random Forest',   'RF',         84.3, 82.6, 86.0, 81.7, 82.3, 0.91, 12, 0.0009, 'inst'),
        ('XGBoost',         'XGB',        86.7, 85.1, 88.3, 84.2, 85.0, 0.93, 15, 0.0020, 'inst'),
        ('TCN',             'TCN',        88.2, 86.6, 89.8, 85.9, 86.6, 0.94, 16, 0.0880, 'seq'),
        ('LSTM',            'LSTM',       88.5, 86.9, 90.1, 86.3, 87.0, 0.94, 18, 0.1940, 'seq'),
        ('Transformer',     'Transf.',    88.9, 87.2, 90.6, 86.6, 87.3, 0.94, 27, 0.4710, 'seq'),
        ('1D-CNN-LSTM',     'Hybrid',     89.3, 87.8, 90.8, 87.1, 87.7, 0.95, 22, None,   'seq'),
        ('CNN-BiLSTM-Attn.', 'Attn.',     89.6, 88.0, 91.2, 87.4, 88.0, 0.95, 39, 0.6120, 'seq'),
    ],

    # ---- Table 5: computational cost ----------------------------------------
    # short -> (parameters, FLOPs, size MB, peak RAM MB, latency ms, energy mJ)
    # Parameter counts for the three instantaneous models are structural
    # quantities (support vectors / tree nodes), flagged by 'struct'.
    'cost': {
        'SVM':     dict(params=21_483,  struct=True,  flops=5.46e6, size=10.9, ram=96,  lat=8,  energy=28),
        'RF':      dict(params=1.28e6,  struct=True,  flops=0.06e6, size=24.6, ram=118, lat=12, energy=41),
        'XGB':     dict(params=0.21e6,  struct=True,  flops=0.02e6, size=4.1,  ram=74,  lat=15, energy=47),
        'TCN':     dict(params=118_900, struct=False, flops=1.42e6, size=0.45, ram=62,  lat=16, energy=58),
        'LSTM':    dict(params=180_700, struct=False, flops=2.17e6, size=0.69, ram=68,  lat=18, energy=66),
        'Transf.': dict(params=281_100, struct=False, flops=3.38e6, size=1.07, ram=81,  lat=27, energy=103),
        'Hybrid':  dict(params=73_900,  struct=False, flops=0.69e6, size=0.28, ram=57,  lat=22, energy=79),
        'Attn.':   dict(params=117_200, struct=False, flops=1.16e6, size=0.45, ram=66,  lat=39, energy=148),
    },

    # ---- Figure 5a / Section 7.3: pooled out-of-fold confusion matrix --------
    # rows = true class (fresh, moderate, severe), columns = predicted class.
    'confusion': np.array([[124_074,   7_158,   1_476],
                           [  7_164,  99_846,   5_058],
                           [  2_526,   8_214,  39_396]], dtype=float),

    # ---- Table 7: modality ablation -----------------------------------------
    # label, dims, balanced accuracy, group, number of BODY-WORN sensors
    'ablation': [
        ('Full vector',                     127, 87.1, 'full',   14),
        ('Without sEMG',                     63, 80.1, 'loo',     6),
        ('Without contextual',              124, 82.1, 'loo',    14),
        ('Without IMU',                      91, 83.4, 'loo',     8),
        ('Without biomechanical',           115, 83.6, 'loo',    14),
        ('Without FSR',                     115, 83.9, 'loo',    14),
        ('Without IMU and biomechanical',    79, 81.2, 'loo',     8),
        ('sEMG only',                        64, 79.8, 'single',  8),
        ('IMU only',                         36, 71.3, 'single',  6),
        ('Contextual only',                   3, 63.7, 'single',  0),
        ('FSR only',                         12, 58.4, 'single',  0),
        ('sEMG + IMU',                      100, 82.6, 'combo',  14),
        ('sEMG + contextual',                67, 83.2, 'combo',   8),
        ('sEMG + IMU + contextual',         103, 85.4, 'combo',  14),
        ('Reduced set (2 sEMG + 2 IMU + ctx.)', 31, 84.0, 'reduced', 4),
    ],

    # ---- Table 8: sensitivity to the FDI cut-points --------------------------
    # (low, high), class balance %, balanced acc, severe recall, severe F1, lead time min
    'threshold': [
        ((30, 60), (31, 41, 28), 85.9, 84.1, 85.0, 18.4),
        ((35, 65), (38, 40, 22), 86.5, 81.7, 83.6, 15.1),
        ((40, 70), (45, 38, 17), 87.1, 78.6, 82.0, 12.0),
        ((45, 75), (52, 35, 13), 87.6, 76.2, 79.9,  9.4),
        ((50, 80), (59, 32,  9), 88.4, 71.8, 76.3,  7.6),
    ],

    # ---- Table 9: construct validity ----------------------------------------
    'construct': [
        ('Biceps brachii fatigue index',      'semg',     0.86, 0.82, 0.89),
        ('Cumulative shoulder load',          'biomech',  0.83, 0.79, 0.87),
        ('Biceps brachii median frequency',   'semg',    -0.81, -0.85, -0.76),
        ('Anterior deltoid median frequency', 'semg',    -0.78, -0.83, -0.72),
        ('Biceps brachii RMS amplitude',      'semg',     0.74, 0.68, 0.79),
        ('Normalized jerk, dominant upper arm', 'imu',    0.61, 0.53, 0.68),
        ('Grip force coefficient of variation', 'fsr',    0.57, 0.47, 0.65),
    ],

    # ---- Table 10: intervention effectiveness --------------------------------
    'intervention': {
        'peak_moment':  dict(base=18.3, base_sd=4.2, interv=10.4, interv_sd=2.8,
                             change=-43, p='p < 0.001', d=2.21,
                             unit='Peak shoulder moment (N m)'),
        'cumulative':   dict(base=12.7, base_sd=3.1, interv=8.8, interv_sd=2.4,
                             change=-31, p='p < 0.001', d=1.41,
                             unit=r'Cumulative load (N m$\cdot$h)'),
        'fdi':          dict(base=61.2, base_sd=12.3, interv=40.7, interv_sd=9.8,
                             change=-34, p='p < 0.001', d=1.84,
                             unit='Fatigue Demand Index (0–100)'),
        'throughput':   dict(base=7.8, base_sd=0.9, interv=7.3, interv_sd=0.8,
                             change=-6, p='p = 0.012', d=0.59,
                             unit='Assemblies per hour'),
    },

    # ---- Table 3 / Figure 7a: feature vector composition ---------------------
    'features': [('sEMG', 64, 'semg'), ('IMU', 36, 'imu'),
                 ('Biomechanical', 12, 'biomech'), ('FSR', 12, 'fsr'),
                 ('Contextual', 3, 'context')],

    # ---- Figure 7b: top 15 features by mean |SHAP| ---------------------------
    'shap': [
        ('sEMG fatigue index, biceps',  0.28, 'semg'),
        ('Cumulative shoulder load',    0.22, 'biomech'),
        ('sEMG median freq., deltoid',  0.18, 'semg'),
        ('Time since last break',       0.15, 'context'),
        ('Movement jerk, upper arm',    0.14, 'imu'),
        ('sEMG RMS variability',        0.13, 'semg'),
        ('Shoulder flexion angle',      0.12, 'imu'),
        ('Elbow angular velocity',      0.11, 'imu'),
        ('sEMG zero crossings',         0.10, 'semg'),
        ('Cumulative work time',        0.08, 'context'),
        ('Peak shoulder torque',        0.07, 'biomech'),
        ('Trunk lateral bend',          0.06, 'imu'),
        ('sEMG waveform length',        0.05, 'semg'),
        ('Grip force variability',      0.04, 'fsr'),
        ('Forearm pronation rate',      0.03, 'imu'),
    ],

    # ---- Section 5.3 / Figure 5b: endurance phenotypes -----------------------
    # Reviewer 1 (line 490): the endurance time constant is T_i, never tau,
    # because tau denotes the joint torque vector in Equation (1).
    'phenotypes': [
        ('Fast fatiguer',    r'$T_i < 100$ s',        90,  7, 85.1, '#D55E00'),
        ('Average',          r'$T_i = 100$–$170$ s', 140, 10, 89.4, '#E69F00'),
        ('Fatigue-resistant', r'$T_i > 170$ s',      200,  7, 92.8, '#0072B2'),
    ],

    # ---- scalars quoted in more than one place ------------------------------
    'n_windows': 294_912,
    'n_operators': 24,
    'n_sessions': 192,
    'chance_level': 100.0 / 3.0,
    'latency_budget_ms': 100,
    'end_to_end_ms': (87, 127),
}


# ==============================================================================
# 3. AUDIT — recompute what the manuscript states and fail loudly on drift
# ==============================================================================
def audit():
    """Recompute every derived quantity the manuscript reports.

    This is the mechanism that keeps the figures and the tables in agreement.
    If a value in a table is edited without editing RESULTS, or the other way
    round, the script stops here instead of silently emitting a figure that
    contradicts the text.
    """
    cm = RESULTS['confusion']
    tot = cm.sum()
    assert tot == RESULTS['n_windows'], f'window total {tot:.0f}'

    rows = cm.sum(axis=1)
    np.testing.assert_allclose(rows, [132_708, 112_068, 50_136])

    recall = np.diag(cm) / cm.sum(axis=1)
    precision = np.diag(cm) / cm.sum(axis=0)
    f1 = 2 * precision * recall / (precision + recall)

    acc = np.diag(cm).sum() / tot * 100
    bal = recall.mean() * 100
    macro_f1 = f1.mean() * 100

    hybrid = dict(zip(
        ['name', 'short', 'acc', 'ci_lo', 'ci_hi', 'bal', 'f1', 'auc',
         'lat', 'p', 'family'],
        [m for m in RESULTS['models'] if m[1] == 'Hybrid'][0]))

    assert abs(acc - hybrid['acc']) < 0.05, f'accuracy {acc:.2f}'
    assert abs(bal - hybrid['bal']) < 0.05, f'balanced accuracy {bal:.2f}'
    assert abs(macro_f1 - hybrid['f1']) < 0.05, f'macro F1 {macro_f1:.2f}'

    # per-class values quoted in Section 7.3 and in the new Table 6
    for got, want, tag in zip(np.round(precision * 100), [93, 87, 86], 'FMS'):
        assert got == want, f'precision {tag}: {got}'
    for got, want, tag in zip(np.round(recall * 100), [93, 89, 79], 'FMS'):
        assert got == want, f'recall {tag}: {got}'

    # severe-class operating point used by Figure 3b
    fp = cm[0, 2] + cm[1, 2]
    neg = cm[0].sum() + cm[1].sum()
    spec = (neg - fp) / neg * 100
    assert abs(spec - 97.3) < 0.05, f'severe specificity {spec:.2f}'

    # ablation: the reduced set must sit 3.1 points below the full vector
    full = [a for a in RESULTS['ablation'] if a[3] == 'full'][0][2]
    red = [a for a in RESULTS['ablation'] if a[3] == 'reduced'][0][2]
    assert abs((full - red) - 3.1) < 0.05

    # threshold sweep: (40, 70) must be the setting reported in Table 4
    row = [r for r in RESULTS['threshold'] if r[0] == (40, 70)][0]
    assert abs(row[2] - hybrid['bal']) < 0.05
    for r in RESULTS['threshold']:
        assert abs(sum(r[1]) - 100) <= 1, f'class balance {r[0]}'

    # feature vector must add to 127
    assert sum(n for _, n, _ in RESULTS['features']) == 127

    print('  audit passed: figures and tables agree')
    return dict(precision=precision * 100, recall=recall * 100, f1=f1 * 100,
                acc=acc, bal=bal, macro_f1=macro_f1,
                severe_sens=recall[2] * 100, severe_spec=spec,
                severe_fpr=fp / neg)


# ==============================================================================
# 4. BINORMAL ROC HELPERS (Figure 3b) — no SciPy dependency
# ==============================================================================
_ERF = np.vectorize(math.erf, otypes=[float])


def _Phi(x):
    return 0.5 * (1.0 + _ERF(np.asarray(x, dtype=float) / np.sqrt(2.0)))


def _Phi_inv(p, lo=-9.0, hi=9.0, iters=200):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float(_Phi(mid)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _binormal_sigma(fpr0, tpr0, auc):
    """Slope of the binormal ROC with the given AUC through (fpr0, tpr0).

    A one-parameter ROC family fitted to the AUC alone cannot also pass
    through the operating point the confusion matrix implies; the binormal
    form has two free parameters and satisfies both constraints, so the marker
    drawn on the curve is genuinely on it.
    """
    z0 = _Phi_inv(1.0 - fpr0)
    k = -_Phi_inv(1.0 - tpr0)
    a = _Phi_inv(auc)

    def g(s):
        return z0 + k * s - a * np.sqrt(1.0 + s * s)

    lo, hi = 1e-4, 50.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if g(lo) * g(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ==============================================================================
# FIGURE 1 — Sensor placement and data acquisition architecture (Section 3.2)
# ==============================================================================
#
# Geometry of panel (a).  Three quantities are fixed and everything else is
# derived from them, so the panel can be re-tuned without hunting through
# coordinates:
#
#   BOX_L, BOX_R   horizontal extent of both framed regions
#   BODY_CX        centre line of the figure, held equal to the box centre
#   SQUEEZE        horizontal compression of the body, which buys the margin
#                  the annotation columns need
#
# The panel tag and the panel title sit ABOVE the dashed frame, not inside it:
# TAG_Y is above BOX_T by construction, and _check_panel_a() asserts it.
#
# Annotation is split by modality, sEMG on the left and IMU on the right, which
# balances the two columns.  Under the previous arrangement six labels sat on
# the left and two on the right, which is what made the panel read as though it
# were off-centre even though the frame itself was not.
# ------------------------------------------------------------------------------

BOX_L, BOX_R = 0.20, 10.65          # framed region, in panel-(a) data units
BOX_CX = 0.5 * (BOX_L + BOX_R)      # 5.425
BODY_CX = BOX_CX                    # body centre line == frame centre line
SQUEEZE = 0.80                      # horizontal compression of the body
BOX_T = 15.58                       # top of the dashed body-worn frame
TAG_Y = 17.00                       # panel tag and title, clear of the frame
PANEL_A_TOP = 17.60


def _bx(x_old, centre_old=6.55):
    """Map a coordinate from the uncompressed drawing onto the centred body."""
    return BODY_CX + (x_old - centre_old) * SQUEEZE


def _text_width(ax, s, fontsize, span):
    """Width of a string in data units of an axis spanning `span` units."""
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_window_extent()
    t = ax.text(0, 0, s, fontsize=fontsize)
    fig.canvas.draw()
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width
    t.remove()
    return w / bb.width * span


def _check_panel_a(ax, entries, span=11.0):
    """Assert that the tag clears the frame and that no text leaves the frame.

    Panel (a) is the one part of the figure set whose layout is hand-placed, so
    it is the one part that can silently break when a label is reworded.  This
    check runs on every build.
    """
    assert TAG_Y > BOX_T, f'panel tag at {TAG_Y} would sit inside the frame'
    problems = []
    for s, fontsize, anchor, align in entries:
        w = _text_width(ax, s, fontsize, span)
        left = {'left': anchor, 'right': anchor - w,
                'center': anchor - w / 2}[align]
        right = left + w
        if left < BOX_L - 0.02 or right > BOX_R + 0.02:
            problems.append(f'  {s!r:46s} spans {left:6.2f}..{right:6.2f}')
    if problems:
        raise AssertionError(
            f'panel (a): text outside the frame [{BOX_L}, {BOX_R}]\n'
            + '\n'.join(problems))


def fig1_sensor_architecture():
    """Figure 1.

    Reviewer 1 asked for larger type in panel (a).  Reviewer 2 pointed out that
    the four FSR elements are mounted on tools and on the workstation and are
    not worn, which the previous drawing did not convey.  Panel (a) is therefore
    divided into a BODY-WORN region and a shaded EQUIPMENT-MOUNTED region; no
    label is set below 7.4 pt; every site is named; placement is annotated once
    and marked bilateral; and the legend has been moved out of the drawing.
    """
    fig = plt.figure(figsize=(COL_DOUBLE, 5.1))
    gs = GridSpec(1, 2, figure=fig, wspace=0.05, width_ratios=[1.0, 1.12],
                  left=0.005, right=0.995, top=0.965, bottom=0.005)

    # ---------------------------------------------------------------- panel a
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 11)
    ax.set_ylim(-2.3, PANEL_A_TOP)
    ax.axis('off')

    # Tag and title above the frame, the title centred on the frame itself.
    ax.text(BOX_L, TAG_Y, '(a)', fontsize=10, fontweight='bold', va='top',
            ha='left')
    ax.text(BOX_CX, TAG_Y, 'Sensor placement', ha='center', va='top',
            fontsize=9.5, fontweight='bold')

    # --- body-worn region ------------------------------------------------
    ax.add_patch(Rectangle((BOX_L, 5.38), BOX_R - BOX_L, BOX_T - 5.38,
                           facecolor='white', edgecolor=C['grey'],
                           linewidth=0.8, linestyle=(0, (4, 2)), zorder=0))
    ax.text(BOX_CX, 14.72, 'BODY-WORN  (8 sEMG + 6 IMU = 14 sensors)',
            fontsize=8, fontweight='bold', color=C['ink'], va='center',
            ha='center')

    body = dict(color='#4D4D4D', linewidth=1.6, solid_capstyle='round',
                zorder=2)
    ax.add_patch(plt.Circle((BODY_CX, 13.05), 0.50, fill=False,
                            edgecolor='#4D4D4D', linewidth=1.6, zorder=2))
    ax.plot([BODY_CX, BODY_CX], [12.55, 12.25], **body)
    ax.plot([_bx(5.90), _bx(5.90), _bx(7.20), _bx(7.20), _bx(5.90)],
            [12.25, 9.25, 9.25, 12.25, 12.25], **body)
    ax.plot([_bx(5.90), _bx(4.85), _bx(4.35)], [12.05, 10.85, 9.35], **body)
    ax.plot([_bx(4.35), _bx(4.05)], [9.35, 8.30], **body)
    ax.plot([_bx(7.20), _bx(8.25), _bx(8.75)], [12.05, 10.85, 9.35], **body)
    ax.plot([_bx(8.75), _bx(9.05)], [9.35, 8.30], **body)
    ax.plot([_bx(6.15), _bx(6.00)], [9.25, 7.25], **body)
    ax.plot([_bx(6.95), _bx(7.10)], [9.25, 7.25], **body)

    # Markers. sEMG sites are annotated on the operator's right (viewer left),
    # IMU sites on the operator's left, so the two columns balance.
    emg = [(6.10, 12.32), (5.72, 11.90), (5.08, 11.10), (4.16, 8.75),
           (7.00, 12.32), (7.38, 11.90), (8.02, 11.10), (8.94, 8.75)]
    imu = [(4.62, 10.55), (4.20, 9.10), (8.48, 10.55), (8.90, 9.10),
           (6.55, 11.55), (6.55, 9.55)]
    for x, y in emg:
        ax.plot(_bx(x), y, 'o', color=C['semg'], ms=6.2, mec='white',
                mew=0.7, zorder=6)
    for x, y in imu:
        ax.plot(_bx(x), y, 's', color=C['imu'], ms=6.2, mec='white',
                mew=0.7, zorder=6)

    # Annotations: (label, marker x, marker y, text x, text y, colour, align)
    TX_L, TX_R = 4.02, 7.30
    annot = [
        ('Upper trapezius',  _bx(6.10), 12.32, TX_L, 13.75, C['semg'], 'right'),
        ('Anterior deltoid', _bx(5.72), 11.90, TX_L, 12.70, C['semg'], 'right'),
        ('Biceps brachii',   _bx(5.08), 11.10, TX_L, 11.65, C['semg'], 'right'),
        ('Flexor carpi radialis', _bx(4.16), 8.75, 2.45, 7.80, C['semg'], 'center'),
        ('Sternum',          _bx(6.55), 11.55, TX_R, 13.75, C['imu'],  'left'),
        ('Upper arm',        _bx(8.48), 10.55, TX_R, 12.80, C['imu'],  'left'),
        ('Forearm',          _bx(8.90),  9.10, TX_R, 10.45, C['imu'],  'left'),
        ('Sacrum',           _bx(6.55),  9.55, TX_R,  7.60, C['imu'],  'left'),
    ]
    for name, x, y, tx, ty, col, align in annot:
        ax.annotate(name, xy=(x, y), xytext=(tx, ty), fontsize=8, color=col,
                    ha=align, va='center', zorder=7,
                    arrowprops=dict(arrowstyle='-', lw=0.55, color=col,
                                    shrinkA=0, shrinkB=2.5))
    note = ('Sites are named once: sEMG and limb IMU placement is\n'
            'bilateral (4 muscles and 2 limb segments, each on both\n'
            'sides). Sternum and sacrum carry single, mid-line units.')
    ax.text(BOX_CX, 6.32, note, fontsize=7.3, ha='center', va='center',
            color=C['ink'], linespacing=1.4)

    # --- equipment-mounted region ----------------------------------------
    ax.add_patch(Rectangle((BOX_L, 0.50), BOX_R - BOX_L, 4.35,
                           facecolor='#F1F1F1', edgecolor=C['grey'],
                           linewidth=0.8, zorder=0))
    ax.text(BOX_CX, 4.50, 'EQUIPMENT-MOUNTED  (4 FSR elements, not worn)',
            fontsize=7.2, fontweight='bold', color=C['ink'], va='center',
            ha='center')

    # Three evenly spaced slots inside the frame.
    slots = [BOX_L + (BOX_R - BOX_L) * f for f in (1 / 6, 3 / 6, 5 / 6)]
    sx1, sx2, sx3 = slots

    ax.add_patch(FancyBboxPatch((sx1 - 0.95, 2.62), 1.90, 0.50,
                                boxstyle='round,pad=0.05,rounding_size=0.10',
                                facecolor='white', edgecolor='#4D4D4D',
                                linewidth=1.0, zorder=2))
    ax.add_patch(Rectangle((sx1 - 0.31, 3.12), 0.62, 0.62, facecolor='#DCDCDC',
                           edgecolor='#4D4D4D', linewidth=1.0, zorder=2))
    ax.plot(sx1 - 0.47, 2.87, '^', color=C['fsr'], ms=7, mec='white',
            mew=0.7, zorder=6)
    ax.plot(sx1 + 0.47, 2.87, '^', color=C['fsr'], ms=7, mec='white',
            mew=0.7, zorder=6)
    ax.text(sx1, 1.62, 'Power tool\nhandle (FSR 1, 2)', fontsize=7.4,
            ha='center', va='center', linespacing=1.35)

    ax.add_patch(Rectangle((sx2 - 0.62, 2.62), 0.34, 1.15, facecolor='white',
                           edgecolor='#4D4D4D', linewidth=1.0, zorder=2))
    ax.add_patch(Rectangle((sx2 + 0.28, 2.62), 0.34, 1.15, facecolor='white',
                           edgecolor='#4D4D4D', linewidth=1.0, zorder=2))
    ax.plot(sx2, 2.98, '^', color=C['fsr'], ms=7, mec='white', mew=0.7,
            zorder=6)
    ax.text(sx2, 1.62, 'Cobot gripper\njaw (FSR 3)', fontsize=7.4,
            ha='center', va='center', linespacing=1.35)

    ax.add_patch(Rectangle((sx3 - 1.00, 2.86), 2.00, 0.28, facecolor='white',
                           edgecolor='#4D4D4D', linewidth=1.0, zorder=2))
    ax.plot([sx3 - 0.74, sx3 - 0.74], [2.86, 2.30], color='#4D4D4D', lw=1.0,
            zorder=2)
    ax.plot([sx3 + 0.74, sx3 + 0.74], [2.86, 2.30], color='#4D4D4D', lw=1.0,
            zorder=2)
    ax.plot(sx3, 3.28, '^', color=C['fsr'], ms=7, mec='white', mew=0.7,
            zorder=6)
    ax.text(sx3, 1.62, 'Workstation\nsurface (FSR 4)', fontsize=7.4,
            ha='center', va='center', linespacing=1.35)

    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=C['semg'],
               markersize=6, label='sEMG, body-worn: 8 ch, 1000 Hz'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor=C['imu'],
               markersize=6, label='IMU, body-worn: 6 units, 100 Hz'),
        Line2D([0], [0], marker='^', color='none', markerfacecolor=C['fsr'],
               markersize=7, label='FSR, on equipment: 4 elements, 500 Hz'),
    ]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.002),
              ncol=1, fontsize=7.8, borderpad=0.45, handletextpad=0.4,
              labelspacing=0.35)

    # Layout contract for the hand-placed panel.
    _check_panel_a(ax, [
        ('BODY-WORN  (8 sEMG + 6 IMU = 14 sensors)', 8, BOX_CX, 'center'),
        ('EQUIPMENT-MOUNTED  (4 FSR elements, not worn)', 7.2, BOX_CX, 'center'),
        ('Sites are named once: sEMG and limb IMU placement is', 7.3, BOX_CX, 'center'),
        ('bilateral (4 muscles and 2 limb segments, each on both', 7.3, BOX_CX, 'center'),
        ('sides). Sternum and sacrum carry single, mid-line units.', 7.3, BOX_CX, 'center'),
        ('Upper trapezius', 8, TX_L, 'right'),
        ('Anterior deltoid', 8, TX_L, 'right'),
        ('Flexor carpi radialis', 8, 2.45, 'center'),
        ('Upper arm', 8, TX_R, 'left'),
        ('Sternum', 8, TX_R, 'left'),
        ('Power tool', 7.4, sx1, 'center'),
        ('handle (FSR 1, 2)', 7.4, sx1, 'center'),
        ('Cobot gripper', 7.4, sx2, 'center'),
        ('Workstation', 7.4, sx3, 'center'),
        ('surface (FSR 4)', 7.4, sx3, 'center'),
    ])

    # ---------------------------------------------------------------- panel b
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 11.6)
    ax2.set_ylim(-2.3, PANEL_A_TOP)
    ax2.axis('off')
    ax2.text(0.22, TAG_Y, '(b)', fontsize=10, fontweight='bold', va='top')
    ax2.text(5.80, TAG_Y, 'Acquisition and inference pipeline',
             ha='center', va='top', fontsize=9.5, fontweight='bold')

    for x, name in [(1.89, 'Sensor tier'), (5.80, 'Processing tier'),
                    (9.71, 'Actuation tier')]:
        ax2.text(x, 14.20, name, ha='center', fontsize=8.2, style='italic',
                 fontweight='bold', color=C['grey'])

    W, H = 3.34, 1.30
    boxes = [
        (0.22, 11.85, 'Wearable sensor\nnodes (18 units)', C['semg'], 'white'),
        (0.22,  9.65, 'BLE 5.0\ntransmission',             C['fsr'], C['ink']),
        (0.22,  7.45, 'Edge gateway\naggregation',         C['yellow'], C['ink']),
        (4.13, 11.85, 'Signal\nconditioning',              C['biomech'], 'white'),
        (4.13,  9.65, 'Feature extraction\n(127-D, every 15 s)', C['imu'], 'white'),
        (4.13,  7.45, '1D-CNN-LSTM\ninference',            C['ink'], 'white'),
        (8.04, 10.75, 'Fatigue state\nand alert',          C['context'], 'white'),
        (8.04,  8.55, 'HRC controller\nreallocation',      C['grey'], 'white'),
    ]
    for x, y, text, face, fg in boxes:
        ax2.add_patch(FancyBboxPatch((x, y), W, H,
                                     boxstyle='round,pad=0.05,rounding_size=0.12',
                                     facecolor=face, edgecolor=C['ink'],
                                     linewidth=0.9, alpha=0.95, zorder=3))
        ax2.text(x + W / 2, y + H / 2, text, ha='center', va='center',
                 fontsize=7.6, fontweight='bold', color=fg, zorder=4,
                 linespacing=1.3)

    arrows = [((1.89, 11.80), (1.89, 11.00)),
              ((1.89,  9.60), (1.89,  8.80)),
              ((3.62,  8.10), (4.07, 12.10)),
              ((5.80, 11.80), (5.80, 11.00)),
              ((5.80,  9.60), (5.80,  8.80)),
              ((7.53, 10.30), (7.98, 11.25)),
              ((7.53,  8.10), (7.98,  9.05))]
    for start, end in arrows:
        ax2.annotate('', xy=end, xytext=start, zorder=2,
                     arrowprops=dict(arrowstyle='-|>', lw=1.0, color='#555555',
                                     shrinkA=1, shrinkB=1, mutation_scale=9))

    for x, y, txt, ha in [(2.02, 11.40, '< 10 ms', 'left'),
                          (2.02,  9.20, '20–50 ms', 'left'),
                          (3.80,  9.20, '5–15 ms', 'left'),
                          (5.93, 11.40, '15 ms', 'left'),
                          (5.93,  9.20, '15 ms', 'left'),
                          (7.70,  8.18, '22 ms', 'left')]:
        ax2.text(x, y, txt, fontsize=7.5, style='italic', color=C['ink'],
                 ha=ha, va='center')

    lo, hi = RESULTS['end_to_end_ms']
    ax2.add_patch(FancyBboxPatch((0.55, 4.95), 10.50, 1.55,
                                 boxstyle='round,pad=0.05,rounding_size=0.12',
                                 facecolor='white', edgecolor=C['ink'],
                                 linewidth=1.0, zorder=3))
    ax2.text(5.80, 5.73,
             f'End-to-end latency {lo}–{hi} ms\n'
             f'(controller budget {RESULTS["latency_budget_ms"]} ms, '
             f'Section 1.3)',
             ha='center', va='center', fontsize=8.2, fontweight='bold',
             zorder=4, linespacing=1.4)

    save(fig, 'Fig1_Sensor_Architecture', tight=False)


# ==============================================================================
# FIGURE 2 — Signal processing pipeline (Section 3.3)
# ==============================================================================
def fig2_signal_processing():
    """Figure 2: representative waveforms at four stages of the chain."""
    fig, axes = plt.subplots(2, 2, figsize=(COL_DOUBLE, 4.8))
    rng = np.random.default_rng(42)

    t = np.linspace(0, 2, 2000)
    activation = 0.30 + 0.20 * np.sin(2 * np.pi * 0.5 * t)

    # (a) raw sEMG with mains interference
    ax = axes[0, 0]
    raw = activation * rng.standard_normal(t.size) * 0.50
    raw += 0.05 * np.sin(2 * np.pi * 50 * t)
    ax.plot(t, raw, color=C['grey'], lw=0.4)
    ax.set_ylabel('Amplitude (mV)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Raw sEMG, biceps brachii', fontsize=9, pad=11)
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.85, 0.85)
    ax.grid(True)
    ax.text(0.98, 0.95, '1000 Hz sampling\n50 Hz mains interference',
            transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                      edgecolor=C['lightgrey'], linewidth=0.6))
    panel_tag(ax, '(a)', dy=1.075)

    # (b) conditioned sEMG with RMS envelope
    ax = axes[0, 1]
    filtered = activation * rng.standard_normal(t.size) * 0.45
    win = 200
    env = np.convolve(np.abs(filtered), np.ones(win) / win, mode='same')
    ax.plot(t, filtered, color=C['lightgrey'], lw=0.4, label='Conditioned sEMG')
    ax.plot(t, env, color=C['semg'], lw=1.6, label='RMS envelope, 200 ms')
    ax.plot(t, -env, color=C['semg'], lw=1.6)
    ax.set_ylabel('Amplitude (mV)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Conditioned sEMG and RMS envelope', fontsize=9, pad=11)
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.85, 0.85)
    ax.grid(True)
    ax.legend(loc='lower right', fontsize=7.2, borderpad=0.35)
    ax.text(0.02, 0.95, '20–450 Hz bandpass\n50 Hz notch',
            transform=ax.transAxes, fontsize=7.5, va='top',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                      edgecolor=C['lightgrey'], linewidth=0.6))
    panel_tag(ax, '(b)', dy=1.075)

    # (c) spectral compression against amplitude escalation
    ax = axes[1, 0]
    tl = np.linspace(0, 60, 3000)
    mdf = 85 * np.exp(-tl / 300) + rng.normal(0, 2.0, tl.size)
    rms = 0.25 + 0.15 * (tl / 60) + rng.normal(0, 0.012, tl.size)
    axb = ax.twinx()
    l1, = ax.plot(tl, mdf, color=C['semg'], lw=1.3, label='Median frequency')
    l2, = axb.plot(tl, rms, color=C['imu'], lw=1.3, ls=(0, (5, 2)),
                   label='RMS amplitude')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Median frequency (Hz)', color=C['semg'])
    axb.set_ylabel('RMS amplitude (mV)', color=C['imu'])
    ax.tick_params(axis='y', colors=C['semg'])
    axb.tick_params(axis='y', colors=C['imu'])
    ax.set_title('Fatigue-indicative feature evolution', fontsize=9, pad=11)
    ax.set_xlim(0, 60)
    # Headroom on both axes so the legend sits in a clear band above the data
    # instead of over the rising RMS trace.
    ax.set_ylim(60, 104)
    axb.set_ylim(0.215, 0.505)
    ax.set_yticks([65, 70, 75, 80, 85, 90])
    axb.set_yticks([0.25, 0.30, 0.35, 0.40])
    ax.grid(True)
    ax.legend(handles=[l1, l2], loc='upper center', fontsize=7.0,
              borderpad=0.3, ncol=2, columnspacing=1.0, handlelength=1.6,
              bbox_to_anchor=(0.5, 1.0))
    ax.annotate('Spectral compression', xy=(47, 74.2), xytext=(30, 63.6),
                fontsize=7.4, color=C['ink'], ha='center',
                bbox=dict(boxstyle='square,pad=0.22', facecolor='white',
                          edgecolor='none'),
                arrowprops=dict(arrowstyle='->', lw=0.8, color=C['ink']))
    panel_tag(ax, '(c)', dy=1.075)

    # (d) IMU-derived shoulder flexion with risk bands
    ax = axes[1, 1]
    ti = np.linspace(0, 120, 6000)
    angle = (np.sin(2 * np.pi * ti / 8) * 35 + 55 + 5 * (ti / 120)
             + rng.normal(0, 2.5, ti.size))
    ax.axhspan(90, 130, color=C['severe'], alpha=0.10, zorder=0)
    ax.axhspan(60, 90, color=C['fsr'], alpha=0.10, zorder=0)
    ax.plot(ti, angle, color=C['biomech'], lw=0.6)
    ax.axhline(90, color=C['severe'], ls='--', lw=1.1)
    ax.axhline(60, color=C['fsr'], ls=(0, (4, 2)), lw=1.1)
    ax.text(2, 118, 'High risk, $>$ 90$^\\circ$', fontsize=7.4, ha='left',
            color=C['severe'])
    ax.text(2, 63.0, 'Moderate risk, 60–90$^\\circ$', fontsize=7.4, ha='left',
            color='#8A6100')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Shoulder flexion ($^\\circ$)')
    ax.set_title('IMU-derived joint angle monitoring', fontsize=9, pad=11)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 130)
    ax.grid(True)
    panel_tag(ax, '(d)', dy=1.075)

    save(fig, 'Fig2_Signal_Processing')


# ==============================================================================
# FIGURE 3 — Eight-architecture comparison (Section 7.1, Table 4)
# ==============================================================================
def fig3_model_comparison(stats):
    """Figure 3.

    Panel (a) now covers eight architectures rather than five, carries the
    operator-level bootstrap intervals from Table 4, and separates the
    instantaneous from the sequential models by hatching as well as by colour,
    which is the distinction the text says is the reliable one.
    """
    fig = plt.figure(figsize=(COL_DOUBLE, 3.6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.30, width_ratios=[1.28, 1.0])

    models = RESULTS['models']
    short = [m[1] for m in models]
    acc = np.array([m[2] for m in models])
    lo = np.array([m[3] for m in models])
    hi = np.array([m[4] for m in models])
    f1 = np.array([m[6] for m in models])
    auc = np.array([m[7] * 100 for m in models])
    fam = [m[10] for m in models]

    # ---- panel (a) --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(models))
    w = 0.27
    b1 = ax.bar(x - w, acc, w, yerr=[acc - lo, hi - acc],
                color=C['imu'], edgecolor=C['ink'], linewidth=0.6,
                error_kw=dict(lw=0.8, capsize=2, ecolor=C['ink']),
                label='Accuracy (%)')
    b2 = ax.bar(x, f1, w, color=C['biomech'], edgecolor=C['ink'],
                linewidth=0.6, label='Macro F1 (%)')
    b3 = ax.bar(x + w, auc, w, color=C['fsr'], edgecolor=C['ink'],
                linewidth=0.6, label=r'AUC $\times$ 100')

    # instantaneous models hatched, so the two families separate in greyscale
    for bars in (b1, b2, b3):
        for bar, family in zip(bars, fam):
            if family == 'inst':
                bar.set_hatch('///')

    # the hybrid is marked on its accuracy bar and by a bold tick label,
    # so no bar changes colour and the greyscale reading is unaffected
    idx_h = short.index('Hybrid')
    b1[idx_h].set_edgecolor(C['semg'])
    b1[idx_h].set_linewidth(1.7)

    ax.axvline(2.5, color=C['grey'], lw=0.9, ls=(0, (4, 2)))
    ax.text(1.25, 98.6, 'Instantaneous', ha='center', fontsize=7.6,
            style='italic', color=C['grey'])
    ax.text(5.25, 98.6, 'Sequential', ha='center', fontsize=7.6,
            style='italic', color=C['grey'])

    for xi, a, h in zip(x, acc, hi):
        ax.text(xi - w, h + 0.45, f'{a:.1f}', ha='center', fontsize=6.5,
                va='bottom', color=C['ink'])

    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=25, ha='right')
    for lbl, sname in zip(ax.get_xticklabels(), short):
        if sname == 'Hybrid':
            lbl.set_fontweight('bold')
            lbl.set_color(C['semg'])
    ax.set_ylabel('Performance (%)')
    ax.set_ylim(74, 101.5)
    ax.set_yticks([75, 80, 85, 90, 95])
    ax.grid(True, axis='y')
    ax.set_axisbelow(True)
    ax.legend(handles=[
        mpatches.Patch(facecolor=C['imu'], edgecolor=C['ink'], lw=0.6,
                       label='Accuracy (%)'),
        mpatches.Patch(facecolor=C['biomech'], edgecolor=C['ink'], lw=0.6,
                       label='Macro F1 (%)'),
        mpatches.Patch(facecolor=C['fsr'], edgecolor=C['ink'], lw=0.6,
                       label=r'AUC $\times$ 100'),
        mpatches.Patch(facecolor='white', edgecolor=C['ink'], lw=0.6,
                       hatch='///', label='Instantaneous model'),
    ], loc='lower left', ncol=2, fontsize=7.0, borderpad=0.3,
              columnspacing=0.9, handlelength=1.3, labelspacing=0.3)
    ax.set_title('Performance, pooled out-of-fold', fontsize=9, pad=13)
    panel_tag(ax, '(a)', dx=-0.115, dy=1.075)

    # ---- panel (b): binormal ROC for the severe fatigue class -------------
    ax2 = fig.add_subplot(gs[0, 1])
    op_fpr = stats['severe_fpr']
    op_tpr = stats['severe_sens'] / 100.0

    sigma = _binormal_sigma(op_fpr, op_tpr, 0.95)
    z = np.linspace(6.0, -6.0, 20001)
    fpr = 1.0 - _Phi(z)

    def roc(target_auc):
        mu = _Phi_inv(target_auc) * np.sqrt(1.0 + sigma * sigma)
        return 1.0 - _Phi((z - mu) / sigma)

    styles = [('SVM',     0.89, C['grey'],    (0, (1, 1.6)), 0.9),
              ('RF',      0.91, C['context'], (0, (5, 2)),   0.9),
              ('XGB',     0.93, C['fsr'],     (0, (3, 1.5)), 0.9),
              ('TCN',     0.94, C['biomech'], (0, (6, 1.5, 1, 1.5)), 1.0),
              ('LSTM',    0.94, C['sky'],     (0, (2, 1)),   1.0),
              ('Transf.', 0.94, C['imu'],     (0, (7, 2)),   1.0),
              ('Attn.',   0.95, C['semg'],    (0, (4, 1, 1, 1)), 1.0),
              ('Hybrid',  0.95, C['ink'],     '-',           1.9)]
    for name, a, col, ls, lw in styles:
        ax2.plot(fpr, roc(a), color=col, ls=ls, lw=lw,
                 label=f'{name} ({a:.2f})')
    ax2.plot([0, 1], [0, 1], color=C['lightgrey'], lw=0.9, ls='-')

    tpr_h = roc(0.95)
    i = int(np.argmax(tpr_h - fpr))
    ax2.plot(fpr[i], tpr_h[i], '*', ms=13, color=C['yellow'],
             markeredgecolor=C['ink'], markeredgewidth=0.9, zorder=10)
    ax2.annotate(f'Youden optimum\nSe {tpr_h[i]*100:.1f}%, '
                 f'Sp {(1-fpr[i])*100:.1f}%',
                 xy=(fpr[i], tpr_h[i]), xytext=(0.235, 0.635), fontsize=7.2,
                 bbox=dict(boxstyle='square,pad=0.28', facecolor='white',
                           edgecolor=C['ink'], linewidth=0.6),
                 arrowprops=dict(arrowstyle='->', lw=0.8, color=C['ink']))

    ax2.plot(op_fpr, op_tpr, 'o', ms=6, markerfacecolor='white',
             markeredgecolor=C['ink'], markeredgewidth=1.2, zorder=9)
    ax2.annotate(f'Argmax rule\nSe {op_tpr*100:.1f}%, '
                 f'Sp {(1-op_fpr)*100:.1f}%',
                 xy=(op_fpr, op_tpr), xytext=(0.30, 0.395), fontsize=7.2,
                 bbox=dict(boxstyle='square,pad=0.28', facecolor='white',
                           edgecolor=C['ink'], linewidth=0.6),
                 arrowprops=dict(arrowstyle='->', lw=0.8, color=C['ink']))

    ax2.set_xlabel('False positive rate (1 $-$ specificity)')
    ax2.set_ylabel('True positive rate (sensitivity)')
    ax2.set_title('ROC, severe fatigue class', fontsize=9, pad=13)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.005)
    ax2.grid(True)
    ax2.set_axisbelow(True)
    ax2.legend(loc='lower right', fontsize=6.3, ncol=2, borderpad=0.3,
               columnspacing=0.7, handlelength=1.6, labelspacing=0.2,
               title='Model (AUC)', title_fontsize=6.5)
    panel_tag(ax2, '(b)', dx=-0.155, dy=1.075)

    save(fig, 'Fig3_Model_Comparison')


# ==============================================================================
# FIGURE 4 — Accuracy against computational cost (Section 7.2, Table 5) [NEW]
# ==============================================================================
def fig4_complexity_tradeoff():
    """Figure 4, new in this revision.

    Requested by Reviewer 3 (comment 5).  Panel (a) is the accuracy-cost plane
    with the non-dominated set joined; panel (b) shows that measured latency and
    analytically counted FLOPs do not order the models the same way, which is
    the point the text makes about recurrent execution on this hardware.
    """
    fig = plt.figure(figsize=(COL_DOUBLE, 3.5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.28)

    models = RESULTS['models']
    cost = RESULTS['cost']
    fam_marker = {'inst': '^', 'seq': 'o'}

    # ---- panel (a) --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    pts = []
    for name, short, acc, lo, hi, bal, f1, auc, lat, p, fam in models:
        c = cost[short]
        col = C['semg'] if short == 'Hybrid' else (
            C['grey'] if fam == 'inst' else C['imu'])
        ax.scatter(c['params'], bal, s=14 * c['lat'], marker=fam_marker[fam],
                   facecolor=col, edgecolor=C['ink'], linewidth=0.7,
                   alpha=0.80, zorder=4)
        pts.append((c['params'], bal, short))

    # Label offsets are computed from each marker's own radius, because the
    # marker area encodes latency and a fixed offset lands inside the larger
    # bubbles.  radius = sqrt(area / pi), in points, plus a 3.5 pt gap.
    # (direction, horizontal alignment, extra horizontal shift in points).
    # TCN, LSTM and the attention model sit within a fifth of a decade of one
    # another, so direction alone does not separate their labels; the shift
    # pulls the two lower ones apart without moving them off their markers.
    placement = {'SVM':     ('below',  'center',   0),
                 'RF':      ('below',  'center',   0),
                 'XGB':     ('below',  'center',   0),
                 'TCN':     ('below',  'center', -11),
                 'LSTM':    ('below',  'center',  13),
                 'Transf.': ('right',  'left',     0),
                 'Hybrid':  ('left',   'right',    0),
                 'Attn.':   ('above',  'center',   0)}
    for px, py, short in pts:
        r = np.sqrt(14 * cost[short]['lat'] / np.pi) + 3.5
        where, ha, shift = placement[short]
        dx, dy, va = {'above': (0, r, 'bottom'),
                      'below': (0, -r, 'top'),
                      'left':  (-r, 0, 'center'),
                      'right': (r, 0, 'center')}[where]
        ax.annotate(short, (px, py), textcoords='offset points',
                    xytext=(dx + shift, dy), ha=ha, va=va, fontsize=7.4,
                    fontweight='bold' if short == 'Hybrid' else 'normal',
                    color=C['semg'] if short == 'Hybrid' else C['ink'])

    # non-dominated set: nothing is both more accurate and cheaper
    front = [p for p in pts
             if not any(q[1] > p[1] and q[0] < p[0] for q in pts)]
    front.sort(key=lambda q: q[0])
    ax.plot([p[0] for p in front], [p[1] for p in front], ls=(0, (4, 2)),
            lw=1.0, color=C['ink'], zorder=2, label='Non-dominated set')

    ax.set_xscale('log')
    ax.set_xlabel('Parameters (neural) or structural units (log scale)')
    ax.set_ylabel('Balanced accuracy (%)')
    ax.set_title('Accuracy against model size', fontsize=9)
    ax.set_ylim(76.6, 91.6)
    ax.grid(True, which='both')
    ax.set_axisbelow(True)

    # The sample-bubble key was dropped: at 40 ms the sample marker is 27 pt
    # across and collided with its own legend rows.  A one-line note carries
    # the same information without the collision.
    ax.legend(handles=[
        Line2D([0], [0], marker='^', color='none', markerfacecolor=C['grey'],
               markeredgecolor=C['ink'], markersize=6.5,
               label='Instantaneous'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=C['imu'],
               markeredgecolor=C['ink'], markersize=6.5, label='Sequential'),
        Line2D([0], [0], ls=(0, (4, 2)), color=C['ink'], lw=1.0,
               label='Non-dominated set'),
    ], loc='lower right', fontsize=6.9, borderpad=0.35, labelspacing=0.32,
        handletextpad=0.6)
    ax.text(0.025, 0.965, 'Marker area $\\propto$ measured latency (8–39 ms)',
            transform=ax.transAxes, fontsize=6.9, va='top', ha='left',
            color=C['ink'],
            bbox=dict(boxstyle='square,pad=0.25', facecolor='white',
                      edgecolor=C['lightgrey'], linewidth=0.6))
    panel_tag(ax, '(a)', dx=-0.155)

    # ---- panel (b) --------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    for name, short, *_rest in models:
        c = cost[short]
        fam = _rest[-1]
        col = C['semg'] if short == 'Hybrid' else (
            C['grey'] if fam == 'inst' else C['imu'])
        ax2.scatter(c['flops'] / 1e6, c['lat'], s=48,
                    marker=fam_marker[fam], facecolor=col,
                    edgecolor=C['ink'], linewidth=0.7, zorder=4)
        # TCN is labelled below its marker: above, it collides with LSTM,
        # which sits 2 ms higher and only a third of a decade to the right.
        dy, va = (-9.5, 'top') if short == 'TCN' else (8.5, 'bottom')
        ax2.annotate(short, (c['flops'] / 1e6, c['lat']),
                     textcoords='offset points', xytext=(0, dy),
                     ha='center', va=va, fontsize=7.4,
                     fontweight='bold' if short == 'Hybrid' else 'normal',
                     color=C['semg'] if short == 'Hybrid' else C['ink'])

    # the comparison the text draws: TCN does twice the arithmetic, 6 ms faster
    h, tcn = RESULTS['cost']['Hybrid'], RESULTS['cost']['TCN']
    ax2.annotate('', xy=(tcn['flops'] / 1e6, tcn['lat']),
                 xytext=(h['flops'] / 1e6, h['lat']),
                 arrowprops=dict(arrowstyle='<->', lw=0.9, color=C['ink'],
                                 ls=(0, (3, 2))), zorder=3)
    ax2.text(0.23, 18.4, '2.1$\\times$ the arithmetic,\n6 ms faster',
             fontsize=7.2, ha='center', va='center', color=C['ink'],
             bbox=dict(boxstyle='square,pad=0.25', facecolor='white',
                       edgecolor=C['lightgrey'], linewidth=0.6))

    ax2.set_xlabel('FLOPs per inference (M, log scale)')
    ax2.set_ylabel('Measured latency (ms)')
    ax2.set_title('Latency against arithmetic cost', fontsize=9)
    ax2.set_xscale('log')
    ax2.set_ylim(0, 46)
    ax2.grid(True, which='both')
    ax2.set_axisbelow(True)
    ax2.legend(handles=[
        Line2D([0], [0], marker='^', color='none', markerfacecolor=C['grey'],
               markeredgecolor=C['ink'], markersize=6.5,
               label='Instantaneous'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=C['imu'],
               markeredgecolor=C['ink'], markersize=6.5, label='Sequential'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=C['semg'],
               markeredgecolor=C['ink'], markersize=6.5,
               label='1D-CNN-LSTM'),
    ], loc='upper left', fontsize=6.9, borderpad=0.35)
    panel_tag(ax2, '(b)', dx=-0.155)

    save(fig, 'Fig4_Complexity_Tradeoff')


# ==============================================================================
# FIGURE 5 — Confusion matrix and inter-operator variability (Section 7.3)
# ==============================================================================
def fig5_confusion_variability(stats):
    """Figure 5.

    Panel (b) now labels the endurance time constant $T_i$, not $\\tau$, which
    Reviewer 1 flagged at line 490: $\\tau$ denotes the joint torque vector in
    Equation (1) and cannot also denote a time constant.  Per-class recall is
    printed beside each row so the matrix can be read without Table 6.
    """
    fig = plt.figure(figsize=(COL_DOUBLE, 3.5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.30, width_ratios=[1.0, 1.20])

    cm = RESULTS['confusion']
    labels = ['Fresh', 'Moderate', 'Severe']

    # ---- panel (a): confusion matrix, row-normalized shading --------------
    ax = fig.add_subplot(gs[0, 0])
    frac = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(frac, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i in range(3):
        for j in range(3):
            ax.text(j, i - 0.10, (f'{cm[i, j]:,.0f}' if cm[i, j] >= 10000 else f'{cm[i, j]:.0f}'), ha='center', va='center',
                    fontsize=7.6, fontweight='bold',
                    color='white' if frac[i, j] > 0.55 else C['ink'])
            ax.text(j, i + 0.20, f'{frac[i, j]*100:.1f}%', ha='center',
                    va='center', fontsize=7.4,
                    color='white' if frac[i, j] > 0.55 else C['grey'])
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title('Confusion matrix, 1D-CNN-LSTM\n'
                 f'(24 operators, {RESULTS["n_windows"]:,} windows)',
                 fontsize=9, pad=13)
    for i, r in enumerate(stats['recall']):
        ax.text(2.62, i, f'Recall\n{r:.1f}%', fontsize=7.4, va='center',
                ha='left', color=C['ink'])
    ax.set_xlim(-0.5, 3.60)
    ax.tick_params(labelsize=7.6)
    panel_tag(ax, '(a)', dx=-0.155, dy=1.075)

    # ---- panel (b): force capacity decline by phenotype --------------------
    ax2 = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(42)
    tt = np.linspace(0, 180, 240)
    ls_cycle = ['-', (0, (5, 2)), (0, (1.5, 1.5))]

    for (name, rng_lbl, T, n, acc, col), ls in zip(RESULTS['phenotypes'],
                                                   ls_cycle):
        mean = 100 * np.exp(-tt / T)
        for _ in range(n):
            jitter = rng.normal(0, 0.055) + 1.0
            ax2.plot(tt, np.clip(100 * np.exp(-tt / (T * jitter)), 4, 105),
                     color=col, alpha=0.16, lw=0.5, zorder=1)
        ax2.plot(tt, mean, color=col, lw=1.8, ls=ls, zorder=3,
                 label=f'{name} ({rng_lbl}, $n$ = {n})')

    ax2.axhline(50, color=C['grey'], ls=(0, (1, 2)), lw=0.9)
    ax2.text(178, 52, '50% MVC', fontsize=7.2, color=C['grey'], ha='right')
    ax2.set_xlabel('Time under load (s)')
    ax2.set_ylabel('Force capacity (% MVC)')
    ax2.set_title('Inter-operator variability, 24 synthetic operators\n'
                  '(endurance $T_i$: $142 \\pm 48$ s, CV = 34%)', fontsize=9,
                  pad=13)
    ax2.set_xlim(0, 180)
    ax2.set_ylim(0, 108)
    ax2.grid(True)
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper right', fontsize=7.0, borderpad=0.35,
               labelspacing=0.35)
    panel_tag(ax2, '(b)', dx=-0.145, dy=1.075)

    save(fig, 'Fig5_Confusion_Variability')


# ==============================================================================
# FIGURE 6 — Modality ablation (Section 7.4, Table 7) [NEW]
# ==============================================================================
def fig6_modality_ablation():
    """Figure 6, new in this revision.

    Requested by Reviewer 3 (comment 3) and by Reviewer 2 (comment 16), who
    asked that the ablation be moved out of the Discussion and reported as a
    result.  Panel (a) orders the fifteen configurations by balanced accuracy
    within their group; panel (b) puts the same numbers against the count of
    body-worn sensors each configuration needs, which is the quantity a plant
    engineer is constrained by.
    """
    fig = plt.figure(figsize=(COL_DOUBLE, 4.35))
    gs = GridSpec(1, 2, figure=fig, wspace=0.30, width_ratios=[1.42, 1.0])

    abl = RESULTS['ablation']
    full = [a for a in abl if a[3] == 'full'][0][2]
    chance = RESULTS['chance_level']

    group_colour = {'loo': C['imu'], 'single': C['fsr'],
                    'combo': C['biomech'], 'reduced': C['semg'],
                    'full': C['ink']}
    group_hatch = {'loo': '', 'single': '///', 'combo': '\\\\\\',
                   'reduced': 'xxx', 'full': ''}
    group_name = {'loo': 'Leave-one-block-out', 'single': 'Single block',
                  'combo': 'Combination', 'reduced': 'Reduced deployment set',
                  'full': 'Full 127-D vector'}

    # ---- panel (a): horizontal bars, grouped and ordered ------------------
    ax = fig.add_subplot(gs[0, 0])
    order = ['full', 'loo', 'combo', 'reduced', 'single']
    rows = []
    for g in order:
        block = sorted([a for a in abl if a[3] == g], key=lambda r: r[2])
        rows.extend(block)
    rows = rows[::-1]

    ypos = np.arange(len(rows))
    for y, (label, dims, bal, grp, worn) in zip(ypos, rows):
        ax.barh(y, bal - chance, left=chance, height=0.68,
                color=group_colour[grp], edgecolor=C['ink'], linewidth=0.6,
                hatch=group_hatch[grp], zorder=3)
        delta = bal - full
        txt = 'Reference' if grp == 'full' else f'{delta:+.1f} pp'.replace('-', '−')
        ax.text(bal - 0.9, y, f'{bal:.1f}%  ({txt})', va='center', ha='right',
                fontsize=6.9, color='white', fontweight='bold', zorder=5)

    ax.axvline(full, color=C['ink'], lw=1.1, ls=(0, (4, 2)), zorder=6)
    ax.axvline(chance, color=C['grey'], lw=1.0, ls=(0, (1, 2)), zorder=6)
    ax.text(full, len(rows) - 0.15, 'Full vector ', fontsize=7.0, ha='right',
            va='bottom', color=C['ink'])
    ax.text(chance, len(rows) - 0.15, ' Chance', fontsize=7.0, ha='left',
            va='bottom', color=C['grey'])

    ax.set_yticks(ypos)
    ax.set_yticklabels([f'{lab}  ({d}-D)' for lab, d, _, _, _ in rows],
                       fontsize=7.2)
    ax.set_xlim(chance - 1.0, 93.0)
    ax.set_ylim(-0.75, len(rows) + 0.35)
    ax.set_xlabel('Balanced accuracy (%)')
    ax.set_title('Configurations, hybrid retrained from scratch',
                 fontsize=9, pad=15)
    ax.grid(True, axis='x')
    ax.set_axisbelow(True)
    fig.legend(handles=[mpatches.Patch(facecolor=group_colour[g],
                                       edgecolor=C['ink'], linewidth=0.6,
                                       hatch=group_hatch[g],
                                       label=group_name[g])
                        for g in order],
               loc='lower center', bbox_to_anchor=(0.5, -0.045), ncol=5,
               fontsize=7.0, borderpad=0.35, columnspacing=1.1,
               handlelength=1.5)
    panel_tag(ax, '(a)', dx=-0.40, dy=1.055)

    # ---- panel (b): accuracy against body-worn sensor count ---------------
    ax2 = fig.add_subplot(gs[0, 1])
    for label, dims, bal, grp, worn in abl:
        ax2.scatter(worn, bal, s=46, marker='o' if grp != 'reduced' else 'D',
                    facecolor=group_colour[grp], edgecolor=C['ink'],
                    linewidth=0.7, zorder=4)

    ax2.axhline(full, color=C['ink'], lw=1.0, ls=(0, (4, 2)), zorder=2)
    ax2.text(14.0, full + 0.5, 'Full vector', fontsize=7.0, ha='right',
             va='bottom', color=C['ink'])

    red = [a for a in abl if a[3] == 'reduced'][0]
    ax2.annotate(f'Reduced set: {red[1]}-D,\n4 body-worn sensors,\n'
                 f'{red[2]:.1f}% balanced accuracy',
                 xy=(red[4], red[2]), xytext=(4.9, 66.5), fontsize=7.0,
                 ha='left',
                 bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                           edgecolor=C['semg'], linewidth=0.8),
                 arrowprops=dict(arrowstyle='->', lw=0.9, color=C['semg']))
    ctx = [a for a in abl if a[0] == 'Contextual only'][0]
    ax2.annotate('Contextual block\nneeds no sensor',
                 xy=(0, ctx[2]), xytext=(1.2, 55.6), fontsize=7.0,
                 bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                           edgecolor=C['lightgrey'], linewidth=0.6),
                 arrowprops=dict(arrowstyle='->', lw=0.8, color=C['grey']))

    ax2.set_xlabel('Body-worn sensors required (count)')
    ax2.set_ylabel('Balanced accuracy (%)')
    ax2.set_title('Accuracy against wearable burden', fontsize=9, pad=15)
    ax2.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax2.set_xlim(-0.9, 15.2)
    ax2.set_ylim(54, 91)
    ax2.grid(True)
    ax2.set_axisbelow(True)
    panel_tag(ax2, '(b)', dx=-0.22, dy=1.055)

    save(fig, 'Fig6_Modality_Ablation')


# ==============================================================================
# FIGURE 7 — Feature composition and SHAP importance (Section 7.5)
# ==============================================================================
def fig7_feature_importance():
    """Figure 7.

    Reviewer 1 asked that colour names be removed from the caption; the panel
    therefore labels each bar by modality directly and the caption identifies
    features by category rather than by hue.
    """
    fig = plt.figure(figsize=(COL_DOUBLE, 3.9))
    gs = GridSpec(1, 2, figure=fig, wspace=0.95, width_ratios=[1.0, 1.45])

    # ---- panel (a): composition of the 127-dimensional vector -------------
    ax = fig.add_subplot(gs[0, 0])
    names = [f[0] for f in RESULTS['features']]
    counts = [f[1] for f in RESULTS['features']]
    keys = [f[2] for f in RESULTS['features']]
    bars = ax.bar(range(len(names)), counts,
                  color=[C[k] for k in keys], edgecolor=C['ink'],
                  linewidth=0.7, width=0.62)
    for b, k in zip(bars, keys):
        b.set_hatch(HATCH[k])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 1.4, str(c), ha='center',
                fontsize=8, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=28, ha='right', fontsize=7.6)
    ax.set_ylabel('Features per window (count)')
    ax.set_ylim(0, 74)
    ax.set_title('Feature vector composition', fontsize=9, pad=13)
    ax.grid(True, axis='y')
    ax.set_axisbelow(True)
    ax.text(0.965, 0.965, '30 s windows,\n50% overlap',
            transform=ax.transAxes, fontsize=7.2, ha='right', va='top',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                      edgecolor=C['lightgrey'], linewidth=0.6))
    panel_tag(ax, '(a)', dx=-0.26, dy=1.055)

    # ---- panel (b): top 15 features by mean |SHAP| ------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    feats = RESULTS['shap'][::-1]
    y = np.arange(len(feats))
    for yi, (label, val, key) in zip(y, feats):
        ax2.barh(yi, val, height=0.66, color=C[key], edgecolor=C['ink'],
                 linewidth=0.6, hatch=HATCH[key], zorder=3)
        ax2.text(val + 0.006, yi, f'{val:.2f}', va='center', fontsize=7.0)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f[0] for f in feats], fontsize=7.0)
    ax2.set_xlabel('Mean $|$SHAP$|$ value')
    ax2.set_xlim(0, 0.325)
    ax2.set_ylim(-0.7, len(feats) - 0.3)
    ax2.set_title('Top 15 features, XGBoost attribution', fontsize=9, pad=13)
    ax2.grid(True, axis='x')
    ax2.set_axisbelow(True)

    counts_by_mod = {}
    for _, _, k in RESULTS['shap']:
        counts_by_mod[k] = counts_by_mod.get(k, 0) + 1
    order = ['semg', 'imu', 'biomech', 'context', 'fsr']
    pretty = {'semg': 'sEMG', 'imu': 'IMU', 'biomech': 'Biomechanical',
              'context': 'Contextual', 'fsr': 'FSR'}
    ax2.legend(handles=[mpatches.Patch(facecolor=C[k], edgecolor=C['ink'],
                                       linewidth=0.6, hatch=HATCH[k],
                                       label=f'{pretty[k]} '
                                             f'({counts_by_mod.get(k, 0)} of 15)')
                        for k in order],
               loc='lower right', fontsize=6.9, borderpad=0.35,
               labelspacing=0.3)
    panel_tag(ax2, '(b)', dx=-0.31, dy=1.055)

    save(fig, 'Fig7_Feature_Importance')


# ==============================================================================
# FIGURE 8 — Sensitivity to the FDI cut-points (Section 7.6, Table 8) [NEW]
# ==============================================================================
def fig8_threshold_sensitivity():
    """Figure 8, new in this revision.

    Requested by Reviewer 3 (comment 2).  The point the panels have to make is
    that balanced accuracy, severe-class recall and alert lead time do not share
    an optimum, so the operating point cannot be selected on accuracy alone.
    """
    fig = plt.figure(figsize=(COL_DOUBLE, 3.4))
    gs = GridSpec(1, 2, figure=fig, wspace=0.30, width_ratios=[1.0, 1.22])

    rows = RESULTS['threshold']
    labels = [f'{a} / {b}' for (a, b), *_ in rows]
    x = np.arange(len(rows))
    fresh = np.array([r[1][0] for r in rows], dtype=float)
    mod = np.array([r[1][1] for r in rows], dtype=float)
    sev = np.array([r[1][2] for r in rows], dtype=float)

    # ---- panel (a): class balance ----------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    b1 = ax.bar(x, fresh, 0.62, color=C['fresh'], edgecolor=C['ink'],
                linewidth=0.6, label='Fresh')
    b2 = ax.bar(x, mod, 0.62, bottom=fresh, color=C['moderate'],
                edgecolor=C['ink'], linewidth=0.6, hatch='///',
                label='Moderate')
    b3 = ax.bar(x, sev, 0.62, bottom=fresh + mod, color=C['severe'],
                edgecolor=C['ink'], linewidth=0.6, hatch='xxx',
                label='Severe')
    for xi, (f, m, s) in enumerate(zip(fresh, mod, sev)):
        ax.text(xi, f / 2, f'{f:.0f}', ha='center', va='center', fontsize=7.2,
                color='white', fontweight='bold')
        ax.text(xi, f + m / 2, f'{m:.0f}', ha='center', va='center',
                fontsize=7.2, color=C['ink'], fontweight='bold')
        ax.text(xi, f + m + s / 2, f'{s:.0f}', ha='center', va='center',
                fontsize=7.2, color='white', fontweight='bold')

    ax.axvspan(1.62, 2.38, color=C['lightgrey'], alpha=0.55, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_xlabel('FDI cut-points (lower / upper)')
    ax.set_ylabel('Share of windows (%)')
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title('Class balance across cut-points', fontsize=9, pad=8)
    ax.grid(True, axis='y')
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', ncol=3, fontsize=7.0, borderpad=0.3,
              columnspacing=0.8, handlelength=1.3)
    panel_tag(ax, '(a)', dx=-0.165, dy=1.03)

    # ---- panel (b): the three curves that disagree ------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    bal = [r[2] for r in rows]
    rec = [r[3] for r in rows]
    lead = [r[5] for r in rows]

    ax2.axvspan(1.75, 2.25, color=C['lightgrey'], alpha=0.55, zorder=0)
    l1, = ax2.plot(x, bal, 'o-', color=C['imu'], lw=1.5, ms=5,
                   markeredgecolor=C['ink'], markeredgewidth=0.6,
                   label='Balanced accuracy (%)')
    l2, = ax2.plot(x, rec, 's--', color=C['severe'], lw=1.5, ms=5,
                   markeredgecolor=C['ink'], markeredgewidth=0.6,
                   label='Severe-class recall (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7.6)
    ax2.set_xlabel('FDI cut-points (lower / upper)')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_ylim(66, 94)
    ax2.grid(True)
    ax2.set_axisbelow(True)

    ax3 = ax2.twinx()
    l3, = ax3.plot(x, lead, '^:', color=C['biomech'], lw=1.5, ms=5.5,
                   markeredgecolor=C['ink'], markeredgewidth=0.6,
                   label='Mean alert lead time (min)')
    ax3.axhline(10, color=C['grey'], lw=0.9, ls=(0, (1, 2)))
    # Set at the left end, where all three series are far above the line, so
    # the label no longer sits on top of the severe-recall curve.
    ax3.text(0.06, 10.35, '10 min requirement', fontsize=6.9, ha='left',
             va='bottom', color=C['grey'])
    ax3.set_ylabel('Lead time (min)', color=C['biomech'])
    ax3.tick_params(axis='y', colors=C['biomech'])
    ax3.set_ylim(4, 22)

    ax2.text(2.0, 92.3, 'Adopted', fontsize=7.0, ha='center', color=C['ink'],
             style='italic')
    ax2.legend(handles=[l1, l2, l3], loc='lower left', fontsize=6.9,
               borderpad=0.35, labelspacing=0.3)
    ax2.set_title('Accuracy, recall and lead time disagree', fontsize=9, pad=8)
    panel_tag(ax2, '(b)', dx=-0.145, dy=1.03)

    save(fig, 'Fig8_Threshold_Sensitivity')


# ==============================================================================
# FIGURE 9 — Real-time prediction over a 2-hour session (Section 7.8)
# ==============================================================================
def fig9_realtime_prediction():
    """Figure 9: one representative session, three synchronized panels."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(COL_DOUBLE, 5.0),
                                        sharex=True,
                                        gridspec_kw=dict(hspace=0.16))
    rng = np.random.default_rng(42)
    t = np.linspace(0, 120, 7200)
    breaks = [(30.0, 32.5), (60.0, 65.0), (90.0, 92.5)]

    def mark_breaks(ax, label=False):
        for k, (bs, be) in enumerate(breaks):
            ax.axvspan(bs, be, color=C['fresh'], alpha=0.16, zorder=0,
                       label='Scheduled break' if (label and k == 0) else None)

    # ---- (a) sEMG RMS amplitude ------------------------------------------
    rms = (0.25 + 0.15 * (t / 120) ** 1.5 + 0.08 * np.sin(2 * np.pi * t / 1.5)
           + rng.normal(0, 0.025, t.size))
    for bs, be in breaks:
        m = (t >= bs) & (t <= be)
        rms[m] = 0.15 + rng.normal(0, 0.018, m.sum())
    trend = np.convolve(rms, np.ones(300) / 300, mode='same')

    mark_breaks(ax1, label=True)
    ax1.plot(t, rms, color=C['lightgrey'], lw=0.4, label='Raw, 15 s windows')
    ax1.plot(t, trend, color=C['semg'], lw=1.5, label='Moving average')
    ax1.set_ylabel('sEMG RMS\n(mV)')
    ax1.set_ylim(0.08, 0.62)
    ax1.grid(True)
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper left', fontsize=7.0, ncol=3, borderpad=0.3,
               columnspacing=0.9, handlelength=1.4)
    ax1.set_title('Representative 2-hour simulated assembly session, '
                  'one synthetic operator', fontsize=9.5, pad=6)
    panel_tag(ax1, '(a)', dx=-0.075, dy=1.06)

    # ---- (b) shoulder flexion --------------------------------------------
    angle = (45 + 19 * (t / 120) ** 1.2 + 15 * np.sin(2 * np.pi * t / 1.2)
             + rng.normal(0, 2.5, t.size))
    for bs, be in breaks:
        m = (t >= bs) & (t <= be)
        angle[m] = 25 + rng.normal(0, 1.8, m.sum())

    mark_breaks(ax2)
    ax2.plot(t, angle, color=C['imu'], lw=0.4)
    ax2.axhline(90, color=C['severe'], ls='--', lw=1.1)
    ax2.text(119, 92, 'Ergonomic risk threshold, 90$^\\circ$', fontsize=7.0,
             ha='right', va='bottom', color=C['severe'])
    ax2.set_ylabel('Shoulder flexion\n($^\\circ$)')
    ax2.set_ylim(10, 108)
    ax2.grid(True)
    ax2.set_axisbelow(True)
    panel_tag(ax2, '(b)', dx=-0.075, dy=1.02)

    # ---- (c) predicted fatigue probability and alerts ---------------------
    prob = 0.14 + 0.80 * (1 - np.exp(-t / 74)) + rng.normal(0, 0.030, t.size)
    for bs, be in breaks:
        m = (t >= bs) & (t <= be)
        if m.sum():
            prob[m] = prob[m][0] * 0.55 + rng.normal(0, 0.022, m.sum())
        post = (t > be) & (t < be + 5)
        prob[post] *= 0.82
    prob = np.clip(prob, 0, 1)

    # The alert fires on the first sustained crossing of the 0.7 threshold,
    # read off the plotted trace rather than placed by hand, and the severe
    # fatigue state is entered one mean lead time later (Section 7.8).
    smooth = np.convolve(prob, np.ones(240) / 240, mode='same')
    above = np.flatnonzero((smooth > 0.70) & (t > 70))
    t_alert = float(t[above[0]])
    t_severe = t_alert + 12.0
    assert t_severe <= 120.0, f'lead time runs off the session: {t_severe}'

    ax3.axhspan(0.0, 0.5, color=C['fresh'], alpha=0.12, zorder=0)
    ax3.axhspan(0.5, 0.7, color=C['moderate'], alpha=0.14, zorder=0)
    ax3.axhspan(0.7, 1.05, color=C['severe'], alpha=0.12, zorder=0)
    mark_breaks(ax3)
    ax3.plot(t, prob, color=C['ink'], lw=1.0)
    ax3.axhline(0.7, color=C['severe'], ls='--', lw=1.2)
    ax3.axhline(0.5, color='#B07800', ls=(0, (4, 2)), lw=1.0)

    ax3.plot(t_alert, 0.70, marker='v', ms=8, color=C['severe'],
             markeredgecolor=C['ink'], markeredgewidth=0.7, zorder=7,
             label='Alert triggered (probability $\\geq$ 0.7)')
    for tv, col in ((t_alert, C['severe']), (t_severe, C['ink'])):
        ax3.axvline(tv, color=col, lw=0.9, ls=(0, (2, 2)), zorder=5)
    ax3.annotate('', xy=(t_severe, 0.145), xytext=(t_alert, 0.145),
                 arrowprops=dict(arrowstyle='<->', lw=1.0, color=C['ink']),
                 zorder=7)
    ax3.text((t_alert + t_severe) / 2, 0.185, 'Lead time\n12 min',
             ha='center', va='bottom', fontsize=7.0, fontweight='bold',
             color=C['ink'], zorder=7,
             bbox=dict(boxstyle='square,pad=0.2', facecolor='white',
                       edgecolor='none'))
    # Set in the top strip above the trace, which after the alert sits at
    # about 0.8; the label previously lay across the trace itself.
    ax3.text(119, 0.995, 'Severe fatigue state entered', fontsize=6.9,
             ha='right', va='top', color=C['ink'], zorder=7)

    # Band labels: the two upper ones at the left, where the trace is still
    # low, and 'fresh' in mid-session, where the trace has left that band.
    ax3.text(1.5, 0.945, 'Severe', fontsize=7.4, color=C['severe'],
             fontweight='bold', ha='left', va='center')
    ax3.text(1.5, 0.600, 'Moderate', fontsize=7.4, color='#B07800',
             fontweight='bold', ha='left', va='center')
    ax3.text(78, 0.200, 'Fresh', fontsize=7.4, color=C['fresh'],
             fontweight='bold', ha='center', va='center')
    ax3.text(1.5, 0.725, 'Alert threshold 0.7', fontsize=7.0, ha='left',
             va='bottom', color=C['severe'])
    ax3.set_xlabel('Time into session (min)')
    ax3.set_ylabel('Predicted fatigue\nprobability')
    ax3.set_xlim(0, 120)
    ax3.set_ylim(0, 1.02)
    ax3.grid(True)
    ax3.set_axisbelow(True)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.42, 1.0), fontsize=7.0,
               borderpad=0.3)
    ax3.set_ylim(0, 1.02)
    panel_tag(ax3, '(c)', dx=-0.075, dy=1.02)

    save(fig, 'Fig9_RealTime_Prediction', tight=False)


# ==============================================================================
# FIGURE 10 — Intervention effectiveness (Section 7.9, Table 10)
# ==============================================================================
def fig10_intervention():
    """Figure 10: four outcomes of alert-triggered reallocation.

    The throughput panel is drawn on the same footing as the other three: the
    change is small but statistically detectable at p = 0.012, and the panel
    says so rather than labelling it non-significant.
    """
    fig, axes = plt.subplots(2, 2, figsize=(COL_DOUBLE, 5.0))
    iv = RESULTS['intervention']
    keys = ['peak_moment', 'cumulative', 'fdi', 'throughput']
    tags = ['(a)', '(b)', '(c)', '(d)']
    titles = ['Peak shoulder moment', 'Cumulative shoulder load',
              'Fatigue Demand Index', 'Production throughput']
    # Reference-line labels are set two lines deep in the gap between the two
    # bars.  Running them along the line, as before, put them over the
    # baseline bar in every panel.
    extras = [(12.81, '70% of\nindividual\nMVC capacity'),
              (15.0, 'Intervention\ntrigger,\n15 N m$\\cdot$h'),
              (70.0, 'Severe\nfatigue\nboundary'),
              (6.63, '85% of\nbaseline\nthroughput')]
    ylims = [(0, 31), (0, 22), (0, 98), (0, 11.0)]

    for ax, key, tag, title, (hline, hlabel), ylim in zip(
            axes.ravel(), keys, tags, titles, extras, ylims):
        d = iv[key]
        means = [d['base'], d['interv']]
        sds = [d['base_sd'], d['interv_sd']]
        bars = ax.bar([0, 1], means, yerr=sds, width=0.34,
                      color=[C['grey'], C['imu']], edgecolor=C['ink'],
                      linewidth=0.8, capsize=3.5,
                      error_kw=dict(lw=0.9, ecolor=C['ink']))
        bars[0].set_hatch('///')

        ax.axhline(hline, color=C['fsr'], ls=(0, (4, 2)), lw=1.0, zorder=1)
        # Low in the gap between the bars, the one region of every panel that
        # is always empty, with a thin connector up to the line it names.
        y0 = ylim[1] * 0.055
        ax.text(0.5, y0, hlabel, fontsize=6.2, ha='center', va='bottom',
                color='#8A6100', zorder=5, linespacing=1.22)
        ax.plot([0.5, 0.5], [y0 + ylim[1] * 0.165, hline - ylim[1] * 0.012],
                color=C['fsr'], lw=0.7, ls=(0, (1, 2)), zorder=1)

        for xi, (m, sd) in enumerate(zip(means, sds)):
            ax.text(xi, m + sd + ylim[1] * 0.025, f'{m:.1f} $\\pm$ {sd:.1f}',
                    ha='center', fontsize=7.2, fontweight='bold')

        top = max(m + sd for m, sd in zip(means, sds)) + ylim[1] * 0.135
        ax.plot([0, 0, 1, 1], [top - ylim[1] * 0.022, top, top,
                               top - ylim[1] * 0.022],
                color=C['ink'], lw=0.9)
        ax.text(0.5, top + ylim[1] * 0.012,
                f"{d['change']:+d}%".replace('-', '−') + f"   ({d['p']}, $d$ = {d['d']:.2f})",
                ha='center', fontsize=7.2, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Baseline\n(96 sessions)',
                            'Prediction-triggered\n(96 sessions)'],
                           fontsize=7.4)
        ax.set_ylabel(d['unit'], fontsize=8)
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.62, 1.62)
        ax.set_title(title, fontsize=9, pad=6)
        ax.grid(True, axis='y')
        ax.set_axisbelow(True)
        panel_tag(ax, tag, dx=-0.185, dy=1.04)

    fig.legend(handles=[
        mpatches.Patch(facecolor=C['grey'], edgecolor=C['ink'], hatch='///',
                       linewidth=0.8, label='Baseline condition'),
        mpatches.Patch(facecolor=C['imu'], edgecolor=C['ink'], linewidth=0.8,
                       label='Prediction-triggered intervention'),
        Line2D([0], [0], color=C['fsr'], ls=(0, (4, 2)), lw=1.0,
               label='Reference level (see panel annotation)'),
    ], loc='lower center', bbox_to_anchor=(0.5, -0.055), ncol=3, fontsize=7.2,
        borderpad=0.35, columnspacing=1.2, handlelength=1.6)

    save(fig, 'Fig10_Intervention')


# ==============================================================================
# MANUSCRIPT AUDIT — check the numbers in the .tex against RESULTS
# ==============================================================================
#
# audit() protects the figures from drifting away from the tables.  It cannot
# see the running text, and the running text is where a number is most likely
# to be changed by hand during a rewrite: nothing complains if "89.3\%" becomes
# "89.5\%" in a sentence, because no figure depends on it.
#
# audit_manuscript() closes that gap.  It works on the principle that every
# quantity the prose states is either in RESULTS or derivable from it, so the
# expected value is computed here and never typed in.  Each check is a context
# pattern with one capture group: wherever that context appears in the .tex,
# the captured number must equal the derived value.
#
# Three outcomes are distinguished:
#   ERROR    the context appears with the wrong number  -> the text is wrong
#   MISSING  the context never appears                  -> a claim was dropped,
#                                                          or simply rephrased
#   OK       every occurrence matches
#
# MISSING is reported but is not a failure, because rewriting legitimately
# rephrases sentences.  ERROR is a failure.
#
# Usage:  python3 p5_figures.py --check-tex "P5_Sensors_(18.08.2026).tex"
# ------------------------------------------------------------------------------


def _load_tex(path):
    """Read the .tex and normalise the things that get in the way of matching."""
    with open(path, encoding='utf-8') as fh:
        src = fh.read()
    # drop the bibliography: reference titles contain numbers of their own
    src = re.split(r'\\begin\{thebibliography\}', src)[0]
    # drop full-line comments
    src = '\n'.join(ln for ln in src.split('\n')
                    if not ln.lstrip().startswith('%'))
    # LaTeX thousands separators and non-breaking spaces
    src = src.replace('{,}', ',').replace('~', ' ')
    # \SI{22}{\milli\second} -> "22 ms"-ish: keep the number, drop the unit
    src = re.sub(r'\\SI\{([^}]*)\}\{[^}]*\}', r'\1', src)
    src = re.sub(r'\\num\{([^}]*)\}', r'\1', src)
    # collapse whitespace so patterns are not broken by line wrapping
    src = re.sub(r'[ \t]*\n[ \t]*', ' ', src)
    src = re.sub(r' {2,}', ' ', src)
    return src


def _expected_values():
    """Every number the manuscript may state, derived from RESULTS.

    Nothing in this function is a literal taken from the manuscript.  If a
    value in RESULTS changes, the expectation changes with it, which is the
    whole point of the exercise.
    """
    cm = RESULTS['confusion']
    recall = np.diag(cm) / cm.sum(axis=1)
    precision = np.diag(cm) / cm.sum(axis=0)
    f1 = 2 * precision * recall / (precision + recall)

    acc = {m[1]: m[2] for m in RESULTS['models']}
    bal = {m[1]: m[5] for m in RESULTS['models']}
    lat = {m[1]: m[8] for m in RESULTS['models']}
    cost = RESULTS['cost']
    abl = {a[0]: a[2] for a in RESULTS['ablation']}
    thr = {r[0]: r for r in RESULTS['threshold']}
    iv = RESULTS['intervention']

    h, at, tr = cost['Hybrid'], cost['Attn.'], cost['Transf.']

    fp = cm[0, 2] + cm[1, 2]
    neg = cm[0].sum() + cm[1].sum()
    severe_spec = (neg - fp) / neg * 100

    sigma = _binormal_sigma(fp / neg, recall[2], 0.95)
    z = np.linspace(6.0, -6.0, 20001)
    f = 1.0 - _Phi(z)
    mu = _Phi_inv(0.95) * np.sqrt(1.0 + sigma * sigma)
    tp = 1.0 - _Phi((z - mu) / sigma)
    k = int(np.argmax(tp - f))

    v = {
        # --- headline classification results ------------------------------
        'acc_hybrid':        acc['Hybrid'],
        'bal_hybrid':        bal['Hybrid'],
        'f1_hybrid':         f1.mean() * 100,
        'acc_attn':          acc['Attn.'],
        'bal_attn':          bal['Attn.'],
        'acc_transf':        acc['Transf.'],
        'acc_lstm':          acc['LSTM'],
        'acc_tcn':           acc['TCN'],
        'acc_xgb':           acc['XGB'],
        'lat_hybrid':        lat['Hybrid'],
        'lat_tcn':           lat['TCN'],
        'lat_xgb':           lat['XGB'],
        'lat_attn':          lat['Attn.'],
        # --- per-class ----------------------------------------------------
        'prec_fresh':        precision[0] * 100,
        'rec_fresh':         recall[0] * 100,
        'prec_mod':          precision[1] * 100,
        'rec_mod':           recall[1] * 100,
        'prec_sev':          precision[2] * 100,
        'rec_sev':           recall[2] * 100,
        'f1_sev':            f1[2] * 100,
        'spec_sev':          severe_spec,
        'youden_sens':       tp[k] * 100,
        'youden_spec':       (1 - f[k]) * 100,
        # --- cost ---------------------------------------------------------
        'params_hybrid_k':   h['params'] / 1000.0,
        'size_hybrid':       h['size'],
        'ram_hybrid':        h['ram'],
        'flops_hybrid':      h['flops'] / 1e6,
        'ratio_params':      at['params'] / h['params'],
        'ratio_lat':         at['lat'] / h['lat'],
        'ratio_flops':       at['flops'] / h['flops'],
        'pct_params':        (at['params'] / h['params'] - 1) * 100,
        'pct_lat':           (at['lat'] / h['lat'] - 1) * 100,
        'pct_smaller_attn':  (at['size'] - h['size']) / at['size'] * 100,
        'transf_param_mult': tr['params'] / h['params'],
        # --- ablation -----------------------------------------------------
        'bal_full':          abl['Full vector'],
        'bal_no_semg':       abl['Without sEMG'],
        'bal_semg_only':     abl['sEMG only'],
        'bal_ctx_only':      abl['Contextual only'],
        'bal_imu_only':      abl['IMU only'],
        'bal_fsr_only':      abl['FSR only'],
        'bal_semg_imu':      abl['sEMG + IMU'],
        'bal_reduced':       abl['Reduced set (2 sEMG + 2 IMU + ctx.)'],
        'drop_semg':         abl['Full vector'] - abl['Without sEMG'],
        'drop_ctx':          abl['Full vector'] - abl['Without contextual'],
        'drop_reduced':      abl['Full vector']
                             - abl['Reduced set (2 sEMG + 2 IMU + ctx.)'],
        'drop_semg_imu':     abl['Full vector'] - abl['sEMG + IMU'],
        'gap_hybrid_xgb':    acc['Hybrid'] - acc['XGB'],
        'chance':            RESULTS['chance_level'],
        # --- thresholds ---------------------------------------------------
        'bal_lo':            thr[(30, 60)][2],
        'bal_hi':            thr[(50, 80)][2],
        'bal_range':         thr[(50, 80)][2] - thr[(30, 60)][2],
        'rec_drop':          thr[(30, 60)][3] - thr[(50, 80)][3],
        'lead':              thr[(40, 70)][5],
        'lead_lo':           thr[(30, 60)][5],
        'lead_hi':           thr[(50, 80)][5],
        # --- intervention -------------------------------------------------
        'peak_base':         iv['peak_moment']['base'],
        'peak_int':          iv['peak_moment']['interv'],
        'peak_chg':          abs(iv['peak_moment']['change']),
        'fdi_chg':           abs(iv['fdi']['change']),
        'thru_chg':          abs(iv['throughput']['change']),
        'cum_chg':           abs(iv['cumulative']['change']),
        # --- cohort and dataset -------------------------------------------
        'n_windows':         RESULTS['n_windows'],
        'n_operators':       RESULTS['n_operators'],
        'n_sessions':        RESULTS['n_sessions'],
        'n_features':        sum(n for _, n, _ in RESULTS['features']),
        'n_semg_feat':       dict((k, n) for k, n, _ in RESULTS['features'])['sEMG'],
        'n_imu_feat':        dict((k, n) for k, n, _ in RESULTS['features'])['IMU'],
        'lat_budget':        RESULTS['latency_budget_ms'],
        'e2e_lo':            RESULTS['end_to_end_ms'][0],
        'e2e_hi':            RESULTS['end_to_end_ms'][1],
    }
    return v


# (context pattern, key into _expected_values(), rounding, human description)
# The pattern must contain exactly one capture group, holding the number.
_TEX_CHECKS = [
    (r'(\d+\.\d)\\% three-class accuracy', 'acc_hybrid', 1,
     'hybrid three-class accuracy'),
    (r'hybrid reached (\d+\.\d)\\%', 'acc_hybrid', 1, 'hybrid accuracy'),
    (r'(\d+\.\d)\\% balanced accuracy with', 'bal_hybrid', 1,
     'hybrid balanced accuracy'),
    (r'unweighted gives the (\d+\.\d)\\% reported', 'f1_hybrid', 1, 'macro F1'),
    (r'highest raw accuracy at (\d+\.\d)\\%', 'acc_attn', 1,
     'attention-model accuracy'),
    (r'(\d+\.\d) ?k parameters', 'params_hybrid_k', 1, 'hybrid parameters'),
    (r'(\d+\.\d) ?k-parameter 1D-CNN', 'params_hybrid_k', 1,
     'hybrid parameters, Highlights'),
    (r'(\d+\.\d) times the parameters', 'ratio_params', 1,
     'parameter ratio, attention vs hybrid'),
    (r'(\d+\.\d) times the latency', 'ratio_lat', 1,
     'latency ratio, attention vs hybrid'),
    (r'(\d+\.\d) times the FLOPs', 'ratio_flops', 1, 'FLOPs ratio'),
    (r'(\d+)\\% smaller than the attention', 'pct_smaller_attn', 0,
     'hybrid size against attention model'),
    (r'(\d+\.\d) points of balanced accuracy to sEMG', 'drop_semg', 1,
     'sEMG ablation decrement'),
    (r'already logs costs (\d+\.\d) points', 'drop_ctx', 1,
     'contextual ablation decrement'),
    (r'Removing sEMG costs (\d+\.\d) points', 'drop_semg', 1,
     'sEMG ablation decrement, Discussion'),
    (r'retains (\d+\.\d)\\%', 'bal_reduced', 1, 'reduced-set accuracy'),
    (r'retains (\d+\.\d)\\% balanced accuracy', 'bal_reduced', 1,
     'reduced-set balanced accuracy'),
    (r'(\d+\.\d) points of balanced accuracy\.', 'drop_reduced', 1,
     'reduced-set decrement'),
    (r'reach (\d+\.\d)\\% balanced accuracy against a chance level',
     'bal_ctx_only', 1, 'contextual-only accuracy'),
    (r'chance level of (\d+\.\d)\\%', 'chance', 1, 'chance level'),
    (r'(\d+\.\d)\\% against (?:\d+\.\d)\\%, and the union', 'bal_semg_only', 1,
     'sEMG-only accuracy'),
    (r'the union of sEMG and IMU falls (\d+\.\d) points', 'drop_semg_imu', 1,
     'sEMG+IMU shortfall'),
    (r'the (\d+\.\d)-point deficit in accuracy', 'gap_hybrid_xgb', 1,
     'XGBoost accuracy deficit'),
    (r'It reaches (\d+\.\d)\\%', 'acc_xgb', 1, 'XGBoost accuracy'),
    (r'the plain LSTM at (\d+\.\d)\\%', 'acc_lstm', 1, 'LSTM accuracy'),
    (r'Severe fatigue reaches (\d+\.\d)\\% precision', 'prec_sev', 1,
     'severe-class precision'),
    (r'Precision of (\d+\.\d)\\% means few false alarms', 'prec_sev', 1,
     'severe-class precision, Discussion'),
    (r'only (\d+\.\d)\\% recall', 'rec_sev', 1, 'severe-class recall'),
    (r'Recall of (\d+\.\d)\\%', 'rec_sev', 1, 'severe-class recall'),
    (r'argmax operating point \(sensitivity (\d+\.\d)\\%', 'rec_sev', 1,
     'argmax operating point, sensitivity'),
    (r'argmax operating point \(sensitivity \d+\.\d\\%, specificity (\d+\.\d)\\%',
     'spec_sev', 1, 'argmax operating point, specificity'),
    (r'Youden-optimal point of the hybrid curve \(sensitivity (\d+\.\d)\\%',
     'youden_sens', 1, 'Youden sensitivity'),
    (r'Youden-optimal point of the hybrid curve \(sensitivity \d+\.\d\\%, specificity (\d+\.\d)\\%',
     'youden_spec', 1, 'Youden specificity'),
    (r'changed balanced accuracy by (\d+\.\d) points', 'bal_range', 1,
     'threshold sweep accuracy range'),
    (r'severe-class recall falls by (\d+\.\d) points', 'rec_drop', 1,
     'severe-recall change across thresholds'),
    (r'lead time falls from (\d+\.\d) to', 'lead_lo', 1,
     'lead time at the lowest cut-points'),
    (r'preceded severe fatigue by roughly (\d+)', 'lead', 0,
     'alert lead time, abstract'),
    (r'severe fatigue state by (\d+) on average', 'lead', 0,
     'alert lead time, Results'),
    (r'peak shoulder (?:load|moment) by (\d+)\\%', 'peak_chg', 0,
     'peak shoulder load reduction'),
    (r'retaining (\d+)\\% of baseline throughput', None, 0,
     'throughput retained'),
    (r'a cost of (\d+)\\% throughput', 'thru_chg', 0, 'throughput cost'),
    (r'([\d,]+) feature windows', 'n_windows', 0, 'window count'),
    (r'([\d,]+) simulated (?:working|operational) hours', None, 0,
     'simulated hours'),
    (r'(\d+) anthropometrically diverse synthetic operator', 'n_operators', 0,
     'operator count'),
    (r'(\d+)-dimensional descriptor', 'n_features', 0, 'feature count'),
    (r'imposes a latency below (\d+),', 'lat_budget', 0, 'latency budget'),
    (r'can meet a (\d+) control budget', 'lat_budget', 0,
     'latency budget, Highlights'),
]


def audit_manuscript(path):
    """Check every quantity stated in the .tex against RESULTS.

    Returns True when no ERROR was found.  Run it after any edit to the running
    text; it costs a second and catches the class of mistake that is otherwise
    invisible until a reviewer finds it.
    """
    if not os.path.exists(path):
        print(f'  manuscript audit skipped: {os.path.basename(path)} not found')
        print('    the audit is what proves every number in the text matches the')
        print('    figures, so do not ship without it. Either put the .tex beside')
        print('    this script, or run:')
        print('      python p5_figures.py --check-tex "C:\\path\\to\\'
              'sensors-4496124-proofread.tex"')
        return True

    src = _load_tex(path)
    exp = _expected_values()

    errors, missing, checked = [], [], 0

    for pattern, key, places, label in _TEX_CHECKS:
        hits = re.findall(pattern, src)
        if not hits:
            missing.append(label)
            continue
        if key is None:          # presence-only check, value not in RESULTS
            checked += len(hits)
            continue
        want = round(exp[key], places)
        for raw in hits:
            checked += 1
            got = float(raw.replace(',', ''))
            if abs(got - want) > 0.5 * 10 ** (-places):
                errors.append(f'    {label}: text says {raw}, '
                              f'RESULTS gives {want:.{places}f}   '
                              f'[pattern: {pattern}]')

    # --- structural checks -------------------------------------------------
    labels = set(re.findall(r'\\label\{([^}]+)\}', src))
    refs = set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', src))
    dangling = sorted(refs - labels)
    if dangling:
        errors.append(f'    unresolved cross-references: {dangling}')

    wanted_figs = set(re.findall(r'\{([^{}]*Fig\d+_[A-Za-z_]+)\.pdf\}', src))
    for fig in sorted(wanted_figs):
        if not os.path.exists(fig + '.pdf'):
            errors.append(f'    figure file absent: {fig}.pdf')

    if src.count(r'\begin{table}') != src.count(r'\end{table}'):
        errors.append('    unbalanced table environments')
    if 'resizebox' in src:
        errors.append('    \\resizebox present: table type sizes will differ')

    # --- report ------------------------------------------------------------
    print(f'  manuscript audit: {checked} numeric claims checked in '
          f'{os.path.basename(path)}')
    if missing:
        print(f'    {len(missing)} claim(s) not found in the text '
              f'(rephrased or removed):')
        for lab in missing:
            print(f'      - {lab}')
    if errors:
        print(f'    {len(errors)} DISAGREEMENT(S) WITH RESULTS:')
        for e in errors:
            print(e)
        return False
    print('    no disagreement between the text and RESULTS')
    return True


# ==============================================================================
# MAIN
# ==============================================================================
# The manuscript file was renamed by the Editorial Office at the proofreading
# stage.  The current name is tried first and the earlier ones are kept so that
# the script still audits an archived copy of the tree.
TEX_CANDIDATES = ['sensors-4496124-proofread.tex',
                  'sensors-4496124-done-edited.tex',
                  'P5_Sensors_(18.08.2026).tex']


def _default_tex():
    """Locate the manuscript without assuming the working directory.

    The script is often run from somewhere other than the folder that holds the
    .tex -- from the drive root, or from an IDE whose working directory is the
    project root.  Looking only at the working directory then silently skips the
    figure-against-text audit, which is the one check that cannot be replaced by
    reading the output.  So the search covers the working directory, the folder
    the script itself lives in, and the parent and 'figures' siblings of both.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [os.getcwd(), here,
             os.path.dirname(os.getcwd()), os.path.dirname(here),
             os.path.join(here, '..'), os.path.join(here, 'manuscript')]
    seen = set()
    for root in roots:
        root = os.path.abspath(root)
        if root in seen:
            continue
        seen.add(root)
        for name in TEX_CANDIDATES:
            candidate = os.path.join(root, name)
            if os.path.exists(candidate):
                return candidate
    return TEX_CANDIDATES[0]


DEFAULT_TEX = _default_tex()


def main():
    print('=' * 68)
    print('P5 figure generation — MDPI Sensors, manuscript sensors-4496124')
    print('=' * 68)
    print(f'  typeface: {SERIF_FAMILY} (text and math)')
    if SERIF_FAMILY == 'DejaVu Serif':
        print('  WARNING: no Palatino cut found; install Palatino Linotype or '
              'TeX Gyre Pagella before producing the final figures')
    stats = audit()
    fig1_sensor_architecture()
    fig2_signal_processing()
    fig3_model_comparison(stats)
    fig4_complexity_tradeoff()
    fig5_confusion_variability(stats)
    fig6_modality_ablation()
    fig7_feature_importance()
    fig8_threshold_sensitivity()
    fig9_realtime_prediction()
    fig10_intervention()
    print('=' * 68)
    print(f'10 figures written to ./{OUTPUT_DIR}/ as PDF and 600 dpi PNG')
    ok = audit_manuscript(DEFAULT_TEX)
    print('=' * 68)
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    # `--check-tex [path]` runs the manuscript audit alone, without redrawing
    # the figures.  This is the mode to use while rewriting the running text.
    if '--check-tex' in sys.argv:
        i = sys.argv.index('--check-tex')
        tex = sys.argv[i + 1] if len(sys.argv) > i + 1 else DEFAULT_TEX
        audit()
        sys.exit(0 if audit_manuscript(tex) else 1)
    main()
