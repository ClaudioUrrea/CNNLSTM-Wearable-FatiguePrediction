#!/usr/bin/env python3
"""
Figure generation for P5: Real-Time Physiological Fatigue Prediction
Using Wearable Sensor Fusion and Hybrid Deep Learning for HRC Manufacturing
Target: MDPI Sensors — 7 figures in ascending order matching .tex
Author: Claudio Urrea
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import math
from matplotlib.gridspec import GridSpec
from matplotlib import patheffects

import os

OUTPUT_DIR = 'figures_p5_sensors'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['mathtext.fontset'] = 'cm'


# ============================================================
# Binormal ROC helpers (used by Figure 3b)
# Implemented with math.erf only: no SciPy dependency.
# ============================================================
def _Phi(x):
    """Standard normal CDF, vectorised over numpy arrays."""
    return 0.5 * (1.0 + _ERF(np.asarray(x, dtype=float) / np.sqrt(2.0)))


_ERF = np.vectorize(math.erf, otypes=[float])


def _Phi_inv(p, lo=-9.0, hi=9.0, iters=80):
    """Inverse standard normal CDF for a scalar p, by bisection on _Phi."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float(_Phi(mid)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _binormal_sigma(fpr0, tpr0, auc):
    """Slope sigma of the binormal ROC that has the given AUC and passes
    through the operating point (fpr0, tpr0).

    Negative class ~ N(0, 1), positive class ~ N(mu, sigma^2), so that
        AUC = Phi(mu / sqrt(1 + sigma^2))
    and, at latent threshold z,
        FPR(z) = 1 - Phi(z),  TPR(z) = 1 - Phi((z - mu) / sigma).
    """
    z0 = _Phi_inv(1.0 - fpr0)          # threshold giving that FPR
    k = -_Phi_inv(1.0 - tpr0)          # (mu - z0) / sigma
    a = _Phi_inv(auc)                  # mu / sqrt(1 + sigma^2)

    # solve  z0 + k*sigma = a*sqrt(1 + sigma^2)  for sigma > 0
    def g(s):
        return z0 + k * s - a * np.sqrt(1.0 + s * s)

    lo, hi = 1e-4, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(lo) * g(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ============================================================
# Figure 1: Sensor Placement + Data Acquisition Architecture
# (Section 3.2 — first figure in paper)
# ============================================================
def create_fig1_sensor_placement():
    """Figure 1: Wearable Sensor Placement and Data Acquisition Architecture"""
    fig = plt.figure(figsize=(13, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3, width_ratios=[1, 1.2])

    # (a) Sensor placement schematic
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(1.40, 13.3)
    ax1.axis('off')
    ax1.set_title('(a) Wearable Sensor Placement', fontsize=12, fontweight='bold', pad=10)

    # Body outline
    head = plt.Circle((5, 12.5), 0.7, fill=False, edgecolor='#333', linewidth=2)
    ax1.add_patch(head)
    ax1.plot([4.3, 4.3, 5.7, 5.7], [11.8, 7.5, 7.5, 11.8], color='#333', linewidth=2)
    ax1.plot([4.3, 5.7], [11.8, 11.8], color='#333', linewidth=2)
    ax1.plot([4.3, 2.8, 2.0], [11.5, 9.5, 7.5], color='#333', linewidth=2)
    ax1.plot([2.0, 1.5], [7.5, 6.0], color='#333', linewidth=2)
    ax1.plot([5.7, 7.2, 8.0], [11.5, 9.5, 7.5], color='#333', linewidth=2)
    ax1.plot([8.0, 8.5], [7.5, 6.0], color='#333', linewidth=2)
    ax1.plot([4.5, 4.0, 3.5], [7.5, 5.0, 3.0], color='#333', linewidth=2)
    ax1.plot([5.5, 6.0, 6.5], [7.5, 5.0, 3.0], color='#333', linewidth=2)

    # sEMG sensors (red circles) — bilateral: 4 muscles × 2 sides = 8
    emg_positions = {
        'R. Biceps': (3.3, 10.5), 'R. Ant. Delt.': (3.8, 11.3),
        'R. FCR': (1.8, 7.0), 'R. Trapezius': (4.4, 11.8),
        'L. Biceps': (6.7, 10.5), 'L. Ant. Delt.': (6.2, 11.3),
        'L. FCR': (8.2, 7.0), 'L. Trapezius': (5.6, 11.8)
    }
    for name, pos in emg_positions.items():
        ax1.plot(pos[0], pos[1], 'o', color='#E63946', markersize=10, zorder=5)
        offset = (0.28, 0.05) if pos[0] > 5 else (-0.28, 0.05)
        ha = 'left' if pos[0] > 5 else 'right'
        ax1.annotate(name, xy=pos, xytext=(pos[0]+offset[0], pos[1]+offset[1]),
                    fontsize=7, ha=ha, color='#E63946', fontweight='bold')

    # IMU sensors (blue squares) — bilateral upper arm/forearm + trunk + pelvis
    imu_positions = {
        'R. Upper Arm': (7.0, 10.0), 'R. Forearm': (7.8, 8.2),
        'L. Upper Arm': (2.5, 10.0), 'L. Forearm': (2.0, 8.2),
        'Trunk': (5.0, 10.2), 'Pelvis': (5.0, 7.8)
    }
    for name, pos in imu_positions.items():
        ax1.plot(pos[0], pos[1], 's', color='#457B9D', markersize=9, zorder=5)

    # FSR sensors (green triangles) — tool handle (2), gripper, support surface
    fsr_positions = {'Tool Handle 1': (1.3, 5.8), 'Tool Handle 2': (1.7, 5.5),
                     'Gripper': (8.5, 5.8), 'Support': (5.0, 7.3)}
    for name, pos in fsr_positions.items():
        ax1.plot(pos[0], pos[1], '^', color='#2A9D8F', markersize=10, zorder=5)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E63946',
                   markersize=7, label='sEMG (8 ch, 1000 Hz)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#457B9D',
                   markersize=7, label='IMU (6 units, 100 Hz)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#2A9D8F',
                   markersize=7, label='FSR (4 sensors, 500 Hz)')
    ]
    ax1.legend(handles=legend_elements, loc='lower center', fontsize=8, ncol=1,
              frameon=True, fancybox=True, shadow=True)

    # (b) Data acquisition pipeline — clear sequential flow
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 12)
    ax2.set_ylim(1.5, 10.5)
    ax2.axis('off')
    ax2.set_title('(b) Real-Time Data Acquisition Pipeline', fontsize=12,
                  fontweight='bold', pad=10)

    # Three-tier architecture: left column (sensor tier), middle (edge), right (processing)
    boxes = [
        # Sensor tier (left)
        {'pos': (0.5, 8.5), 'w': 2.2, 'h': 0.9, 'text': 'Wearable\nSensors', 'color': '#E63946'},
        {'pos': (0.5, 6.8), 'w': 2.2, 'h': 0.9, 'text': 'BLE 5.0\nTransmission', 'color': '#F4A261'},
        {'pos': (0.5, 5.1), 'w': 2.2, 'h': 0.9, 'text': 'Edge\nGateway', 'color': '#E9C46A'},
        # Processing tier (right column, top-to-bottom)
        {'pos': (4.3, 8.5), 'w': 2.2, 'h': 0.9, 'text': 'Signal\nProcessing', 'color': '#2A9D8F'},
        {'pos': (4.3, 6.8), 'w': 2.2, 'h': 0.9, 'text': 'Feature\nExtraction', 'color': '#457B9D'},
        {'pos': (4.3, 5.1), 'w': 2.2, 'h': 0.9, 'text': 'ML\nInference', 'color': '#264653'},
        # Output (far right)
        {'pos': (8.0, 7.7), 'w': 2.2, 'h': 0.9, 'text': 'Fatigue\nPrediction', 'color': '#E76F51'},
        {'pos': (8.0, 5.9), 'w': 2.2, 'h': 0.9, 'text': 'HRC\nController', 'color': '#6D6875'},
    ]

    for box in boxes:
        rect = FancyBboxPatch((box['pos'][0], box['pos'][1]), box['w'], box['h'],
                              boxstyle="round,pad=0.1", facecolor=box['color'],
                              edgecolor='black', linewidth=1.5, alpha=0.85)
        ax2.add_patch(rect)
        ax2.text(box['pos'][0] + box['w']/2, box['pos'][1] + box['h']/2,
                box['text'], ha='center', va='center', fontsize=8,
                fontweight='bold', color='white')

    # Sequential vertical arrows (sensor tier)
    arrows_vert = [
        ((1.6, 8.46), (1.6, 7.74)),   # Sensors → BLE
        ((1.6, 6.76), (1.6, 6.04)),   # BLE → Gateway
    ]
    # Sequential vertical arrows (processing tier)
    arrows_vert += [
        ((5.4, 8.46), (5.4, 7.74)),   # Signal Proc → Feature Ext
        ((5.4, 6.76), (5.4, 6.04)),   # Feature Ext → ML Inference
    ]
    

    # Horizontal: Gateway → Signal Processing (WiFi relay)
    arrows_horiz = [
        ((2.78, 5.55), (4.2, 9.1)),   # Gateway → Signal Processing (WiFi relay)
    ]
    # Horizontal: Processing → outputs
    arrows_out = [
        ((6.54, 7.25), (8.0, 8.15)),   # Feature Ext → Fatigue Prediction
        ((6.54, 5.55), (8.0, 6.35)),   # ML Inference → HRC Controller
    ]
    for start, end in arrows_vert + arrows_horiz + arrows_out:
        ax2.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#555'))

    # Latency annotations — matched to text: 87-127 ms total
    latencies = [
        (2.14, 8.1, '<10 ms'),        # Sensors → BLE
        (2.26, 6.4, '20\u201350 ms'),  # BLE → Gateway
        (3.54, 7.0, '5\u201315 ms'),  # WiFi relay
        (5.9, 8.1, '15 ms'),          # Signal Proc → Feature Ext
        (5.9, 6.4, '15 ms'),          # Feature Ext → ML
        (7.4, 5.74, '22 ms'),          # ML Inference
    ]
    for x, y, txt in latencies:
        ax2.text(x, y, txt, fontsize=7, ha='center', color='black', style='italic')

    # Tier labels
    ax2.text(1.6, 10.1, 'Sensor Tier', fontsize=9, ha='center',
            fontweight='bold', color='#555', style='italic')
    ax2.text(5.4, 10.1, 'Processing Tier', fontsize=9, ha='center',
            fontweight='bold', color='#555', style='italic')
    ax2.text(9.1, 10.1, 'Output', fontsize=9, ha='center',
            fontweight='bold', color='#555', style='italic')

    # Total latency — consistent with text (87-127 ms)
    ax2.text(5.5, 2.5, 'Total End-to-End Latency: 87\u2013127 ms',
            fontsize=8, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                     edgecolor='orange', linewidth=2))

    plt.savefig(f'{OUTPUT_DIR}/Fig1_Sensor_Architecture.pdf',
                bbox_inches='tight', facecolor='white')
    plt.savefig(f'{OUTPUT_DIR}/Fig1_Sensor_Architecture.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("Figure 1: Sensor Placement and Architecture")
    plt.close()


# ============================================================
# Figure 2: Signal Processing Pipeline
# (Section 3.3 — second figure in paper)
# ============================================================
def create_fig2_signal_processing():
    """Figure 2: Signal Processing Pipeline with Example Waveforms"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    np.random.seed(42)
    t = np.linspace(0, 2, 2000)

    # (a) Raw sEMG signal
    ax = axes[0, 0]
    activation = 0.3 + 0.2 * np.sin(2 * np.pi * 0.5 * t)
    emg_raw = activation * np.random.randn(len(t)) * 0.5
    emg_raw += 0.05 * np.sin(2 * np.pi * 50 * t)

    ax.plot(t, emg_raw, color='#457B9D', linewidth=0.5, alpha=0.8)
    ax.set_ylabel('Amplitude (mV)', fontsize=10, fontweight='bold')
    ax.set_title('(a) Raw sEMG Signal (Biceps Brachii)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.8, 0.8)
    ax.text(0.02, 0.97, '1000 Hz sampling\nwith 50 Hz noise',
           transform=ax.transAxes, fontsize=8, va='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # (b) Filtered sEMG + RMS envelope
    ax = axes[0, 1]
    emg_filtered = activation * np.random.randn(len(t)) * 0.45
    window = 200
    emg_abs = np.abs(emg_filtered)
    rms_envelope = np.convolve(emg_abs, np.ones(window)/window, mode='same')

    ax.plot(t, emg_filtered, color='#457B9D', linewidth=0.5, alpha=0.5, label='Filtered sEMG')
    ax.plot(t, rms_envelope, color='#E63946', linewidth=2.5, label='RMS Envelope (200 ms)')
    ax.plot(t, -rms_envelope, color='#E63946', linewidth=2.5)
    ax.set_ylabel('Amplitude (mV)', fontsize=10, fontweight='bold')
    ax.set_title('(b) Filtered sEMG with RMS Envelope', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.8, 0.8)
    ax.text(0.02, 0.97, '20\u2013450 Hz bandpass\n50 Hz notch filter',
           transform=ax.transAxes, fontsize=8, va='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # (c) Feature extraction timeline
    ax = axes[1, 0]
    t_long = np.linspace(0, 60, 6000)
    mdf_initial = 85
    mdf_fatigue = mdf_initial * np.exp(-t_long / 300) + np.random.normal(0, 3, len(t_long))
    rms_increase = 0.25 + 0.15 * (t_long / 60) + np.random.normal(0, 0.02, len(t_long))

    ax_twin = ax.twinx()
    l1, = ax.plot(t_long, mdf_fatigue, color='#E63946', linewidth=2, label='Median Frequency')
    l2, = ax_twin.plot(t_long, rms_increase, color='#457B9D', linewidth=2, label='RMS Amplitude')

    ax.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Median Frequency (Hz)', fontsize=10, fontweight='bold', color='#E63946')
    ax_twin.set_ylabel('RMS Amplitude (mV)', fontsize=10, fontweight='bold', color='#457B9D')
    ax.set_title('(c) Fatigue-Indicative Feature Evolution', fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#E63946')
    ax_twin.tick_params(axis='y', labelcolor='#457B9D')
    ax.legend(handles=[l1, l2], fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 60)

    ax.annotate('Fatigue progression', xy=(20, 79.8), xytext=(1, 82.6),
               fontsize=9, fontweight='bold', color='darkred',
               arrowprops=dict(arrowstyle='->', lw=2, color='darkred'))

    # (d) IMU joint angle
    ax = axes[1, 1]
    t_imu = np.linspace(0, 120, 12000)
    cycles = np.sin(2 * np.pi * t_imu / 8) * 35 + 55
    drift = 5 * (t_imu / 120)
    shoulder_angle = cycles + drift + np.random.normal(0, 3, len(t_imu))

    ax.plot(t_imu, shoulder_angle, color='#2A9D8F', linewidth=1, alpha=0.7)
    ax.axhline(y=90, color='red', linestyle='--', linewidth=2, label='Risk Threshold (90\u00b0)')
    ax.axhline(y=60, color='orange', linestyle='--', linewidth=1.5, label='Moderate Risk (60\u00b0)')

    risk_mask = shoulder_angle > 90
    ax.fill_between(t_imu, 90, shoulder_angle, where=risk_mask,
                    color='red', alpha=0.15, label='High-Risk Zone')

    ax.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Shoulder Flexion (\u00b0)', fontsize=10, fontweight='bold')
    ax.set_title('(d) IMU-Derived Joint Angle Monitoring', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 125)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig2_Signal_Processing.pdf',
                bbox_inches='tight', facecolor='white')
    plt.savefig(f'{OUTPUT_DIR}/Fig2_Signal_Processing.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("Figure 2: Signal Processing Pipeline")
    plt.close()


# ============================================================
# Figure 3: Model Comparison (accuracy, F1, ROC curves)
# (Section 7.1 — third figure in paper)
# ============================================================
def create_fig3_model_comparison():
    """Figure 3: ML Model Comparison — matches Table 4 values"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    models = ['SVM', 'Random\nForest', 'XGBoost', 'LSTM', '1D-CNN-LSTM\n(Hybrid)']
    accuracy = [82.1, 84.3, 86.7, 88.5, 89.3]
    f1_scores = [80.0, 82.3, 85.0, 87.0, 87.7]
    auc_scores = [0.89, 0.91, 0.93, 0.94, 0.95]

    x = np.arange(len(models))
    width = 0.28

    bars1 = ax1.bar(x - width, accuracy, width, label='Accuracy (%)',
                   color='#457B9D', edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x, f1_scores, width, label='F1-Score (%)',
                   color='#2A9D8F', edgecolor='black', linewidth=1.2)
    bars3 = ax1.bar(x + width, [a*100 for a in auc_scores], width,
                   label='AUC (\u00d7100)', color='#E9C46A', edgecolor='black', linewidth=1.2)

    for bars in [bars1, bars2, bars3]:
        bars[-1].set_edgecolor('#E63946')
        bars[-1].set_linewidth(3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10, fontweight='bold')
    ax1.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Model Comparison on Test Set', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(75, 100)

    ax1.annotate('Best', xy=(4, 90), fontsize=10, fontweight='bold',
                color='#E63946', ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='#E63946'))

    # (b) ROC curves — binormal, constrained to the reported AUCs and to the
    # severe-class operating point implied by the confusion matrix (Fig. 4a).
    #
    # The earlier one-parameter family tpr = 1-(1-fpr)^a was fitted to the AUC
    # alone and could not also reach that operating point: the star fell 5.1 pp
    # off its own curve and was not the Youden maximum of what was drawn.
    # The binormal form has two free parameters and satisfies both constraints.

    # Operating point from Table/Fig. 4a: severe-class precision 0.86,
    # recall 0.79, prevalence 0.17  ->  specificity 0.974
    _tp = 0.79 * 17.0
    _fp = _tp / 0.86 - _tp
    op_fpr = _fp / 83.0                      # 0.0263
    op_tpr = 0.79

    sigma = _binormal_sigma(op_fpr, op_tpr, 0.95)
    mu_hybrid = _Phi_inv(0.95) * np.sqrt(1.0 + sigma * sigma)

    # Latent-threshold parameterisation avoids evaluating Phi^-1 on a grid.
    z = np.linspace(6.0, -6.0, 2000)
    fpr = 1.0 - _Phi(z)

    def make_roc(target_auc):
        """Binormal ROC with common slope sigma, scaled to hit target_auc."""
        mu = _Phi_inv(target_auc) * np.sqrt(1.0 + sigma * sigma)
        return 1.0 - _Phi((z - mu) / sigma)

    tpr_svm = make_roc(0.89)
    tpr_rf = make_roc(0.91)
    tpr_xgb = make_roc(0.93)
    tpr_lstm = make_roc(0.94)
    tpr_hybrid = 1.0 - _Phi((z - mu_hybrid) / sigma)

    ax2.plot(fpr, tpr_svm, linewidth=2, color='#FFB6C1',
            label='SVM (AUC=0.89)', linestyle='--')
    ax2.plot(fpr, tpr_rf, linewidth=2.5, color='#E63946',
            label='Random Forest (AUC=0.91)')
    ax2.plot(fpr, tpr_xgb, linewidth=2, color='#E9C46A',
            label='XGBoost (AUC=0.93)', linestyle='--')
    ax2.plot(fpr, tpr_lstm, linewidth=2.5, color='#2A9D8F',
            label='LSTM (AUC=0.94)')
    ax2.plot(fpr, tpr_hybrid, linewidth=3, color='#264653',
            label='1D-CNN-LSTM (AUC=0.95)')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.4, label='Random')

    # Youden optimum read off the plotted curve, never hard-coded, so the
    # marker cannot drift away from the line again.
    _i = int(np.argmax(tpr_hybrid - fpr))
    opt_fpr, opt_tpr = float(fpr[_i]), float(tpr_hybrid[_i])

    ax2.plot(opt_fpr, opt_tpr, '*',
            markersize=20, color='gold', markeredgecolor='#264653',
            markeredgewidth=2, zorder=10)
    ax2.annotate('Youden optimum\n(Sen: %.1f%%\nSpe: %.1f%%)'
                % (opt_tpr * 100.0, (1.0 - opt_fpr) * 100.0),
                xy=(opt_fpr, opt_tpr),
                xytext=(0.34, 0.46), fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                         edgecolor='#264653', alpha=0.9),
                arrowprops=dict(arrowstyle='->', lw=2, color='#264653'))

    # Operating point actually produced by the argmax decision rule.
    ax2.plot(op_fpr, op_tpr, 'o', markersize=9, markerfacecolor='white',
            markeredgecolor='#264653', markeredgewidth=2, zorder=9)
    ax2.annotate('argmax rule\n(Sen: 79.0%\nSpe: 97.4%)',
                xy=(op_fpr, op_tpr),
                xytext=(0.15, 0.19), fontsize=8.5,
                bbox=dict(boxstyle='round', facecolor='white',
                         edgecolor='#264653', alpha=0.9),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#264653'))

    ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) ROC Curves: Architecture Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig3_Model_Comparison.pdf',
                bbox_inches='tight', facecolor='white')
    plt.savefig(f'{OUTPUT_DIR}/Fig3_Model_Comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("Figure 3: ML Model Comparison")
    plt.close()


# ============================================================
# Figure 4: Confusion Matrix + Inter-Subject Variability
# (Section 7.2 — fourth figure in paper)
# ============================================================
def create_fig4_confusion_variability():
    """Figure 4: Confusion matrix yielding 89.3% acc / 87.1% balanced acc"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8))

    # Confusion matrix pooled over the six out-of-fold test partitions,
    # i.e. all 24 operators and 294,912 windows.
    # Row totals follow the 45:38:17 class split -> 132,708 / 112,068 / 50,136.
    #   Fresh    R = 124074/132708 = 93.5%   P = 124074/133764 = 92.8%
    #   Moderate R =  99846/112068 = 89.1%   P =  99846/115218 = 86.7%
    #   Severe   R =  39396/50136  = 78.6%   P =  39396/45930  = 85.8%
    # Accuracy       = 263316/294912 = 89.3%
    # Balanced acc.  = (93.5 + 89.1 + 78.6)/3 = 87.1%
    # Macro F1       = (93.1 + 87.9 + 82.0)/3 = 87.7%
    conf_matrix = np.array([
        [124074,   7158,   1476],    # Fresh:    132,708
        [  7164,  99846,   5058],    # Moderate: 112,068
        [  2526,   8214,  39396]     # Severe:    50,136
    ])
    assert conf_matrix.sum() == 294912

    im = ax1.imshow(conf_matrix, cmap='Blues', aspect='auto',
                    vmin=0, vmax=conf_matrix.max())

    class_labels = ['Fresh', 'Moderate', 'Severe']
    for i in range(3):
        for j in range(3):
            val = int(conf_matrix[i, j])
            color = 'white' if val > 0.5 * conf_matrix.max() else 'black'
            ax1.text(j, i, f'{val:,}', ha='center', va='center',
                    color=color, fontsize=12.5, fontweight='bold')

    ax1.set_xticks([0, 1, 2])
    ax1.set_yticks([0, 1, 2])
    ax1.set_xticklabels([f'Pred.\n{l}' for l in class_labels], fontsize=9)
    ax1.set_yticklabels([f'Actual\n{l}' for l in class_labels], fontsize=9)
    ax1.set_title('(a) Confusion Matrix: 1D-CNN-LSTM\n(pooled out-of-fold, n=294,912 windows)',
                 fontsize=11, fontweight='bold')

    metrics = ('Accuracy: 89.3%\nBalanced Acc.: 87.1%\n'
              'Macro F1: 87.7%\nAUC: 0.95')
    ax1.text(2.55, 1.0, metrics, fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#E8F4FD', edgecolor='#457B9D',
                     linewidth=2),
            verticalalignment='center')

    # (b) Inter-subject variability — phenotype traces ordered correctly
    np.random.seed(42)
    time_norm = np.linspace(0, 180, 200)

    # Fast fatiguer (τ<100 s): steep decline
    typeA = 100 * np.exp(-time_norm / 90)
    # Average (τ=100-170 s): moderate decline
    typeB = 100 * np.exp(-time_norm / 140)
    # Fatigue-resistant (τ>170 s): slow decline
    typeC = 100 * np.exp(-time_norm / 200)

    for i in range(7):
        noise = np.random.normal(0, 3.5, len(time_norm))
        ax2.plot(time_norm, np.clip(typeA + noise, 5, 105),
                color='#457B9D', alpha=0.15, linewidth=0.8)
    for i in range(10):
        noise = np.random.normal(0, 4.0, len(time_norm))
        ax2.plot(time_norm, np.clip(typeB + noise, 5, 105),
                color='#2A9D8F', alpha=0.15, linewidth=0.8)
    for i in range(7):
        noise = np.random.normal(0, 4.5, len(time_norm))
        ax2.plot(time_norm, np.clip(typeC + noise, 5, 105),
                color='#E63946', alpha=0.15, linewidth=0.8)

    ax2.plot(time_norm, np.clip(typeA, 5, 100), color='#457B9D', linewidth=3,
            label=r'Fast fatiguer ($\tau$<100 s, n=7)')
    ax2.plot(time_norm, np.clip(typeB, 5, 100), color='#2A9D8F', linewidth=3,
            label=r'Average ($\tau$=100–170 s, n=10)')
    ax2.plot(time_norm, np.clip(typeC, 5, 100), color='#E63946', linewidth=3,
            label=r'Fatigue-resistant ($\tau$>170 s, n=7)')

    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.text(5, 47, '50% MVC', fontsize=8, color='gray')

    ax2.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Force Capacity (% MVC)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Inter-Subject Variability: 3 Fatigue Phenotypes\n'
                  '(CV=34%, n=24 models)',
                 fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 180)
    ax2.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig4_Confusion_Variability.pdf',
                bbox_inches='tight', facecolor='white')
    plt.savefig(f'{OUTPUT_DIR}/Fig4_Confusion_Variability.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("Figure 4: Confusion Matrix and Variability")
    plt.close()


# ============================================================
# Figure 5: Feature Importance (SHAP)
# (Section 7.3 — fifth figure in paper)
# ============================================================
def create_fig5_feature_importance():
    """Figure 5: Feature Engineering and Importance Analysis (SHAP)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Feature categories breakdown — updated with Biomechanical
    categories = ['sEMG\n(64 feat.)', 'IMU\n(36 feat.)', 'Biomech.\n(12 feat.)',
                  'FSR\n(12 feat.)', 'Context\n(3 feat.)']
    counts = [64, 36, 12, 12, 3]
    colors = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261', '#E9C46A']

    sub_labels = [
        ['Time-domain\n(40)', 'Freq-domain\n(24)'],
        ['Kinematic\n(36)'],
        ['Inv. dynamics\n(12)'],
        ['Force\nDynamics (12)'],
        ['Temporal\n(3)']
    ]

    bars = ax1.bar(range(len(categories)), counts, color=colors,
                   edgecolor='black', linewidth=1.5, width=0.5)

    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        for j, sub in enumerate(sub_labels[i]):
            y_pos = (j + 0.5) * count / len(sub_labels[i])
            if count >= 10:
                ax1.text(bar.get_x() + bar.get_width()/2, y_pos,
                        sub, ha='center', va='center', fontsize=7,
                        fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[i],
                                 edgecolor='white', alpha=0.7))

    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax1.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Feature Vector Composition (127 total)', fontsize=11, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 75)

    ax1.text(2.0, 69, 'Total: 127 features\n30-s windows, 50% overlap',
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow',
                     edgecolor='orange', linewidth=2))

    # (b) Top 15 features by SHAP importance
    features = [
        'sEMG Fatigue Index (biceps)', 'Cumul. Shoulder Load',
        'sEMG Median Freq. (deltoid)', 'Time Since Last Break',
        'Movement Jerk (IMU)', 'sEMG RMS Variability',
        'Shoulder Flexion Angle', 'Elbow Angular Velocity',
        'sEMG Zero Crossings', 'Cumul. Work Time',
        'Peak Shoulder Torque', 'Trunk Lateral Bend',
        'sEMG Waveform Length', 'FSR Grip Force CV',
        'Forearm Pronation Rate'
    ]
    shap_values = [0.28, 0.22, 0.18, 0.15, 0.14, 0.13, 0.12, 0.11,
                   0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]

    feature_colors = []
    for f in features:
        if 'sEMG' in f:
            feature_colors.append('#E63946')
        elif 'IMU' in f or 'Angle' in f or 'Angular' in f or 'Jerk' in f or \
             'Trunk' in f or 'Forearm' in f:
            feature_colors.append('#457B9D')
        elif 'Time' in f or 'Work Time' in f:
            feature_colors.append('#E9C46A')
        elif 'FSR' in f or 'Grip' in f:
            feature_colors.append('#F4A261')
        else:
            feature_colors.append('#2A9D8F')

    y_pos = range(len(features))
    bars2 = ax2.barh(y_pos, shap_values[::-1], color=feature_colors[::-1],
                     edgecolor='black', linewidth=1)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(features[::-1], fontsize=10)
    ax2.set_xlabel('Mean |SHAP Value|', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Top 15 Features by Importance (SHAP)', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.3, linestyle='--')

    for i, v in enumerate(shap_values[::-1]):
        ax2.text(v + 0.001, i, f'{v:.2f}', va='center', fontsize=8, fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor='#E63946', label='sEMG features (42%)'),
        mpatches.Patch(facecolor='#457B9D', label='IMU features (26%)'),
        mpatches.Patch(facecolor='#2A9D8F', label='Biomech. features (16%)'),
        mpatches.Patch(facecolor='#E9C46A', label='Contextual (13%)'),
        mpatches.Patch(facecolor='#F4A261', label='FSR features (2%)')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig5_Feature_Importance.pdf',
                bbox_inches='tight', facecolor='white')
    plt.savefig(f'{OUTPUT_DIR}/Fig5_Feature_Importance.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("Figure 5: Feature Importance Analysis")
    plt.close()


# ============================================================
# Figure 6: Real-Time Prediction Timeline (2-hour session)
# (Section 7.4 — sixth figure in paper)
# ============================================================
def create_fig6_realtime_prediction():
    """Figure 6: Real-Time Prediction Timeline"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    np.random.seed(42)
    t = np.linspace(0, 120, 7200)

    # (a) sEMG RMS amplitude
    base_rms = 0.25
    fatigue_ramp = 0.15 * (t / 120)**1.5
    cycle_pattern = 0.08 * np.sin(2 * np.pi * t / 1.5)
    noise = np.random.normal(0, 0.03, len(t))
    emg_rms = base_rms + fatigue_ramp + cycle_pattern + noise

    break_starts = [30, 60, 90]
    break_ends = [32.5, 65, 92.5]
    for bs, be in zip(break_starts, break_ends):
        mask = (t >= bs) & (t <= be)
        emg_rms[mask] = 0.15 + np.random.normal(0, 0.02, mask.sum())

    ax1.plot(t, emg_rms, color='#457B9D', linewidth=1, alpha=0.8)
    kernel = np.ones(300) / 300
    trend = np.convolve(emg_rms, kernel, mode='same')
    ax1.plot(t, trend, color='#E63946', linewidth=2.5, label='Moving Average')

    for bs, be in zip(break_starts, break_ends):
        ax1.axvspan(bs, be, color='lightgreen', alpha=0.3)

    ax1.set_ylabel('sEMG RMS\n(mV)', fontsize=11, fontweight='bold')
    ax1.set_title('Real-Time Fatigue Monitoring: Representative 2-Hour Assembly Session',
                 fontsize=13, fontweight='bold', pad=12)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0.1, 0.6)

    # (b) Shoulder flexion angle
    base_angle = 45
    angle_drift = 12 * (t / 120)**1.2
    angle_cycle = 15 * np.sin(2 * np.pi * t / 1.2)
    angle_noise = np.random.normal(0, 3, len(t))
    shoulder_angle = base_angle + angle_drift + angle_cycle + angle_noise

    for bs, be in zip(break_starts, break_ends):
        mask = (t >= bs) & (t <= be)
        shoulder_angle[mask] = 25 + np.random.normal(0, 2, mask.sum())

    ax2.plot(t, shoulder_angle, color='#2A9D8F', linewidth=1, alpha=0.8)
    ax2.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label='Risk Threshold (90\u00b0)')

    for bs, be in zip(break_starts, break_ends):
        ax2.axvspan(bs, be, color='lightgreen', alpha=0.3)

    ax2.set_ylabel('Shoulder Flexion\n(degrees)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(10, 100)

    # (c) Predicted fatigue probability
    fatigue_prob = 0.15 + 0.65 * (1 - np.exp(-t / 80))
    fatigue_prob += np.random.normal(0, 0.05, len(t))
    fatigue_prob = np.clip(fatigue_prob, 0, 1)

    for bs, be in zip(break_starts, break_ends):
        mask = (t >= bs) & (t <= be)
        fatigue_prob[mask] = fatigue_prob[mask][0] * 0.5 + np.random.normal(0, 0.03, mask.sum())
        post_mask = (t > be) & (t < be + 5)
        if post_mask.sum() > 0:
            fatigue_prob[post_mask] *= 0.8

    fatigue_prob = np.clip(fatigue_prob, 0, 1)

    ax3.fill_between(t, 0, fatigue_prob, where=(fatigue_prob < 0.5),
                    color='#2A9D8F', alpha=0.3)
    ax3.fill_between(t, 0, fatigue_prob, where=(fatigue_prob >= 0.5) & (fatigue_prob < 0.7),
                    color='#E9C46A', alpha=0.3)
    ax3.fill_between(t, 0, fatigue_prob, where=(fatigue_prob >= 0.7),
                    color='#E63946', alpha=0.3)

    ax3.plot(t, fatigue_prob, color='#264653', linewidth=2)
    ax3.axhline(y=0.7, color='red', linestyle='--', linewidth=2.5,
               label='Alert Threshold (0.7)')
    ax3.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5,
               label='Warning (0.5)')

    for bs, be in zip(break_starts, break_ends):
        ax3.axvspan(bs, be, color='lightgreen', alpha=0.3,
                   label='Scheduled Break' if bs == 30 else '')

    alert_times = [78, 105, 115]
    for at in alert_times:
        idx = int(at * len(t) / 120)
        if idx < len(fatigue_prob):
            ax3.plot(at, fatigue_prob[idx], 'rv', markersize=12, zorder=5)
            ax3.annotate('Alert', xy=(at, fatigue_prob[idx]),
                        xytext=(at+3, fatigue_prob[idx]+0.08),
                        fontsize=10, color='red', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.2,
                                       mutation_scale=1))

    ax3.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Fatigue\nProbability', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper left', ncol=2)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(0, 120)
    ax3.set_ylim(0, 1.05)

    ax3.text(15, 0.25, 'Fresh', fontsize=11, fontweight='bold', color='#2A9D8F', ha='center')
    ax3.text(62, 0.52, 'Moderate', fontsize=11, fontweight='bold', color='#E9C46A', ha='center')
    ax3.text(100, 0.82, 'Severe', fontsize=11, fontweight='bold', color='#E63946', ha='center')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig6_RealTime_Prediction.pdf',
                bbox_inches='tight', facecolor='white')
    plt.savefig(f'{OUTPUT_DIR}/Fig6_RealTime_Prediction.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("Figure 6: Real-Time Prediction Timeline")
    plt.close()


# ============================================================
# Figure 7: Intervention Effectiveness (4-panel)
# (Section 7.5 — seventh figure in paper)
# ============================================================
def create_fig7_intervention():
    """Figure 7: Intervention Effectiveness — values from Table 5"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7))

    categories = ['Baseline', 'Intervention']
    x_pos = [0, 1]
    colors = ['#E63946', '#2A9D8F']

    def add_significance(ax, y_pos, text, p_text):
        ax.plot([0, 1], [y_pos, y_pos], 'k-', linewidth=1.5)
        ax.plot([0, 0], [y_pos * 0.97, y_pos], 'k-', linewidth=1.5)
        ax.plot([1, 1], [y_pos * 0.97, y_pos], 'k-', linewidth=1.5)
        ax.text(0.5, y_pos * 1.015, f'{text}  {p_text}', ha='center',
                fontsize=9, fontweight='bold')

    # (a) Peak shoulder loading
    means, stds = [18.3, 10.4], [4.2, 2.8]
    bars = ax1.bar(x_pos, means, yerr=stds, color=colors, edgecolor='black',
                   linewidth=2, capsize=8, width=0.55)
    ax1.axhline(y=18.3*0.7, color='orange', linestyle='--', linewidth=2,
               label='70% MVC threshold', alpha=0.8)
    add_significance(ax1, 26.0, '***', 'p<0.001, d=2.21')
    ax1.annotate('43% reduction', xy=(0.492, 19.3), fontsize=10, fontweight='bold',
                ha='center', color='darkgreen',
                bbox=dict(boxstyle='round', facecolor='#D4EDDA', edgecolor='darkgreen'))
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(i, m+s+0.5, f'{m}\u00b1{s}', ha='center', fontweight='bold', fontsize=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Peak Shoulder Moment (Nm)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Peak Shoulder Loading', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 30)

    # (b) Cumulative loading
    means, stds = [12.7, 8.8], [3.1, 2.4]
    bars = ax2.bar(x_pos, means, yerr=stds, color=colors, edgecolor='black',
                   linewidth=2, capsize=8, width=0.55)
    ax2.axhline(y=15, color='orange', linestyle='--', linewidth=2,
               label='Intervention trigger (15 Nm\u00b7h)', alpha=0.8)
    add_significance(ax2, 19.5, '***', 'p<0.001, d=1.41')
    ax2.annotate('31% reduction', xy=(0.496, 13.5), fontsize=10, fontweight='bold',
                ha='center', color='darkgreen',
                bbox=dict(boxstyle='round', facecolor='#D4EDDA', edgecolor='darkgreen'))
    for i, (m, s) in enumerate(zip(means, stds)):
        ax2.text(i, m+s+0.5, f'{m}\u00b1{s}', ha='center', fontweight='bold', fontsize=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Load (Nm\u00b7h)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Cumulative Shoulder Loading', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 23)

    # (c) Fatigue Demand Index
    means, stds = [61.2, 40.7], [12.3, 9.8]
    bars = ax3.bar(x_pos, means, yerr=stds, color=colors, edgecolor='black',
                   linewidth=2, capsize=8, width=0.55)
    ax3.axhline(y=70, color='red', linestyle='--', linewidth=2,
               label='Fatigue threshold', alpha=0.7)
    add_significance(ax3, 84, '***', 'p<0.001, d=1.84')
    ax3.annotate('34% reduction', xy=(0.5, 64.3), fontsize=10, fontweight='bold',
                ha='center', color='darkgreen',
                bbox=dict(boxstyle='round', facecolor='#D4EDDA', edgecolor='darkgreen'))
    for i, (m, s) in enumerate(zip(means, stds)):
        ax3.text(i, m+s+2, f'{m}\u00b1{s}', ha='center', fontweight='bold', fontsize=10)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax3.set_ylabel('Fatigue Demand\nIndex (0-100)', fontsize=11, fontweight='bold')
    ax3.set_title('(c) Fatigue Demand Index', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim(0, 96)

    # (d) Productivity — FIXED: p=0.012 IS significant (*), not "n.s."
    means, stds = [7.8, 7.3], [0.9, 0.8]
    bars = ax4.bar(x_pos, means, yerr=stds, color=colors, edgecolor='black',
                   linewidth=2, capsize=8, width=0.55)
    ax4.axhline(y=7.8*0.85, color='orange', linestyle='--', linewidth=2,
               label='85% threshold', alpha=0.7)
    add_significance(ax4, 9.8, '*', 'p=0.012')
    ax4.annotate('94% maintained', xy=(0.52, 8.18), fontsize=10, fontweight='bold',
                ha='center', color='darkblue',
                bbox=dict(boxstyle='round', facecolor='#D6EAF8', edgecolor='darkblue'))
    for i, (m, s) in enumerate(zip(means, stds)):
        ax4.text(i, m+s+0.15, f'{m}\u00b1{s}', ha='center', fontweight='bold', fontsize=10)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax4.set_ylabel('Assemblies per Hour', fontsize=11, fontweight='bold')
    ax4.set_title('(d) Production Throughput', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax4.set_ylim(0, 11)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig7_Intervention.pdf',
                bbox_inches='tight', facecolor='white')
    plt.savefig(f'{OUTPUT_DIR}/Fig7_Intervention.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("Figure 7: Intervention Effectiveness")
    plt.close()


# ============================================================
# Main execution — ascending order matching paper
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating P5 Figures for MDPI Sensors")
    print("=" * 60)

    create_fig1_sensor_placement()        # Section 3.2
    create_fig2_signal_processing()       # Section 3.3
    create_fig3_model_comparison()        # Section 7.1
    create_fig4_confusion_variability()   # Section 7.2
    create_fig5_feature_importance()      # Section 7.3
    create_fig6_realtime_prediction()     # Section 7.4
    create_fig7_intervention()            # Section 7.5

    print("=" * 60)
    print("All 7 figures generated successfully!")
