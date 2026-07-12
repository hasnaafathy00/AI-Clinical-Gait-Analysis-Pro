"""Unit tests for the pure gait-analysis logic (no video / MediaPipe needed).

Run with:   pytest         (or)   python -m pytest
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gait_analysis as g


# --- geometry --------------------------------------------------------------

def test_angle_right():
    assert g.calculate_angle_3d([1, 0, 0], [0, 0, 0], [0, 1, 0]) == 90.0

def test_angle_straight():
    assert g.calculate_angle_3d([1, 0, 0], [0, 0, 0], [-1, 0, 0]) == 180.0

def test_angle_degenerate_is_nan():
    assert np.isnan(g.calculate_angle_3d([0, 0, 0], [0, 0, 0], [1, 0, 0]))


# --- symmetry --------------------------------------------------------------

def test_symmetry_equal():
    assert g.symmetry_index(160, 160) == 100.0

def test_symmetry_partial():
    assert g.symmetry_index(150, 160) == 93.55

def test_symmetry_div_zero_is_nan():
    assert np.isnan(g.symmetry_index(0, 0))


# --- phases and steps ------------------------------------------------------

def test_phase_unknown_on_nan():
    assert g.detect_phase(np.nan, 100) == "Unknown"

def test_phase_swing():
    assert g.detect_phase(120, 100) == "Swing Phase"

def test_heel_strike_onsets_counts_rising_edges():
    phases = ["Stance", "Heel Strike", "Heel Strike", "Swing Phase", "Heel Strike"]
    assert g.heel_strike_onsets(phases) == [1, 4]

def test_stance_swing_pct():
    assert g.stance_swing_pct(["Stance Phase", "Stance Phase", "Swing Phase"]) == (66.7, 33.3)

def test_stance_swing_all_unknown_is_nan():
    s, w = g.stance_swing_pct(["Unknown", "Unknown"])
    assert np.isnan(s) and np.isnan(w)


# --- cycle normalization ---------------------------------------------------

def test_normalize_cycles_shape():
    sig = list(np.sin(np.linspace(0, 6 * np.pi, 90)) * 10 + 150)
    cycles = g.normalize_cycles(sig, [0, 30, 60, 89])
    assert cycles.shape == (3, 101)

def test_ensemble_average_empty():
    assert g.ensemble_average(g.normalize_cycles([1, 2, 3], [0])) == (None, None)
