"""Wraps the synthetic accuracy harness as automated assertions."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import validate


def test_synthetic_validation_passes():
    res, ok = validate.run_validation()
    assert ok, res["checks"]


def test_angle_recovery_is_near_exact():
    res, _ = validate.run_validation()
    for joint, (mae, rmse) in res["angle_err"].items():
        assert mae < 1e-6, f"{joint} MAE too high: {mae}"


def test_known_symmetry_recovered():
    res, _ = validate.run_validation()
    assert abs(res["expected_sym"] - res["meas_sym"]) < 0.5
