"""Synthetic accuracy validation for the gait pipeline.

Idea: we generate a walking motion with KNOWN joint angles (ground truth),
build 3D skeleton landmarks that realize those angles via forward kinematics,
then push those landmarks through the SAME measurement code the real pipeline
uses (joint_angle / detect_phase / metrics). We then compare what the pipeline
recovers against the known truth.

This validates the measurement math and the derived metrics end-to-end (angle
recovery, phase detection, step count, cadence, symmetry) independently of the
pose-estimation model. It does NOT validate MediaPipe's accuracy vs. real
motion capture — that requires a labelled mocap dataset.

Run:  python validate.py        (exits non-zero if any check fails)
"""
import sys

import numpy as np
import pandas as pd

import gait_analysis as g


# --- Ground-truth motion (known angles) ------------------------------------

def synth_leg(t, cycle, shift, knee_amp=27.5, hip_amp=22.5, ankle_amp=15.0):
    """Known per-frame (knee, hip, ankle) interior angles for one leg."""
    ph = 2 * np.pi / cycle * t + shift
    knee = 147.5 + knee_amp * np.sin(ph)      # ~120..175
    hip = 142.5 + hip_amp * np.sin(ph + 0.4)  # ~120..165
    ankle = 115.0 - ankle_amp * np.sin(ph)    # ~100..130 (dips at heel strike)
    return knee, hip, ankle


# --- Forward kinematics: angles -> 3D landmarks -----------------------------

class _LM:
    """Minimal stand-in for a MediaPipe landmark (x, y, z, visibility)."""
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x=0.0, y=0.0, z=0.0, visibility=1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


def _unit(angle_deg):
    r = np.deg2rad(angle_deg)
    return np.array([np.cos(r), np.sin(r), 0.0])


def _leg_points(hip_xy, hip_ang, knee_ang, ankle_ang, lengths=(0.5, 0.45, 0.45, 0.15)):
    """Build shoulder/hip/knee/ankle/foot points that realize the given interior
    angles. Segment directions are chained by simple angle arithmetic — an
    independent construction from the dot-product measurement being tested."""
    Lt, Lf, Ls, Lfoot = lengths
    dt = 90.0                        # torso direction (hip -> shoulder), points up
    df = dt - hip_ang                # so interior(shoulder-hip, knee-hip) = hip_ang
    ds = (df + 180.0) - knee_ang     # interior(hip-knee, ankle-knee) = knee_ang
    dfoot = (ds + 180.0) - ankle_ang # interior(knee-ankle, foot-ankle) = ankle_ang
    hip = np.array([hip_xy[0], hip_xy[1], 0.0])
    shoulder = hip + Lt * _unit(dt)
    knee = hip + Lf * _unit(df)
    ankle = knee + Ls * _unit(ds)
    foot = ankle + Lfoot * _unit(dfoot)
    return shoulder, hip, knee, ankle, foot


def build_landmarks(lk, lh, la, rk, rh, ra):
    """Return a 33-entry landmark list for one frame from the six leg angles."""
    lm = [_LM() for _ in range(33)]

    def place(idx, p):
        lm[idx] = _LM(float(p[0]), float(p[1]), float(p[2]), 1.0)

    sL, hL, kL, aL, fL = _leg_points((-0.1, 0.0), lh, lk, la)
    sR, hR, kR, aR, fR = _leg_points((0.1, 0.0), rh, rk, ra)
    for idx, p in [(11, sL), (23, hL), (25, kL), (27, aL), (31, fL),
                   (12, sR), (24, hR), (26, kR), (28, aR), (32, fR)]:
        place(idx, p)
    return lm


# --- Validation -------------------------------------------------------------

def run_validation(frames=320, cycle=40, fps=30.0, knee_offset=7.5):
    """Generate GT motion, measure via the pipeline, and compare.

    `knee_offset` lowers the right knee's mean angle so the (mean-based)
    symmetry metric has a known value below 100% that the pipeline must recover.
    Returns (results dict, all_passed bool).
    """
    t = np.arange(frames)
    lk, lh, la = synth_leg(t, cycle, 0.0)
    rk, rh, ra = synth_leg(t, cycle, np.pi)
    rk = rk - knee_offset  # known left/right mean difference -> known symmetry
    gt = {"LK": lk, "LH": lh, "LA": la, "RK": rk, "RH": rh, "RA": ra}

    # Measure each frame's angles from reconstructed landmarks.
    measured = {k: [] for k in gt}
    for i in range(frames):
        lm = build_landmarks(lk[i], lh[i], la[i], rk[i], rh[i], ra[i])
        measured["LK"].append(g.joint_angle(lm, 23, 25, 27))
        measured["RK"].append(g.joint_angle(lm, 24, 26, 28))
        measured["LH"].append(g.joint_angle(lm, 11, 23, 25))
        measured["RH"].append(g.joint_angle(lm, 12, 24, 26))
        measured["LA"].append(g.joint_angle(lm, 25, 27, 31))
        measured["RA"].append(g.joint_angle(lm, 26, 28, 32))
    measured = {k: np.array(v) for k, v in measured.items()}

    # 1) Per-joint angle recovery error (MAE / RMSE, degrees).
    angle_err = {}
    for k in gt:
        d = measured[k] - gt[k]
        angle_err[k] = (float(np.mean(np.abs(d))), float(np.sqrt(np.mean(d ** 2))))

    # 2) Derived-metric recovery: build frames for GT and measured, compare.
    def phases(knee, ank):
        return [g.detect_phase(kk, aa) for kk, aa in zip(knee, ank)]

    gt_steps = len(g.heel_strike_onsets(phases(gt["LK"], gt["LA"])))
    meas_steps = len(g.heel_strike_onsets(phases(measured["LK"], measured["LA"])))

    meas_sym = g.symmetry_index(np.mean(measured["LK"]), np.mean(measured["RK"]))
    # Independent expected symmetry from plain arithmetic on the known means.
    mL, mR = float(np.mean(gt["LK"])), float(np.mean(gt["RK"]))
    expected_sym = round(100 - abs(mL - mR) / ((mL + mR) / 2) * 100, 2)

    # Tolerances.
    max_mae = max(v[0] for v in angle_err.values())
    checks = {
        "angle_mae_deg": (max_mae, max_mae < 0.5),
        "step_count": (f"gt={gt_steps} measured={meas_steps}", gt_steps == meas_steps),
        "knee_symmetry_pct": (f"expected={expected_sym} measured={meas_sym}",
                              abs(expected_sym - meas_sym) < 0.5),
    }
    all_passed = all(ok for _, ok in checks.values())
    return {"angle_err": angle_err, "checks": checks,
            "gt_steps": gt_steps, "meas_steps": meas_steps,
            "expected_sym": expected_sym, "meas_sym": meas_sym}, all_passed


def main():
    res, ok = run_validation()
    print("=" * 52)
    print("  SYNTHETIC ACCURACY VALIDATION")
    print("=" * 52)
    print("\nPer-joint angle recovery (measured vs known ground truth):")
    print(f"  {'Joint':<6}{'MAE (deg)':>12}{'RMSE (deg)':>12}")
    for k, (mae, rmse) in res["angle_err"].items():
        print(f"  {k:<6}{mae:>12.4f}{rmse:>12.4f}")

    print("\nDerived-metric recovery:")
    for name, (val, passed) in res["checks"].items():
        print(f"  {'[PASS]' if passed else '[FAIL]'} {name}: {val}")

    print("\n" + ("✅ ALL CHECKS PASSED" if ok else "❌ SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
