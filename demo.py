"""Demo driver — generate a realistic synthetic walking signal and run the
full analysis/reporting pipeline WITHOUT needing a real video or MediaPipe.

This proves the metrics, symmetry, gait-cycle normalization, graphs, PDF, and
CSV all work end-to-end. The numbers are synthetic, not clinical.

Usage:
    python demo.py --out ./results
"""
import argparse

import numpy as np
import pandas as pd

import gait_analysis as g


def synth_leg(t, cycle, phase_shift=0.0, knee_amp=27.5, hip_amp=22.5, ankle_amp=15.0):
    """Return (knee, hip, ankle) angle arrays for one leg over frames t."""
    w = 2 * np.pi / cycle
    ph = w * t + phase_shift
    knee = 147.5 + knee_amp * np.sin(ph)      # ~120 (swing) .. ~175 (stance)
    hip = 142.5 + hip_amp * np.sin(ph + 0.4)  # ~120 .. ~165
    # Ankle dips (dorsiflexion) as the knee extends, so heel strike registers
    # when knee > 165 and ankle < 110 — the pattern of real gait.
    ankle = 115.0 - ankle_amp * np.sin(ph)    # ~100 at heel strike .. ~130
    return knee, hip, ankle


def main():
    p = argparse.ArgumentParser(description="Generate a synthetic gait demo and run the pipeline.")
    p.add_argument("--out", default="./results", help="Output directory (default: ./results).")
    p.add_argument("--name", default="DemoPatient", help="Patient name for the report.")
    p.add_argument("--frames", type=int, default=300, help="Number of frames to synthesize.")
    p.add_argument("--fps", type=float, default=30.0, help="Frames per second.")
    args = p.parse_args()

    t = np.arange(args.frames)
    cycle = 40  # frames per gait cycle

    lk, lh, la = synth_leg(t, cycle, phase_shift=0.0)
    # Right leg is half a cycle out of phase, with a small asymmetry so the
    # symmetry score lands realistically below 100%.
    rk, rh, ra = synth_leg(t, cycle, phase_shift=np.pi, knee_amp=25.0, hip_amp=21.0)

    phase_l = [g.detect_phase(k, a) for k, a in zip(lk, la)]
    phase_r = [g.detect_phase(k, a) for k, a in zip(rk, ra)]

    df = pd.DataFrame({
        'F': t + 1,
        'LK': lk, 'RK': rk, 'LH': lh, 'RH': rh, 'LA': la, 'RA': ra,
        'Phase_L': phase_l, 'Phase_R': phase_r,
    })

    print(f"🧪 Generated {args.frames} synthetic frames ({args.frames / args.fps:.1f}s). Running pipeline...")
    g.generate_outputs(df, args.fps, args.out, args.name, "30", display=False)


if __name__ == "__main__":
    main()
