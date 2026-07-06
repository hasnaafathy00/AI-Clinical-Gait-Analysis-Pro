"""Pre/post comparison — compare two gait_data.csv exports side by side.

Run gait_analysis.py on the 'before' and 'after' videos first, then:

    python compare.py before_gait_data.csv after_gait_data.csv --out ./results
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gait_analysis as g

JOINTS = [("Hip", "LH", "RH"), ("Knee", "LK", "RK"), ("Ankle", "LA", "RA")]


def summarize(df):
    """Overall symmetry + per-joint symmetry for one session."""
    sym = {name: g.symmetry_index(df[l].mean(), df[r].mean()) for name, l, r in JOINTS}
    valid = [v for v in sym.values() if np.isfinite(v)]
    sym["Overall"] = round(sum(valid) / len(valid), 2) if valid else float("nan")
    return sym


def main():
    p = argparse.ArgumentParser(description="Compare two gait sessions (before vs after).")
    p.add_argument("before", help="gait_data.csv from the first/baseline session.")
    p.add_argument("after", help="gait_data.csv from the second/follow-up session.")
    p.add_argument("--out", default="./results", help="Output directory (default: ./results).")
    args = p.parse_args()

    for path in (args.before, args.after):
        if not os.path.isfile(path):
            sys.exit(f"❌ File not found: {path}")
    os.makedirs(args.out, exist_ok=True)

    before, after = pd.read_csv(args.before), pd.read_csv(args.after)
    s_before, s_after = summarize(before), summarize(after)

    # Text report
    report_path = os.path.join(args.out, "Comparison_Report.txt")
    with open(report_path, "w") as f:
        f.write(f"GAIT COMPARISON REPORT\n{'=' * 30}\n")
        f.write(f"Before: {os.path.basename(args.before)}\n")
        f.write(f"After : {os.path.basename(args.after)}\n\n")
        f.write(f"{'Metric':<10}{'Before':>10}{'After':>10}{'Change':>10}\n")
        for key in list(s_before):
            b, a = s_before[key], s_after[key]
            delta = round(a - b, 2) if np.isfinite(a) and np.isfinite(b) else float("nan")
            fmt = lambda x: "n/a" if not np.isfinite(x) else f"{x}%"
            sign = "+" if np.isfinite(delta) and delta >= 0 else ""
            dtxt = "n/a" if not np.isfinite(delta) else f"{sign}{delta}"
            f.write(f"{key:<10}{fmt(b):>10}{fmt(a):>10}{dtxt:>10}\n")

    # Bar chart of before/after symmetry
    keys = list(s_before)
    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, [s_before[k] for k in keys], width, label="Before", color="#888")
    ax.bar(x + width / 2, [s_after[k] for k in keys], width, label="After", color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Symmetry (%)")
    ax.set_title("Gait symmetry: before vs after")
    ax.legend()
    fig.tight_layout()
    chart_path = os.path.join(args.out, "Comparison_Chart.png")
    fig.savefig(chart_path)

    print("✅ Comparison complete.")
    print(f"   Report: {report_path}")
    print(f"   Chart : {chart_path}")


if __name__ == "__main__":
    main()
