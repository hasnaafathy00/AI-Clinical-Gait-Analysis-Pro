import argparse
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)

# --- 0. Cross-platform helpers ---------------------------------------------

def beep():
    """Cross-platform audible alert (best-effort, never fatal)."""
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.Beep(500, 30)
        else:
            # \a is the terminal bell; harmless if unsupported.
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception:
        pass


def default_output_dir():
    """Return the user's Desktop if it exists, else the home directory."""
    home = os.path.expanduser("~")
    desktop = os.path.join(home, "Desktop")
    return desktop if os.path.isdir(desktop) else home


# Clinical reference ranges for peak joint flexion during normal gait (degrees).
# Used only to annotate the report with context; not a diagnostic threshold.
NORMAL_RANGES = {
    "Hip":   (120, 165),   # hip angle span across the cycle
    "Knee":  (130, 175),   # knee near-extension at stance to flexion in swing
    "Ankle": (100, 130),   # ankle dorsi/plantar-flexion span
}

# Minimum landmark visibility (0-1) required to trust a computed angle.
VIS_THRESHOLD = 0.5


# --- 1. Settings and patient info ------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AI-Powered Clinical Gait Analysis (bilateral kinematics)."
    )
    p.add_argument("video", help="Path to the input gait video file.")
    p.add_argument("--name", help="Patient name (prompted if omitted).")
    p.add_argument("--age", help="Patient age (prompted if omitted).")
    p.add_argument(
        "--out", default=default_output_dir(),
        help="Directory to write the report and graphs (default: Desktop).",
    )
    p.add_argument(
        "--no-display", action="store_true",
        help="Run headless: don't open OpenCV/Matplotlib windows.",
    )
    p.add_argument(
        "--no-video", action="store_true",
        help="Do not export the annotated overlay video (.mp4).",
    )
    return p.parse_args()


# --- 2. Geometry (3D, using MediaPipe metric world landmarks) --------------

def calculate_angle_3d(a, b, c):
    """Angle at vertex b formed by 3D points a-b-c, in degrees (0-180).

    Uses metric world coordinates, so the result is anatomically correct
    regardless of the camera viewing angle.
    """
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return np.nan
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def world_point(lm, idx):
    """Return a world landmark as a 3D point, or None if not visible enough."""
    p = lm[idx]
    if p.visibility < VIS_THRESHOLD:
        return None
    return [p.x, p.y, p.z]


def joint_angle(world_lm, i, j, k):
    """Angle at joint j (i-j-k) from world landmarks, or NaN if unreliable."""
    a = world_point(world_lm, i)
    b = world_point(world_lm, j)
    c = world_point(world_lm, k)
    if a is None or b is None or c is None:
        return np.nan
    return calculate_angle_3d(a, b, c)


def detect_phase(k_ang, a_ang):
    """Classify a single leg's gait phase from its knee and ankle angle."""
    if np.isnan(k_ang) or np.isnan(a_ang):
        return "Unknown"
    if k_ang > 165 and a_ang < 110:
        return "Heel Strike"
    if k_ang < 140:
        return "Swing Phase"
    return "Stance Phase"


# --- 3. Metrics ------------------------------------------------------------

def symmetry_index(mean_l, mean_r):
    """Percent symmetry between two mean angles; 100 = perfect, guards div-by-0."""
    denom = (mean_l + mean_r) / 2.0
    if not np.isfinite(denom) or denom == 0:
        return float("nan")
    return round(100 - (abs(mean_l - mean_r) / denom * 100), 2)


def heel_strike_onsets(phases):
    """Frame indices where a leg transitions *into* 'Heel Strike'."""
    onsets = []
    prev = None
    for i, ph in enumerate(phases):
        if ph == "Heel Strike" and prev != "Heel Strike":
            onsets.append(i)
        prev = ph
    return onsets


def stance_swing_pct(phases):
    """Return (stance %, swing %) for a leg, ignoring Unknown frames."""
    stance = sum(1 for p in phases if p in ("Heel Strike", "Stance Phase"))
    swing = sum(1 for p in phases if p == "Swing Phase")
    total = stance + swing
    if total == 0:
        return float("nan"), float("nan")
    return round(stance / total * 100, 1), round(swing / total * 100, 1)


def temporal_spatial(df, fps):
    """Compute temporal-spatial gait parameters from the per-frame phases."""
    left, right = df['Phase_L'].tolist(), df['Phase_R'].tolist()

    l_stance, l_swing = stance_swing_pct(left)
    r_stance, r_swing = stance_swing_pct(right)

    # Double support: both legs bearing weight (neither in swing) simultaneously.
    both_stance = sum(
        1 for pl, pr in zip(left, right)
        if pl in ("Heel Strike", "Stance Phase") and pr in ("Heel Strike", "Stance Phase")
    )
    double_support_time = round(both_stance / fps, 2) if fps > 0 else float("nan")

    # Step time per leg from consecutive heel-strike onsets, then asymmetry.
    def mean_step_time(phases):
        onsets = heel_strike_onsets(phases)
        if len(onsets) < 2:
            return float("nan")
        gaps = np.diff(onsets) / fps
        return float(np.mean(gaps))

    l_step, r_step = mean_step_time(left), mean_step_time(right)
    if np.isfinite(l_step) and np.isfinite(r_step) and (l_step + r_step) > 0:
        step_asym = round(abs(l_step - r_step) / ((l_step + r_step) / 2) * 100, 1)
    else:
        step_asym = float("nan")

    return {
        "l_stance": l_stance, "l_swing": l_swing,
        "r_stance": r_stance, "r_swing": r_swing,
        "double_support_time": double_support_time,
        "l_step_time": round(l_step, 2) if np.isfinite(l_step) else float("nan"),
        "r_step_time": round(r_step, 2) if np.isfinite(r_step) else float("nan"),
        "step_asymmetry": step_asym,
    }


def _fmt(x, suffix=""):
    """Format a possibly-NaN number for the report."""
    return "n/a" if not np.isfinite(x) else f"{x}{suffix}"


def normalize_cycles(series, onsets, n_points=101):
    """Resample each gait cycle to 0-100% and stack them.

    `onsets` are the frame indices that bound each cycle (e.g. heel strikes).
    Returns an array of shape (n_cycles, n_points); empty if <2 onsets.
    """
    values = np.asarray(series, dtype=float)
    grid = np.linspace(0, 100, n_points)
    cycles = []
    for start, end in zip(onsets[:-1], onsets[1:]):
        seg = values[start:end]
        seg = seg[np.isfinite(seg)]
        if len(seg) < 3:
            continue
        x = np.linspace(0, 100, len(seg))
        cycles.append(np.interp(grid, x, seg))
    return np.array(cycles) if cycles else np.empty((0, n_points))


def ensemble_average(cycles):
    """Mean and ±1 SD across normalized cycles; (None, None) if empty."""
    if cycles.size == 0:
        return None, None
    return cycles.mean(axis=0), cycles.std(axis=0)


# --- 4. Reporting ----------------------------------------------------------

# Threshold lines drawn on the joint-angle graphs (knee/hip/ankle).
K_T, H_T, A_T = 150, 145, 120


def generate_outputs(df, fps, out_dir, p_name, p_age, display=False,
                     detected_frames=None, total_frames=None, video_path=None):
    """Compute metrics and write the report, graphs, normalized curves,
    PDF, and CSV from a per-frame dataframe. Reusable by the video pipeline
    and by demo/synthetic drivers. Returns the overall symmetry score.
    """
    os.makedirs(out_dir, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in p_name) or "patient"
    if total_frames is None:
        total_frames = len(df)
    if detected_frames is None:
        detected_frames = len(df)

    # Smooth ankle data (rolling mean tolerates the NaNs from occluded frames).
    df['LA'] = df['LA'].rolling(window=10, min_periods=1).mean()
    df['RA'] = df['RA'].rolling(window=10, min_periods=1).mean()

    # Per-joint symmetry (NaN-safe means), plus an overall average.
    sym = {
        "Hip":   symmetry_index(df['LH'].mean(), df['RH'].mean()),
        "Knee":  symmetry_index(df['LK'].mean(), df['RK'].mean()),
        "Ankle": symmetry_index(df['LA'].mean(), df['RA'].mean()),
    }
    valid = [v for v in sym.values() if np.isfinite(v)]
    overall = round(sum(valid) / len(valid), 2) if valid else float("nan")

    # Range of motion per joint (mean of left/right spans).
    def rom(col_l, col_r):
        spans = []
        for col in (col_l, col_r):
            s = df[col].dropna()
            if not s.empty:
                spans.append(s.max() - s.min())
        return round(float(np.mean(spans)), 1) if spans else float("nan")

    roms = {"Hip": rom('LH', 'RH'), "Knee": rom('LK', 'RK'), "Ankle": rom('LA', 'RA')}

    # Cadence / step metrics from left-leg heel strikes.
    steps = len(heel_strike_onsets(df['Phase_L'].tolist()))
    duration_s = len(df) / fps
    cadence = round(steps / duration_s * 60, 1) if duration_s > 0 else 0.0
    stride_time = round(duration_s / (steps / 2), 2) if steps >= 2 else float("nan")

    # Temporal-spatial parameters.
    ts = temporal_spatial(df, fps)

    # Analysis quality: fraction of frames with a reliable pose.
    quality = round(detected_frames / total_frames * 100, 1) if total_frames else float("nan")

    # Most common non-Unknown phase for the summary line.
    known = df['Phase_L'][df['Phase_L'] != "Unknown"]
    main_phase = known.mode()[0] if not known.empty else "Unknown"

    # Save the raw per-frame data as CSV for further analysis.
    csv_path = os.path.join(out_dir, f"{safe_name}_gait_data.csv")
    df.to_csv(csv_path, index=False)

    # Save report
    report_path = os.path.join(out_dir, f"{safe_name}_Final_Master_Report.txt")
    with open(report_path, "w") as f:
        f.write(f"GAIT ANALYSIS MASTER REPORT\n{'=' * 30}\n")
        f.write(f"Patient Name: {p_name}\nPatient Age: {p_age}\n")
        f.write(f"Frames analysed: {len(df)}   Duration: {duration_s:.1f}s @ {fps:.0f} fps\n")
        f.write(f"Analysis quality: {_fmt(quality, '%')} of frames with reliable pose\n\n")

        f.write("SYMMETRY (100% = perfectly symmetric)\n")
        for joint, score in sym.items():
            f.write(f"  {joint:<6}: {_fmt(score, '%')}\n")
        f.write(f"  {'Overall':<6}: {_fmt(overall, '%')}\n\n")

        f.write("RANGE OF MOTION (degrees, mean of both legs)\n")
        for joint, val in roms.items():
            lo, hi = NORMAL_RANGES[joint]
            f.write(f"  {joint:<6}: {_fmt(val)}   (typical span {lo}-{hi})\n")
        f.write("\n")

        f.write("TEMPORAL-SPATIAL PARAMETERS\n")
        f.write(f"  Stance (L/R)   : {_fmt(ts['l_stance'], '%')} / {_fmt(ts['r_stance'], '%')}\n")
        f.write(f"  Swing  (L/R)   : {_fmt(ts['l_swing'], '%')} / {_fmt(ts['r_swing'], '%')}\n")
        f.write(f"  Double support : {_fmt(ts['double_support_time'], 's')}\n")
        f.write(f"  Step time (L/R): {_fmt(ts['l_step_time'], 's')} / {_fmt(ts['r_step_time'], 's')}\n")
        f.write(f"  Step asymmetry : {_fmt(ts['step_asymmetry'], '%')}\n\n")

        f.write("GAIT METRICS\n")
        f.write(f"  Steps detected : {steps}\n")
        f.write(f"  Cadence        : {cadence} steps/min\n")
        f.write(f"  Stride time    : {_fmt(stride_time, 's')}\n")
        f.write(f"  Main phase     : {main_phase}\n\n")

        status = "Excellent Symmetry" if np.isfinite(overall) and overall > 90 else "Asymmetry Detected"
        f.write(f"Status: {status}\n")

    # Build side-by-side graphs (3x2)
    fig, axs = plt.subplots(3, 2, figsize=(15, 20))

    def draw_bilateral(ax_pair, l_data, r_data, thresh, label):
        ax_pair[0].plot(df['F'], l_data, color='green', label='Left')
        ax_pair[0].axhline(y=thresh, color='red', linestyle='--')
        ax_pair[0].set_title(f"{label} - Left")
        ax_pair[0].legend()
        ax_pair[1].plot(df['F'], r_data, color='blue', label='Right')
        ax_pair[1].axhline(y=thresh, color='red', linestyle='--')
        ax_pair[1].set_title(f"{label} - Right")
        ax_pair[1].legend()

    draw_bilateral(axs[0], df['LH'], df['RH'], H_T, 'Hip Angle')
    draw_bilateral(axs[1], df['LK'], df['RK'], K_T, 'Knee Angle')
    draw_bilateral(axs[2], df['LA'], df['RA'], A_T, 'Ankle Angle (Filtered)')

    plt.tight_layout()
    graph_path = os.path.join(out_dir, f"{safe_name}_Full_Grafts.png")
    fig.savefig(graph_path)

    # Gait-cycle normalization (ensemble-averaged curves).
    onsets = heel_strike_onsets(df['Phase_L'].tolist())
    norm_fig, norm_axs = plt.subplots(3, 1, figsize=(10, 14))
    grid = np.linspace(0, 100, 101)
    joint_cols = [("Hip", 'LH', 'RH'), ("Knee", 'LK', 'RK'), ("Ankle", 'LA', 'RA')]
    n_cycles = max(len(onsets) - 1, 0)
    for ax, (label, lcol, rcol) in zip(norm_axs, joint_cols):
        for col, colour, side in ((lcol, 'green', 'Left'), (rcol, 'blue', 'Right')):
            mean, sd = ensemble_average(normalize_cycles(df[col].tolist(), onsets))
            if mean is not None:
                ax.plot(grid, mean, color=colour, label=f"{side} (mean)")
                ax.fill_between(grid, mean - sd, mean + sd, color=colour, alpha=0.15)
        ax.set_title(f"{label} — normalized gait cycle ({n_cycles} cycles)")
        ax.set_xlabel("Gait cycle (%)")
        ax.set_ylabel("Angle (deg)")
        if ax.get_legend_handles_labels()[1]:
            ax.legend()
    norm_fig.tight_layout()
    norm_path = os.path.join(out_dir, f"{safe_name}_Normalized_Cycles.png")
    norm_fig.savefig(norm_path)

    # Combined PDF report.
    pdf_path = os.path.join(out_dir, f"{safe_name}_Report.pdf")
    with open(report_path) as rf:
        report_text = rf.read()
    text_fig = plt.figure(figsize=(8.5, 11))
    text_fig.text(0.07, 0.97, report_text, family="monospace", fontsize=10, va="top")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(text_fig)
        pdf.savefig(fig)
        pdf.savefig(norm_fig)

    if display:
        plt.show(block=True)
    plt.close('all')

    print(f"✅ DONE! Overall symmetry: {_fmt(overall, '%')}")
    print(f"   Report: {report_path}")
    print(f"   PDF   : {pdf_path}")
    print(f"   Graphs: {graph_path}")
    print(f"   Cycles: {norm_path}")
    print(f"   Data  : {csv_path}")
    if video_path is not None:
        print(f"   Video : {video_path}")
    return overall


# --- 5. Main ---------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.isfile(args.video):
        sys.exit(f"❌ Video not found: {args.video}")
    os.makedirs(args.out, exist_ok=True)

    print("--- 🩺 Welcome to Hasna's Ultra Gait Analysis System ---")
    p_name = args.name or input("Enter Patient Name: ")
    p_age = args.age or input("Enter Patient Age: ")
    # Sanitise the name so it is safe to use in output filenames.
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in p_name) or "patient"

    display = not args.no_display
    if not display:
        matplotlib.use("Agg")  # headless backend

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils
    data_log = []

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"❌ Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 30.0  # sensible fallback when the container lacks metadata
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    K_T, H_T, A_T = 150, 145, 120  # alert thresholds (knee/hip/ankle)

    print(f"🚀 Processing {p_name} ({p_age} years old)... Please wait.")

    below_threshold = False   # debounce state for the audible alert
    writer = None             # annotated-video writer (lazy-init on first frame)
    video_path = os.path.join(args.out, f"{safe_name}_annotated.mp4")
    detected_frames = 0

    progress = tqdm(total=total_frames or None, unit="frame", desc="Analysing")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if progress is not None:
            progress.update(1)
        h, w, _ = frame.shape

        if writer is None and not args.no_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if res.pose_landmarks and res.pose_world_landmarks:
            detected_frames += 1
            wlm = res.pose_world_landmarks.landmark
            mp_drawing.draw_landmarks(image, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Bilateral 3D angle calculations (NaN when a joint is occluded).
            lk = joint_angle(wlm, 23, 25, 27)
            rk = joint_angle(wlm, 24, 26, 28)
            lh = joint_angle(wlm, 11, 23, 25)
            rh = joint_angle(wlm, 12, 24, 26)
            la = joint_angle(wlm, 25, 27, 31)
            ra = joint_angle(wlm, 26, 28, 32)

            # Phase is detected per leg, then reported for both sides.
            phase_l = detect_phase(lk, la)
            phase_r = detect_phase(rk, ra)
            data_log.append({'F': cap.get(cv2.CAP_PROP_POS_FRAMES),
                             'LK': lk, 'RK': rk, 'LH': lh, 'RH': rh,
                             'LA': la, 'RA': ra,
                             'Phase_L': phase_l, 'Phase_R': phase_r})

            # Audible alert — fire once on crossing below threshold, not every frame.
            now_below = (not np.isnan(lk)) and lk < K_T
            if now_below and not below_threshold:
                beep()
            below_threshold = now_below

            # On-frame UI overlay
            def fmt(x):
                return "--" if np.isnan(x) else str(int(x))

            cv2.rectangle(image, (0, 0), (350, 160), (0, 0, 0), -1)
            cv2.putText(image, f"Patient: {p_name} | Age: {p_age}", (10, 30), 1, 1.2, (255, 255, 255), 1)
            cv2.putText(image, f"PHASE L/R: {phase_l} / {phase_r}", (10, 65), 1, 1.2, (0, 255, 255), 2)
            cv2.putText(image, f"KNEE L/R: {fmt(lk)}|{fmt(rk)}", (10, 105), 1, 1.3, (0, 255, 0), 1)
            cv2.putText(image, f"HIP  L/R: {fmt(lh)}|{fmt(rh)}", (10, 140), 1, 1.3, (0, 255, 0), 1)

        if writer is not None:
            writer.write(image)

        if display:
            cv2.imshow('Hasna Advanced Clinical Gait', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if progress is not None:
        progress.close()
    if not total_frames:
        total_frames = detected_frames
    cap.release()
    if writer is not None:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    if not data_log:
        sys.exit("❌ No pose landmarks detected in the video — nothing to report.")

    df = pd.DataFrame(data_log)
    generate_outputs(
        df, fps, args.out, p_name, p_age,
        display=display,
        detected_frames=detected_frames, total_frames=total_frames,
        video_path=video_path if writer is not None else None,
    )


if __name__ == "__main__":
    main()
