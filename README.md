# 🩺 AI-Powered Clinical Gait Analysis Pro

An automated, **bilateral** gait-analysis tool for physical therapists and researchers.
It takes a video of a person walking and uses computer vision (MediaPipe pose
estimation) to measure hip, knee, and ankle angles for both legs, detect gait
phases, and produce a clinical report with symmetry and cadence metrics.

## 🚀 Key Features
- **3D bilateral kinematics** — hip, knee, and ankle angles for both legs computed
  from MediaPipe **metric world landmarks**, so angles are anatomically correct
  regardless of camera viewing angle.
- **Per-leg gait-phase detection** — classifies Heel Strike, Stance, and Swing for
  the left and right leg independently.
- **Symmetry scoring** — per-joint symmetry (hip / knee / ankle) plus an overall
  score, computed from occlusion-filtered data.
- **Temporal-spatial parameters** — stance %, swing %, double-support time, per-leg
  step time, and left-vs-right step asymmetry.
- **Gait metrics** — step count, cadence (steps/min), stride time, and per-joint
  range of motion (ROM).
- **Annotated video export** — saves the live skeleton/overlay as an `.mp4` for
  replay and record-keeping.
- **Clinical reporting** — an automated text report, a 3×2 kinematic graph grid
  (`.png`), and the full per-frame data as `.csv` for further analysis.
- **Robustness** — skips low-visibility landmarks, reports an analysis-quality
  score, tolerates missing frames, and runs headless for batch processing.

## 🛠️ Technology Stack
- **Python 3.9+**
- **MediaPipe** — pose estimation
- **OpenCV** — video processing
- **NumPy / Pandas** — numerical analysis
- **Matplotlib** — visualization

## 📦 Installation
```bash
git clone https://github.com/hasnaafathy00/AI-Clinical-Gait-Analysis-Pro.git
cd AI-Clinical-Gait-Analysis-Pro

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Download the pose model (required — the MediaPipe Tasks API needs it):
curl -L -o pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

> **Note:** This project uses the modern **MediaPipe Tasks API** (`PoseLandmarker`).
> The legacy `mp.solutions.pose` API was removed in recent MediaPipe releases, so
> the `pose_landmarker.task` model file above is required. Place it next to
> `gait_analysis.py`, or point to it with the `POSE_MODEL_PATH` environment variable.

## ▶️ Usage
```bash
# Interactive — prompts for patient name and age, opens live windows:
python gait_analysis.py path/to/walk.mp4

# Fully specified, headless (no windows — good for servers/batch):
python gait_analysis.py path/to/walk.mp4 \
    --name "Jane Doe" --age 30 \
    --out ./results --no-display
```

### Arguments
| Argument        | Description                                             |
|-----------------|---------------------------------------------------------|
| `video`         | Path to the input gait video (required).                |
| `--name`        | Patient name (prompted if omitted).                     |
| `--age`         | Patient age (prompted if omitted).                      |
| `--out`         | Output directory for report/graphs/CSV/video (default: Desktop).|
| `--no-display`  | Run headless — don't open OpenCV/Matplotlib windows.    |
| `--no-video`    | Skip exporting the annotated `.mp4` overlay.            |

During live display, press **`q`** to stop processing early.

## 📊 Outputs
For a patient named `Jane`, the tool writes to `--out`:
- `Jane_Final_Master_Report.txt` — symmetry, ROM, temporal-spatial, cadence, and phase summary.
- `Jane_Report.pdf` — the report plus all graphs combined into one PDF.
- `Jane_Full_Grafts.png` — 3×2 bilateral hip/knee/ankle angle graphs.
- `Jane_Normalized_Cycles.png` — ensemble-averaged gait-cycle curves (0–100%) with ±1 SD bands.
- `Jane_gait_data.csv` — raw per-frame angles and phases.
- `Jane_annotated.mp4` — the skeleton/overlay video (unless `--no-video`).

## 🧰 Additional Tools
```bash
# Web UI — drag-drop a video in the browser:
streamlit run app.py

# Batch mode — analyse every video in a folder:
python batch.py path/to/videos --out ./results

# Pre/post comparison — compare two sessions' *_gait_data.csv files:
python compare.py before_gait_data.csv after_gait_data.csv --out ./results

# Run the test suite:
pytest
```

## ⚠️ Disclaimer
This is a research/educational tool and is **not** a certified medical device.
Results depend on camera angle, lighting, and clothing, and should not be used as
the sole basis for clinical diagnosis.
