"""Streamlit web UI for AI Clinical Gait Analysis.

Drag-drop a walking video in the browser and get the report, graphs, and
downloadable outputs — no command line needed.

Run with:
    streamlit run app.py
"""
import glob
import os
import subprocess
import sys
import tempfile

import streamlit as st

st.set_page_config(page_title="AI Clinical Gait Analysis", page_icon="🩺", layout="centered")
st.title("🩺 AI Clinical Gait Analysis Pro")
st.caption("Upload a side-on walking video to generate a bilateral gait report.")

name = st.text_input("Patient name", value="patient")
age = st.text_input("Patient age", value="0")
video_file = st.file_uploader("Walking video", type=["mp4", "mov", "avi", "mkv", "m4v"])

if video_file and st.button("Run analysis", type="primary"):
    workdir = tempfile.mkdtemp(prefix="gait_")
    video_path = os.path.join(workdir, video_file.name)
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gait_analysis.py")
    with st.spinner("Analysing video… this can take a minute."):
        result = subprocess.run(
            [sys.executable, script, video_path,
             "--name", name or "patient", "--age", age or "0",
             "--out", workdir, "--no-display", "--no-video"],
            capture_output=True, text=True,
        )

    if result.returncode != 0:
        st.error("Analysis failed.")
        st.code(result.stderr or result.stdout)
    else:
        st.success("Done!")

        # Report text
        reports = glob.glob(os.path.join(workdir, "*_Final_Master_Report.txt"))
        if reports:
            with open(reports[0]) as f:
                st.subheader("Report")
                st.code(f.read())

        # Graphs
        for pattern, caption in [("*_Full_Grafts.png", "Bilateral joint angles"),
                                 ("*_Normalized_Cycles.png", "Normalized gait cycles")]:
            imgs = glob.glob(os.path.join(workdir, pattern))
            if imgs:
                st.image(imgs[0], caption=caption, use_container_width=True)

        # Downloads
        st.subheader("Downloads")
        for pattern, label, mime in [
            ("*_Report.pdf", "PDF report", "application/pdf"),
            ("*_gait_data.csv", "Raw data (CSV)", "text/csv"),
        ]:
            files = glob.glob(os.path.join(workdir, pattern))
            if files:
                with open(files[0], "rb") as f:
                    st.download_button(label, f.read(), file_name=os.path.basename(files[0]), mime=mime)
