"""Batch gait analysis — run gait_analysis over every video in a folder.

Usage:
    python batch.py path/to/videos --out ./results
"""
import argparse
import glob
import os
import subprocess
import sys

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def find_videos(folder):
    vids = []
    for ext in VIDEO_EXTS:
        vids.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        vids.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(vids))


def main():
    p = argparse.ArgumentParser(description="Batch-process a folder of gait videos.")
    p.add_argument("folder", help="Folder containing video files.")
    p.add_argument("--out", default="./results", help="Output directory (default: ./results).")
    p.add_argument("--age", default="0", help="Age to record for every clip (default: 0).")
    args = p.parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"❌ Not a folder: {args.folder}")

    videos = find_videos(args.folder)
    if not videos:
        sys.exit(f"❌ No videos found in {args.folder}")

    os.makedirs(args.out, exist_ok=True)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gait_analysis.py")

    print(f"📂 Found {len(videos)} video(s). Processing headlessly...")
    failures = []
    for i, video in enumerate(videos, 1):
        name = os.path.splitext(os.path.basename(video))[0]
        print(f"\n[{i}/{len(videos)}] {name}")
        cmd = [sys.executable, script, video,
               "--name", name, "--age", args.age,
               "--out", args.out, "--no-display", "--no-video"]
        if subprocess.run(cmd).returncode != 0:
            failures.append(name)

    print(f"\n✅ Batch complete: {len(videos) - len(failures)} ok, {len(failures)} failed.")
    if failures:
        print("   Failed:", ", ".join(failures))


if __name__ == "__main__":
    main()
