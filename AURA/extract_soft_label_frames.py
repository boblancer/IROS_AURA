"""
Extract AURA video frames listed in annotations/soft_labels.csv.

This keeps the soft label attached in a manifest and optionally places each
frame into normal/anomalous folders using a threshold.

Usage:
    python extract_soft_label_frames.py --threshold 0.5
    python extract_soft_label_frames.py --scene scene_A --output soft_label_frames_a
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


def project_root():
    """Return the AURA directory whether this is run from repo root or AURA/."""
    here = Path(__file__).resolve().parent
    if (here / "annotations" / "soft_labels.csv").exists():
        return here
    if (Path.cwd() / "annotations" / "soft_labels.csv").exists():
        return Path.cwd()
    print("Error: could not find annotations/soft_labels.csv")
    sys.exit(1)


def extract_video_frames(video_path, rows, output_root, threshold):
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    records = []
    video_name = video_path.stem
    scene = rows.iloc[0]["scene"]

    for row in tqdm(rows.itertuples(index=False), total=len(rows), desc=f"{scene}/{video_name}"):
        frame_idx = int(row.frame_idx)
        soft_label = float(row.soft_label)
        label = "anomalous" if soft_label >= threshold else "normal"

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            records.append(
                {
                    "scene": scene,
                    "video": video_name,
                    "frame_idx": frame_idx,
                    "soft_label": soft_label,
                    "label": label,
                    "path": "",
                    "status": "missing",
                }
            )
            continue

        out_dir = output_root / scene / label / video_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"frame_{frame_idx:04d}_soft_{soft_label:.4f}.png"
        cv2.imwrite(str(out_path), frame)

        records.append(
            {
                "scene": scene,
                "video": video_name,
                "frame_idx": frame_idx,
                "soft_label": soft_label,
                "label": label,
                "path": str(out_path),
                "status": "ok",
            }
        )

    cap.release()
    return records


def main():
    parser = argparse.ArgumentParser(description="Extract frames using AURA soft labels")
    parser.add_argument("--scene", choices=["scene_A", "scene_B"], help="Extract one scene only")
    parser.add_argument("--threshold", type=float, default=0.5, help="Soft label threshold for anomalous")
    parser.add_argument(
        "--min-soft-label",
        type=float,
        help="Only extract rows with soft_label greater than or equal to this value",
    )
    parser.add_argument("--output", default="soft_label_frames", help="Output directory")
    args = parser.parse_args()

    root = project_root()
    labels = pd.read_csv(root / "annotations" / "soft_labels.csv")
    if args.scene:
        labels = labels[labels["scene"] == args.scene]
    if args.min_soft_label is not None:
        labels = labels[labels["soft_label"] >= args.min_soft_label]

    output_root = root / args.output
    all_records = []

    for (scene, video), rows in labels.groupby(["scene", "video"], sort=True):
        video_path = root / scene / "videos" / f"{video}.mp4"
        if not video_path.exists():
            print(f"Warning: missing video {video_path}")
            continue
        all_records.extend(extract_video_frames(video_path, rows, output_root, args.threshold))

    manifest = output_root / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_records).to_csv(manifest, index=False)

    ok_count = sum(record["status"] == "ok" for record in all_records)
    missing_count = len(all_records) - ok_count
    print(f"\nDone: extracted {ok_count} frames to {output_root}")
    print(f"Manifest: {manifest}")
    if missing_count:
        print(f"Warning: {missing_count} frame(s) could not be read")


if __name__ == "__main__":
    main()
