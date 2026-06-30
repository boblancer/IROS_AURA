"""
Create train/val/test CSVs for AURA anomaly detection from the provided labels.

Train uses only normal frames from normal_frames.csv. Val and test use soft labels
from the dataset's held-out test videos so thresholds can be tuned without
touching final test videos.

Usage:
    python create_soft_label_splits.py
    python create_soft_label_splits.py --threshold 0.5 --output splits
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


DEFAULT_VAL_VIDEOS = {
    "scene_A": ["v07"],
    "scene_B": ["v22"],
}


def project_root():
    here = Path(__file__).resolve().parent
    if (here / "annotations" / "soft_labels.csv").exists():
        return here
    if (Path.cwd() / "annotations" / "soft_labels.csv").exists():
        return Path.cwd()
    print("Error: could not find annotations/soft_labels.csv")
    sys.exit(1)


def add_binary_label(df, threshold):
    df = df.copy()
    df["label"] = df["soft_label"].apply(lambda value: "anomalous" if value >= threshold else "normal")
    return df


def main():
    parser = argparse.ArgumentParser(description="Create AURA train/val/test splits from soft labels")
    parser.add_argument("--threshold", type=float, default=0.5, help="Soft label threshold for binary labels")
    parser.add_argument("--output", default="splits", help="Output directory under AURA")
    args = parser.parse_args()

    root = project_root()
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)

    soft = pd.read_csv(root / "annotations" / "soft_labels.csv")
    train_parts = []
    val_parts = []
    test_parts = []

    for scene in ["scene_A", "scene_B"]:
        train = pd.read_csv(root / scene / "normal_frames.csv")
        train.insert(0, "scene", scene)
        train["soft_label"] = 0.0
        train["label"] = "normal"
        train_parts.append(train[["scene", "video", "frame_idx", "soft_label", "label"]])

        test_frame_videos = pd.read_csv(root / scene / "test_frames.csv")["video"].unique()
        scene_soft = add_binary_label(
            soft[(soft["scene"] == scene) & (soft["video"].isin(test_frame_videos))],
            args.threshold,
        )
        val_videos = DEFAULT_VAL_VIDEOS[scene]
        val_parts.append(scene_soft[scene_soft["video"].isin(val_videos)])
        test_parts.append(scene_soft[~scene_soft["video"].isin(val_videos)])

    splits = {
        "train": pd.concat(train_parts, ignore_index=True),
        "val": pd.concat(val_parts, ignore_index=True),
        "test": pd.concat(test_parts, ignore_index=True),
    }

    for name, df in splits.items():
        path = output / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"{name}: {len(df)} rows -> {path}")
        print(df.groupby(["scene", "label"]).size().to_string())
        print()

    print("Validation videos: scene_A/v07, scene_B/v22")
    print("The remaining held-out soft-labeled videos are in test. Train is normal-only.")


if __name__ == "__main__":
    main()
