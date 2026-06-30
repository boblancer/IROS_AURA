"""
Create the paper train/test split CSVs for AURA anomaly detection.

Training frames are normal-only frames from each scene's normal_frames.csv,
filtered to the video IDs reported for paper Split 1 and Split 2. Test frames
come from each scene's test_frames.csv videos with soft labels converted to a
binary label using the configured threshold.

Usage:
    python create_soft_label_splits.py
    python create_soft_label_splits.py --threshold 0.5 --output splits
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


SCENES = ("scene_A", "scene_B")

PAPER_SPLITS = {
    "split_1": {
        "scene_A": ("v02", "v03", "v06", "v09"),
        "scene_B": ("v10", "v12", "v13", "v18", "v20"),
    },
    "split_2": {
        "scene_A": ("v01", "v02", "v03", "v05", "v06", "v08", "v09"),
        "scene_B": ("v10", "v12", "v13", "v14", "v15", "v18", "v20", "v21", "v23", "v24"),
    },
}

EXPECTED_TRAIN_COUNTS = {
    "split_1": {"scene_A": 3387, "scene_B": 508},
    "split_2": {"scene_A": 6516, "scene_B": 844},
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


def build_train(root, split_name):
    parts = []
    for scene in SCENES:
        videos = PAPER_SPLITS[split_name][scene]
        frames = pd.read_csv(root / scene / "normal_frames.csv")
        train = frames[frames["video"].isin(videos)].copy()
        train.insert(0, "scene", scene)
        train["soft_label"] = 0.0
        train["label"] = "normal"

        expected_count = EXPECTED_TRAIN_COUNTS[split_name][scene]
        if len(train) != expected_count:
            found_videos = ", ".join(sorted(train["video"].unique()))
            raise ValueError(
                f"{split_name}/{scene} expected {expected_count} training frames, "
                f"got {len(train)} from videos: {found_videos}"
            )

        parts.append(train[["scene", "video", "frame_idx", "soft_label", "label"]])

    return pd.concat(parts, ignore_index=True)


def build_test(root, threshold):
    soft = pd.read_csv(root / "annotations" / "soft_labels.csv")
    parts = []

    for scene in SCENES:
        test_videos = pd.read_csv(root / scene / "test_frames.csv")["video"].unique()
        scene_soft = soft[(soft["scene"] == scene) & (soft["video"].isin(test_videos))]
        parts.append(add_binary_label(scene_soft, threshold))

    return pd.concat(parts, ignore_index=True)[["scene", "video", "frame_idx", "soft_label", "label"]]


def write_split(output, split_name, train, test):
    split_dir = output / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(split_dir / "train.csv", index=False)
    test.to_csv(split_dir / "test.csv", index=False)


def print_summary(split_name, train, test):
    print(f"{split_name}:")
    print(f"  train: {len(train)} rows")
    print(train.groupby(["scene", "label"]).size().to_string())
    print("  train videos:")
    print(train.groupby("scene")["video"].unique().to_string())
    print(f"  test: {len(test)} rows")
    print(test.groupby(["scene", "label"]).size().to_string())
    print()


def main():
    parser = argparse.ArgumentParser(description="Create paper AURA train/test splits from soft labels")
    parser.add_argument("--threshold", type=float, default=0.5, help="Soft label threshold for binary labels")
    parser.add_argument("--output", default="splits", help="Output directory under AURA")
    args = parser.parse_args()

    root = project_root()
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)

    test = build_test(root, args.threshold)

    for split_name in PAPER_SPLITS:
        train = build_train(root, split_name)
        write_split(output, split_name, train, test)
        print_summary(split_name, train, test)

    print("Generated paper split artifacts only: split_1/train.csv, split_1/test.csv, split_2/train.csv, split_2/test.csv")


if __name__ == "__main__":
    main()
