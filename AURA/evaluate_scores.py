"""
Evaluate per-frame anomaly scores against AURA soft labels and consensus events.

The score CSV must contain one row per evaluated frame with scene, video,
frame_idx, and an anomaly score column. Common score column names such as
score, anomaly_score, pred_score, and image_score are detected automatically.

Usage:
    python evaluate_scores.py --scores rd_split2_scores.csv --split split_2
    python evaluate_scores.py --scores scores.csv --score-column image_score
    python evaluate_scores.py --scores scores.csv --normalized-output normalized_scores.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths


SCORE_COLUMN_CANDIDATES = ("score", "anomaly_score", "pred_score", "image_score")


def project_root():
    here = Path(__file__).resolve().parent
    if (here / "annotations" / "soft_labels.csv").exists():
        return here
    if (Path.cwd() / "annotations" / "soft_labels.csv").exists():
        return Path.cwd()
    print("Error: could not find annotations/soft_labels.csv")
    sys.exit(1)


def infer_score_column(scores, requested):
    if requested:
        if requested not in scores.columns:
            raise ValueError(f"Score column '{requested}' not found in {list(scores.columns)}")
        return requested

    for column in SCORE_COLUMN_CANDIDATES:
        if column in scores.columns:
            return column

    raise ValueError(
        "Could not infer score column. Pass --score-column. "
        f"Available columns: {list(scores.columns)}"
    )


def read_scores(path, score_column):
    scores = pd.read_csv(path)
    column = infer_score_column(scores, score_column)
    required = {"scene", "video", "frame_idx", column}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Scores file is missing required columns: {sorted(missing)}")

    scores = scores[["scene", "video", "frame_idx", column]].copy()
    scores = scores.rename(columns={column: "score"})
    scores["frame_idx"] = scores["frame_idx"].astype(int)
    return scores


def normalize_per_video(df):
    df = df.sort_values(["scene", "video", "frame_idx"]).copy()
    grouped = df.groupby(["scene", "video"])["score"]
    mins = grouped.transform("min")
    maxs = grouped.transform("max")
    df["score_norm"] = (df["score"] - mins) / (maxs - mins + 1e-8)
    return df


def load_eval_frames(root, split):
    split_path = root / "splits" / split / "test.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Split test file not found: {split_path}")
    return pd.read_csv(split_path)[["scene", "video", "frame_idx", "soft_label"]]


def merge_scores(eval_frames, scores):
    merged = eval_frames.merge(scores, on=["scene", "video", "frame_idx"], how="left")
    missing = merged["score"].isna().sum()
    if missing:
        examples = merged[merged["score"].isna()][["scene", "video", "frame_idx"]].head()
        raise ValueError(f"Missing scores for {missing} eval frames. Examples:\n{examples.to_string(index=False)}")
    return normalize_per_video(merged)


def temporal_iou(pred_start, pred_end, true_start, true_end):
    overlap = max(0.0, min(pred_end, true_end) - max(pred_start, true_start))
    union = max(pred_end, true_end) - min(pred_start, true_start)
    if union <= 0:
        return 0.0
    return overlap / union


def widest_peak_interval(frame_indices, scores_norm, relative_height):
    peaks, _ = find_peaks(scores_norm)

    if len(peaks) == 0:
        peak = int(np.argmax(scores_norm))
        return float(frame_indices[peak]), float(frame_indices[peak])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        widths, _, left_ips, right_ips = peak_widths(scores_norm, peaks, rel_height=relative_height)
    widest = int(np.argmax(widths))
    x = np.arange(len(scores_norm))
    pred_start = float(np.interp(left_ips[widest], x, frame_indices))
    pred_end = float(np.interp(right_ips[widest], x, frame_indices))
    return pred_start, pred_end


def event_detection(merged, consensus):
    rows = []
    heights = np.round(np.arange(0.0, 1.0001, 0.01), 2)

    for (scene, video), video_df in merged.groupby(["scene", "video"]):
        true_row = consensus[(consensus["scene"] == scene) & (consensus["video"] == video)]
        if true_row.empty:
            continue

        video_df = video_df.sort_values("frame_idx")
        frame_indices = video_df["frame_idx"].to_numpy(dtype=float)
        scores_norm = video_df["score_norm"].to_numpy(dtype=float)
        true_start = float(true_row.iloc[0]["start_frame"])
        true_end = float(true_row.iloc[0]["end_frame"])

        best = None
        for height in heights:
            pred_start, pred_end = widest_peak_interval(frame_indices, scores_norm, height)
            tiou = temporal_iou(pred_start, pred_end, true_start, true_end)
            candidate = {
                "scene": scene,
                "video": video,
                "height": height,
                "pred_start": pred_start,
                "pred_end": pred_end,
                "true_start": true_start,
                "true_end": true_end,
                "tiou": tiou,
            }
            if best is None or candidate["tiou"] > best["tiou"]:
                best = candidate

        rows.append(best)

    return pd.DataFrame(rows)


def print_metrics(merged, events):
    mae_by_scene = merged.assign(abs_error=(merged["score_norm"] - merged["soft_label"]).abs())
    mae_by_scene = mae_by_scene.groupby("scene")["abs_error"].mean()

    print("MAE vs soft labels")
    for scene, mae in mae_by_scene.items():
        print(f"  {scene}: {mae:.4f}")
    print(f"  overall: {merged['score_norm'].sub(merged['soft_label']).abs().mean():.4f}")
    print()

    if events.empty:
        print("No consensus events matched the scored test frames.")
        return

    print("Peak-finding event t-IoU")
    for scene, tiou in events.groupby("scene")["tiou"].mean().items():
        print(f"  {scene}: {tiou:.4f}")
    print(f"  overall: {events['tiou'].mean():.4f}")
    print()
    print("Per-video best peak intervals")
    columns = ["scene", "video", "height", "pred_start", "pred_end", "true_start", "true_end", "tiou"]
    print(events[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def main():
    parser = argparse.ArgumentParser(description="Evaluate AURA per-frame anomaly scores")
    parser.add_argument("--scores", required=True, help="CSV with scene, video, frame_idx, and anomaly score columns")
    parser.add_argument("--score-column", help="Name of anomaly score column in --scores")
    parser.add_argument("--split", default="split_2", choices=("split_1", "split_2"), help="Paper split to evaluate")
    parser.add_argument("--normalized-output", help="Optional CSV path for merged normalized per-frame scores")
    parser.add_argument("--events-output", help="Optional CSV path for per-video peak event results")
    args = parser.parse_args()

    root = project_root()
    scores = read_scores(args.scores, args.score_column)
    eval_frames = load_eval_frames(root, args.split)
    merged = merge_scores(eval_frames, scores)
    consensus = pd.read_csv(root / "annotations" / "consensus_events.csv")
    events = event_detection(merged, consensus)

    print_metrics(merged, events)

    if args.normalized_output:
        merged.to_csv(args.normalized_output, index=False)
        print(f"\nWrote normalized scores to {args.normalized_output}")

    if args.events_output:
        events.to_csv(args.events_output, index=False)
        print(f"\nWrote event results to {args.events_output}")


if __name__ == "__main__":
    main()
