"""
Evaluate AURA per-frame anomaly scores against soft labels and event intervals.

The score CSV must contain one row per evaluated frame with scene, video,
frame_idx, and an anomaly score column. Common score column names such as
score, anomaly_score, pred_score, and image_score are detected automatically.

By default this script evaluates all rows in annotations/soft_labels.csv, which
matches the paper-style evaluation. Use --eval-scope split_test for the smaller
held-out debug set in splits/<split>/test.csv.

Usage:
    python evaluate_scores.py --scores rd_split2_scores.csv --split split_2
    python evaluate_scores.py --scores scores.csv --score-column image_score
    python evaluate_scores.py --scores scores.csv --out-dir results/rd_split2
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths


SCENES = ("scene_A", "scene_B")
SCORE_COLUMN_CANDIDATES = ("score", "anomaly_score", "pred_score", "image_score")
REFERENCE_TARGETS = {
    "split_2": {
        "Reverse Distillation": {
            "mae": {"scene_A": 0.271, "scene_B": 0.259},
            "peak_tiou": {"scene_A": 0.673, "scene_B": 0.861},
        }
    }
}


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

    duplicate_keys = scores.duplicated(["scene", "video", "frame_idx"]).sum()
    if duplicate_keys:
        raise ValueError(f"Scores file has {duplicate_keys} duplicated scene/video/frame_idx rows")

    return scores


def load_eval_frames(root, split, eval_scope):
    if eval_scope == "all_labeled":
        path = root / "annotations" / "soft_labels.csv"
    else:
        path = root / "splits" / split / "test.csv"

    if not path.exists():
        raise FileNotFoundError(f"Evaluation frame file not found: {path}")

    return pd.read_csv(path)[["scene", "video", "frame_idx", "soft_label"]]


def normalize_per_video(df):
    df = df.sort_values(["scene", "video", "frame_idx"]).copy()
    grouped = df.groupby(["scene", "video"])["score"]
    mins = grouped.transform("min")
    maxs = grouped.transform("max")
    df["score_norm"] = (df["score"] - mins) / (maxs - mins + 1e-8)
    return df


def merge_scores(eval_frames, scores):
    merged = eval_frames.merge(scores, on=["scene", "video", "frame_idx"], how="left")
    missing = merged["score"].isna().sum()
    if missing:
        examples = merged[merged["score"].isna()][["scene", "video", "frame_idx"]].head()
        raise ValueError(f"Missing scores for {missing} eval frames. Examples:\n{examples.to_string(index=False)}")
    return normalize_per_video(merged)


def per_video_mae(merged):
    rows = []
    for (scene, video), video_df in merged.groupby(["scene", "video"]):
        mae = np.abs(video_df["score_norm"] - video_df["soft_label"]).mean()
        rows.append({"scene": scene, "video": video, "frames": len(video_df), "mae": mae})
    return pd.DataFrame(rows)


def mae_summary(per_video):
    rows = []
    for scene, scene_df in per_video.groupby("scene"):
        rows.append(
            {
                "scene": scene,
                "videos": len(scene_df),
                "mae_mean": scene_df["mae"].mean(),
                "mae_std": scene_df["mae"].std(ddof=0),
            }
        )
    return pd.DataFrame(rows)


def temporal_iou(pred_start, pred_end, true_start, true_end):
    overlap = max(0.0, min(pred_end, true_end) - max(pred_start, true_start))
    union = max(pred_end, true_end) - min(pred_start, true_start)
    if union <= 0:
        return 0.0
    return overlap / union


def longest_threshold_interval(frame_indices, scores_norm, threshold):
    mask = scores_norm >= threshold
    if not mask.any():
        peak = int(np.argmax(scores_norm))
        frame = float(frame_indices[peak])
        return frame, frame

    best_start = best_end = None
    best_length = -1
    start = None

    for index, is_anomalous in enumerate(mask):
        if is_anomalous and start is None:
            start = index
        if start is not None and (not is_anomalous or index == len(mask) - 1):
            end = index if is_anomalous and index == len(mask) - 1 else index - 1
            length = end - start + 1
            if length > best_length:
                best_start = start
                best_end = end
                best_length = length
            start = None

    return float(frame_indices[best_start]), float(frame_indices[best_end])


def widest_peak_interval(frame_indices, scores_norm, relative_height):
    peaks, _ = find_peaks(scores_norm)

    if len(peaks) == 0:
        peak = int(np.argmax(scores_norm))
        frame = float(frame_indices[peak])
        return frame, frame

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        widths, _, left_ips, right_ips = peak_widths(scores_norm, peaks, rel_height=relative_height)

    widest = int(np.argmax(widths))
    x = np.arange(len(scores_norm))
    pred_start = float(np.interp(left_ips[widest], x, frame_indices))
    pred_end = float(np.interp(right_ips[widest], x, frame_indices))
    return pred_start, pred_end


def predict_event(method, frame_indices, scores_norm, parameter):
    if method == "threshold":
        return longest_threshold_interval(frame_indices, scores_norm, parameter)
    if method == "peak":
        return widest_peak_interval(frame_indices, scores_norm, parameter)
    raise ValueError(f"Unknown event method: {method}")


def parameter_sweep(merged, consensus):
    rows = []
    parameters = np.round(np.arange(0.0, 1.0001, 0.01), 2)

    for method in ("threshold", "peak"):
        for parameter in parameters:
            for (scene, video), video_df in merged.groupby(["scene", "video"]):
                true_row = consensus[(consensus["scene"] == scene) & (consensus["video"] == video)]
                if true_row.empty:
                    continue

                video_df = video_df.sort_values("frame_idx")
                frame_indices = video_df["frame_idx"].to_numpy(dtype=float)
                scores_norm = video_df["score_norm"].to_numpy(dtype=float)
                true_start = float(true_row.iloc[0]["start_frame"])
                true_end = float(true_row.iloc[0]["end_frame"])
                pred_start, pred_end = predict_event(method, frame_indices, scores_norm, parameter)

                rows.append(
                    {
                        "method": method,
                        "scene": scene,
                        "video": video,
                        "parameter": parameter,
                        "pred_start": pred_start,
                        "pred_end": pred_end,
                        "true_start": true_start,
                        "true_end": true_end,
                        "tiou": temporal_iou(pred_start, pred_end, true_start, true_end),
                    }
                )

    predictions = pd.DataFrame(rows)
    summary = (
        predictions.groupby(["method", "scene", "parameter"], as_index=False)
        .agg(mean_tiou=("tiou", "mean"), std_tiou=("tiou", lambda values: values.std(ddof=0)), videos=("video", "nunique"))
        .sort_values(["method", "scene", "mean_tiou", "parameter"], ascending=[True, True, False, True])
    )
    return predictions, summary


def best_scene_parameters(sweep_summary):
    return sweep_summary.groupby(["method", "scene"], as_index=False).first()


def best_event_predictions(all_predictions, best_parameters):
    rows = []
    for _, best in best_parameters.iterrows():
        selected = all_predictions[
            (all_predictions["method"] == best["method"])
            & (all_predictions["scene"] == best["scene"])
            & (all_predictions["parameter"] == best["parameter"])
        ].copy()
        selected["scene_mean_tiou"] = best["mean_tiou"]
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def weighted_mae_by_scene(merged):
    errors = merged.assign(abs_error=(merged["score_norm"] - merged["soft_label"]).abs())
    return errors.groupby("scene")["abs_error"].mean().to_dict()


def build_summary(args, merged, per_video, mae_by_scene, best_parameters):
    summary = {
        "scores": args.scores,
        "split": args.split,
        "eval_scope": args.eval_scope,
        "frames": int(len(merged)),
        "videos_by_scene": {scene: int(count) for scene, count in merged.groupby("scene")["video"].nunique().items()},
        "mae_macro_by_scene": {},
        "mae_frame_weighted_by_scene": weighted_mae_by_scene(merged),
        "event_detection": {},
        "reference_targets": REFERENCE_TARGETS.get(args.split, {}),
    }

    for row in mae_by_scene.to_dict(orient="records"):
        summary["mae_macro_by_scene"][row["scene"]] = {
            "mean": row["mae_mean"],
            "std": row["mae_std"],
            "videos": int(row["videos"]),
        }

    for row in best_parameters.to_dict(orient="records"):
        method = row["method"]
        summary["event_detection"].setdefault(method, {})
        summary["event_detection"][method][row["scene"]] = {
            "parameter": row["parameter"],
            "mean_tiou": row["mean_tiou"],
            "std_tiou": row["std_tiou"],
            "videos": int(row["videos"]),
        }

    return summary


def print_summary(summary):
    print(f"AURA evaluation ({summary['eval_scope']}, {summary['split']})")
    print(f"Frames: {summary['frames']}")
    print("Videos:")
    for scene in SCENES:
        print(f"  {scene}: {summary['videos_by_scene'].get(scene, 0)}")
    print()

    print("MAE vs soft labels (macro over videos)")
    for scene in SCENES:
        if scene not in summary["mae_macro_by_scene"]:
            continue
        values = summary["mae_macro_by_scene"][scene]
        weighted = summary["mae_frame_weighted_by_scene"][scene]
        print(
            f"  {scene}: {values['mean']:.4f} +/- {values['std']:.4f} "
            f"({values['videos']} videos; frame-weighted {weighted:.4f})"
        )
    print()

    for method, label in (("threshold", "Threshold event t-IoU"), ("peak", "Peak-finding event t-IoU")):
        print(label)
        method_summary = summary["event_detection"].get(method, {})
        for scene in SCENES:
            if scene not in method_summary:
                continue
            values = method_summary[scene]
            print(
                f"  {scene}: {values['mean_tiou']:.4f} +/- {values['std_tiou']:.4f} "
                f"at parameter {values['parameter']:.2f} ({values['videos']} videos)"
            )
        print()

    reference = summary.get("reference_targets", {}).get("Reverse Distillation")
    if reference:
        print("Reference targets: Reverse Distillation")
        for scene in SCENES:
            mae = reference["mae"].get(scene)
            peak_tiou = reference["peak_tiou"].get(scene)
            if mae is not None and peak_tiou is not None:
                print(f"  {scene}: MAE ~ {mae:.3f}, peak t-IoU ~ {peak_tiou:.3f}")


def write_outputs(out_dir, merged, per_video, sweep_summary, event_predictions, summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "normalized_scores.csv", index=False)
    per_video.to_csv(out_dir / "per_video_mae.csv", index=False)
    sweep_summary.to_csv(out_dir / "parameter_sweep.csv", index=False)
    event_predictions.to_csv(out_dir / "event_predictions.csv", index=False)
    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate AURA per-frame anomaly scores")
    parser.add_argument("--scores", required=True, help="CSV with scene, video, frame_idx, and anomaly score columns")
    parser.add_argument("--score-column", help="Name of anomaly score column in --scores")
    parser.add_argument("--split", default="split_2", choices=("split_1", "split_2"), help="Training split metadata")
    parser.add_argument(
        "--eval-scope",
        default="all_labeled",
        choices=("all_labeled", "split_test"),
        help="Evaluate all soft-labeled videos or only splits/<split>/test.csv",
    )
    parser.add_argument("--out-dir", help="Optional directory for detailed evaluation outputs")
    args = parser.parse_args()

    root = project_root()
    scores = read_scores(args.scores, args.score_column)
    eval_frames = load_eval_frames(root, args.split, args.eval_scope)
    merged = merge_scores(eval_frames, scores)
    consensus = pd.read_csv(root / "annotations" / "consensus_events.csv")

    video_mae = per_video_mae(merged)
    mae_by_scene = mae_summary(video_mae)
    all_predictions, sweep_summary = parameter_sweep(merged, consensus)
    best_parameters = best_scene_parameters(sweep_summary)
    event_predictions = best_event_predictions(all_predictions, best_parameters)
    summary = build_summary(args, merged, video_mae, mae_by_scene, best_parameters)

    print_summary(summary)

    if args.out_dir:
        write_outputs(Path(args.out_dir), merged, video_mae, sweep_summary, event_predictions, summary)
        print(f"\nWrote evaluation outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
