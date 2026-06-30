"""
Extract frames from AURA videos.

This script extracts all frames from the videos and saves them as PNG images.
Frame numbering starts at 0 and matches the frame_idx in the annotation files.

Usage:
    python extract_frames.py

Output:
    scene_A/frames/v00/frame_0000.jpg
    scene_A/frames/v00/frame_0001.jpg
    ...
    scene_B/frames/v10/frame_0000.jpg
    ...
"""

import cv2
from pathlib import Path
from tqdm import tqdm


def extract_frames(video_path, output_dir):
    """
    Extract all frames from a video file.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc=f"Extracting {video_path.name}", unit="frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = output_dir / f"frame_{frame_idx:04d}.png"
        cv2.imwrite(str(frame_path), frame)
        
        frame_idx += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()
    
    return frame_idx


def main():
    """Extract frames from all videos in the dataset."""
    
    total_videos = 0
    total_frames = 0
    
    # Scene A
    print("\n=== Processing Scene A ===")
    scene_a_videos = sorted(Path("scene_A/videos").glob("*.mp4"))
    
    for video_path in scene_a_videos:
        output_dir = Path("scene_A/frames") / video_path.stem
        num_frames = extract_frames(video_path, output_dir)
        total_videos += 1
        total_frames += num_frames
    
    # Scene B
    print("\n=== Processing Scene B ===")
    scene_b_videos = sorted(Path("scene_B/videos").glob("*.mp4"))
    
    for video_path in scene_b_videos:
        output_dir = Path("scene_B/frames") / video_path.stem
        num_frames = extract_frames(video_path, output_dir)
        total_videos += 1
        total_frames += num_frames
    
    # Summary
    print("\n" + "="*50)
    print(f"✓ Extraction complete!")
    print(f"  Total videos processed: {total_videos}")
    print(f"  Total frames extracted: {total_frames}")
    print(f"  Frames saved to: scene_A/frames/ and scene_B/frames/")
    print("="*50)


if __name__ == "__main__":
    main()