"""
Organize AURA dataset for anomalib using pre-defined splits.

Usage: python setup_anomalib.py --method symlink
"""

import argparse
import shutil
import sys
from pathlib import Path
import pandas as pd


def setup_anomalib_dataset(method='symlink'):
    # Check if frames exist
    if not Path('scene_A/frames').exists() or not Path('scene_B/frames').exists():
        print("Error: Run extract_frames.py first")
        sys.exit(1)
    
    # Clear output directory
    output = Path('anomalib_dataset')
    if output.exists():
        shutil.rmtree(output)
    
    # Process both scenes
    for scene in ['scene_A', 'scene_B']:
        print(f"\n{scene}:")
        
        # Create dirs
        for d in ['train/normal', 'test/normal', 'test/anomalous']:
            (output / scene / d).mkdir(parents=True, exist_ok=True)
        
        # Load CSVs
        train_df = pd.read_csv(f'{scene}/normal_frames.csv')
        test_df = pd.read_csv(f'{scene}/test_frames.csv')
        
        # Organize training frames
        train_count = 0
        for _, row in train_df.iterrows():
            src = Path(f'{scene}/frames/{row["video"]}/frame_{row["frame_idx"]:04d}.png')
            if not src.exists():
                continue
            
            dst = output / scene / 'train/normal' / f'{row["video"]}_{row["frame_idx"]:04d}.png'
            
            if method == 'symlink':
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
            train_count += 1
        
        # Organize test frames
        normal_count = 0
        anom_count = 0
        
        for _, row in test_df.iterrows():
            src = Path(f'{scene}/frames/{row["video"]}/frame_{row["frame_idx"]:04d}.png')
            if not src.exists():
                continue
            
            dst = output / scene / 'test' / row['label'] / f'{row["video"]}_{row["frame_idx"]:04d}.png'
            
            if method == 'symlink':
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
            
            if row['label'] == 'normal':
                normal_count += 1
            else:
                anom_count += 1
        
        print(f"  train: {train_count} | test normal: {normal_count} | test anomalous: {anom_count}")
    
    print(f"\n✓ Done → {output}/ (method={method})")


def main():
    parser = argparse.ArgumentParser(description='Organize AURA for anomalib')
    parser.add_argument('--method', choices=['copy', 'symlink'], default='symlink')
    args = parser.parse_args()
    
    setup_anomalib_dataset(args.method)


if __name__ == "__main__":
    main()