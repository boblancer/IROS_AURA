# AURA: Anomalous Underwater Reef Activity

First multi-annotator benchmark for visual anomaly detection in underwater scenes.

## Overview

AURA contains underwater video footage from two marine locations with annotated anomalous events (fish, crabs, and other biological activity).

- **25 videos** (10 from Scene A, 15 from Scene B)
- **15,083 total frames**
- **16 annotators** per video
- **Soft labels** capturing annotation uncertainty
- **Consensus event boundaries** for temporal evaluation

## Dataset Structure
```
AURA/
├── annotations/
│   ├── soft_labels.csv          # Frame-level soft labels (0.0-1.0)
│   └── consensus_events.csv     # Event start/end frames
├── splits/
│   ├── split_1/
│   │   ├── train.csv            # Paper Split 1 normal training frames
│   │   └── test.csv             # Held-out soft-labeled test frames
│   └── split_2/
│       ├── train.csv            # Paper Split 2 normal training frames
│       └── test.csv             # Held-out soft-labeled test frames
├── scene_A/
│   ├── videos/                  # 10 videos from Anemo Robotics
│   ├── normal_frames.csv        # Normal frames for training
│   └── test_frames.csv          # Test frames with labels
└── scene_B/
    ├── videos/                  # 15 videos from Brackish dataset
    ├── normal_frames.csv        # Normal frames for training
    └── test_frames.csv          # Test frames with labels
```

## Installation

```bash
pip install -r reqs.txt
```

`scipy` is required for the peak-finding event evaluation.

## Scenes

**Scene A (Anemo)**: 10 videos from Hundested Harbour, Denmark at 11m depth. Features artificial reef, sandy bottom, water column, and harbor wall with varying visibility and marine snow.

**Scene B (Brackish)**: 15 videos from Limfjords-bridge, Denmark at 9m depth. Features seafloor environment and benthic habitat with consistent visual conditions.

## Files

### soft_labels.csv
Frame-level soft labels (proportion of annotators marking each frame as anomalous):
```csv
scene,video,frame_idx,soft_label
scene_A,v00,0,0.0000
scene_A,v00,370,0.5000
scene_A,v00,510,1.0000
```

### consensus_events.csv
Consensus start/end frames for anomalous events (averaged across annotators):
```csv
scene,video,start_frame,end_frame
scene_A,v00,225,419
scene_A,v01,308,655
```

### normal_frames.csv
Normal (non-anomalous) frames for training:
```csv
video,frame_idx
v02,0
v02,1
```

### test_frames.csv
Test frames with labels:
```csv
video,frame_idx
v00,0
v00,1
```

### splits
Paper train/test split CSVs:
```csv
scene,video,frame_idx,soft_label,label
scene_A,v00,0,0.0,normal
scene_A,v00,370,0.5,anomalous
```

`split_1` and `split_2` differ only in the normal training videos/images:

| Split | Scene A training videos | Scene A images | Scene B training videos | Scene B images |
|-------|--------------------------|---------------:|--------------------------|---------------:|
| split_1 | v02, v03, v06, v09 | 3,387 | v10, v12, v13, v18, v20 | 508 |
| split_2 | v01, v02, v03, v05, v06, v08, v09 | 6,516 | v10, v12, v13, v14, v15, v18, v20, v21, v23, v24 | 844 |

Both splits use the same held-out soft-labeled test frames for final evaluation.

## Usage

### Extract Frames
```bash
python extract_frames.py
```

### Setup for Anomalib
```bash
python setup_anomalib.py --method symlink
```

Then use with anomalib:
```python
from anomalib.data import Folder

datamodule = Folder(
    root="anomalib_dataset/scene_A",
    normal_dir="train/normal",
    abnormal_dir="test/anomalous",
)
```

## Training and Evaluation

**Training**: Use `splits/split_1/train.csv` or `splits/split_2/train.csv` to train on normal frames only.

**Split 1**: Smaller paper training set.

**Split 2**: Larger paper training set used for the Reverse Distillation reference numbers.

**Final Evaluation**: Run inference on every row in the split's `test.csv`, then evaluate the per-frame anomaly scores against soft labels and consensus events.

### Evaluate Scores

`evaluate_scores.py` expects a CSV with one anomaly score per frame:

```csv
scene,video,frame_idx,pred_score
scene_A,v00,0,0.123
scene_A,v00,1,0.128
```

Required columns are `scene`, `video`, `frame_idx`, and a score column. The score column is detected automatically if it is named `score`, `anomaly_score`, `pred_score`, or `image_score`.

Evaluate Split 2 scores:

```bash
python evaluate_scores.py --scores rd_split2_scores.csv --split split_2
```

Use a custom score column:

```bash
python evaluate_scores.py \
  --scores rd_split2_scores.csv \
  --split split_2 \
  --score-column my_score
```

Write normalized per-frame scores and peak-finding event predictions:

```bash
python evaluate_scores.py \
  --scores rd_split2_scores.csv \
  --split split_2 \
  --normalized-output rd_split2_normalized.csv \
  --events-output rd_split2_events.csv
```

The evaluator normalizes scores to `[0, 1]` per video, reports MAE against `soft_label`, sweeps peak relative height from `0.00` to `1.00`, and reports temporal IoU against `consensus_events.csv`.

**Important**: Some frames appear in both training and evaluation sets. This is intentional: models must detect when anomalies occur within learned scenes.

## Statistics

| Scene | Videos | Total Frames | Normal Frames | Anomalous Frames |
|-------|--------|--------------|---------------|------------------|
| A     | 10     | 12,524       | 9,910         | 883              |
| B     | 15     | 2,559        | 1,170         | 628              |

## Citation
```bibtex
@inproceedings{weihl2025aura,
  title={Uncovering Anomalous Events for Marine Environmental Monitoring via Visual Anomaly Detection},
  author={Weihl, Laura and Bengtson, Stefan Hein and Novak, Nejc and Pedersen, Malte},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  pages={2085--2094},
  year={2025}
}
```

## License

CC BY 4.0

## Acknowledgments

Funded by the European Union's Horizon 2020 under Marie Skłodowska-Curie grant No. 956200 and AI Denmark (Danish Industry Foundation). Scene B from Brackish dataset (Pedersen et al., 2019).


## Version

v1.0 (January 2025): Initial release 
