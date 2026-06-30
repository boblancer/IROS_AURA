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
├── scene_A/
│   ├── videos/                  # 10 videos from Anemo Robotics
│   ├── normal_frames.csv        # Normal frames for training
│   └── test_frames.csv          # Test frames with labels
└── scene_B/
    ├── videos/                  # 15 videos from Brackish dataset
    ├── normal_frames.csv        # Normal frames for training
    └── test_frames.csv          # Test frames with labels
```

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
video,start_frame,end_frame
v00,245,512
v01,280,520
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
video,frame_idx,label
v00,0,normal
v00,370,anomalous
```


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

**Training**: Use `normal_frames.csv` to train on normal scenes.

**Validation**: Use `test_frames.csv` for validation during training (required by anomalib).

**Final Evaluation**: Evaluate on complete video sequences using `soft_labels.csv`.

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
