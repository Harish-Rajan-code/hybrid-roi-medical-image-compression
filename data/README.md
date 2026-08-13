# Dataset

The project uses medical chest X-ray images and corresponding ROI masks.

The datasets are not included in this repository.

For ROI segmentation training, the expected structure is:

data/
├── images/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
│
└── masks/
    ├── image_001_mask.png
    ├── image_002_mask.png
    └── ...

For Transformer-based ROI compression, extracted ROI patches should be organized as:

roi-patches/
├── train/
│   ├── roi_001.png
│   ├── roi_002.png
│   └── ...
│
└── val/
    ├── roi_001.png
    ├── roi_002.png
    └── ...

The dataset is not included due to size and dataset distribution/licensing considerations.
