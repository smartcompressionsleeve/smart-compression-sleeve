# ML Pipeline

Python scripts for training the per-exercise Random Forest classifier from
labeled sensor trials. Includes feature extraction, GroupKFold cross-validation
(by trial, to prevent leakage), and confusion matrix generation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Models

Three per-exercise classifiers, dispatched by exercise ID at inference time:

| Exercise | Accuracy (GroupKFold, by trial) | Notes |
|---|---|---|
| Bicep curl | **96.6% ± 3.5%** (N=417 reps) | Primary validated result |
| Hammer curl | 98.9% ± 0.9% | |
| Tricep extension | 90.2% ± 12.5% | High variance: limited bad-form demonstrators |

## Features

Classifier inputs are IMU-derived per-rep features (elbow-angle kinematics,
gyroscope-velocity statistics). EMG is captured as an envelope for activation
display only — it is **not** an input to the classifier.

## Usage

See `train_classifier.py` for the training entry point. Raw trial CSVs go in
`data/` (gitignored by default).
