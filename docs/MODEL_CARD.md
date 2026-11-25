# Model and experiment card

## Intended use

Normal-only anomaly ranking, image alarms and localization. This is a compact research and engineering study for technical review and reproducibility.

## Architecture

Fixed patch memory, normal-trained encoder memory and convolutional reconstruction baseline.

## Recorded data and fitting

5 or 15 normal fit images, 32 additional normal calibration images and 80 mixed procedural test images; three seeds.

Farthest-first coreset or 25 autoencoder epochs; image threshold from normal-only split-conformal order statistic.

## Evaluation and observed evidence

Image AUROC/AP, pixel AUROC/AP and Dice, threshold recall/false alarms, image bootstrap and Wilson intervals.

At 15 shots / 256 prototypes, fixed descriptors reach image AUROC 1.0 but pixel AP about 0.460. Encoder recall remains low at its calibrated threshold.

Raw counts and configuration are in `results/demo/metrics.json`; a readable comparison is in `results/demo/EXPERIMENTS.md`. The recorded runtime is CPU PyTorch 2.5.1+cpu with Python 3.12.14. Runtime timestamps identify when the recorded measurements were taken.

## Verification coverage

Unique coreset indices, monotone radius, chunked distance agreement, finite-sample calibration and unfitted scoring.

The supplied GitHub workflow is prepared configuration. Local checks do not imply a remote workflow has already run.

## Material limitations

No pretrained ImageNet features, PatchCore benchmark reproduction, guarantee under distribution shift or total-data budget of 15 images.

The procedural generator is narrow and has exact generated masks. It cannot represent the full variation in materials, optics, defect severity, annotation quality or production batches. Real deployment requires independent representative data and an operating point tied to inspection costs. No proprietary/customer dataset is bundled.

## Data handling

Private datasets stay outside version control via `data/` and generated private runs via `runs/`. The manifest audit rejects structural leaks but cannot validate the meaning of manually assigned batch groups or detect all near-duplicates. Do not publish private run folders that contain sensitive sample IDs or paths.
