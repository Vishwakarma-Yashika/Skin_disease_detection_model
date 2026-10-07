# Experiment E0 — fixed EfficientNet-B0 + ResNet-34 fusion baseline

E0 is a **fresh, fully documented baseline** of the unchanged fusion classifier
on a lesion-level HAM10000 split. It is the reference point for every later
reliability experiment (calibration, uncertainty, selective prediction, referral).

E0 deliberately contains **no** uncertainty method, MC Dropout, ensemble,
temperature scaling, conformal prediction, OOD detection, extra backbone or
accuracy tuning. Calibration and selective-prediction numbers are
*measurements of the raw softmax outputs only*.

The historical checkpoint (`backend/models/best_fusion_model.pt`, 83.5% val
accuracy) is **not** reproduced or modified. Its training notebook, split and
hyperparameters are unavailable, so it is not comparable to E0.

## Layout

```
research/
  configs/e0.yaml                 frozen E0 configuration (hashed into every run)
  splits/ham10000_lesion_split.csv          frozen split (image_id, lesion_id, dx, split)
  splits/ham10000_lesion_split_summary.json counts / class distributions / sha256
  data/data.py        metadata + file index, leakage checks, transforms, Dataset
  data/split.py       one-time split generation / --check validation
  models/fusion_model.py  imports FusionClassifier from backend/app/model.py (no copy) + verification
  training/train.py   trains one seed, selects checkpoint on validation, exports logits
  training/predict.py re-exports val/test logits from a run directory
  evaluation/metrics.py, calibration_metrics.py, selective_prediction.py, evaluate.py
  utils/              seeding, environment capture, hashing, config, JSON io
  outputs/            git-ignored run directories
```

## Setup

```bash
pip install -r backend/requirements.txt -r research/requirements.txt
export HAM10000_ROOT="/path/to/folder/with/HAM10000_metadata.csv"   # or pass --data-root
```

The dataset root must contain `HAM10000_metadata.csv`, `HAM10000_images_part_1/`
and `HAM10000_images_part_2/`. ImageNet weights are downloaded by torchvision on
first use. All commands run from the repository root.

## Running E0

```bash
# 0. Validate the frozen split (never regenerates it). Add --verify-images to decode every file.
python -m research.data.split --config research/configs/e0.yaml --check

# 1. Train each predeclared seed (one directory per seed)
for s in 42 123 2024 3407 5001; do
  python -m research.training.train --config research/configs/e0.yaml --seed $s
done

# 2. Evaluate all five seeds from the saved logits
python -m research.evaluation.evaluate --config research/configs/e0.yaml
```

`train.py` options that do not affect the experiment (recorded in
`run_metadata.json`, not in the config hash): `--data-root`, `--device`,
`--num-workers`, `--overwrite`. `--smoke-batches N` is a pipeline test only:
1 epoch, N batches, written to `outputs/smoke/`, never an E0 result.

Compute: on this project's development machine (CPU only, no CUDA) a smoke
test suggests roughly 20+ minutes per epoch, i.e. on the order of 10 hours
per seed. A CUDA GPU is strongly recommended for the five full runs.

## Protocol (what the config freezes)

| Item | E0 value |
|---|---|
| Data | HAM10000, 10,015 images, 7,470 lesions, 7 classes |
| Split | lesion-level, stratified by dx at lesion level, 70/15/15 of lesions, `random_state=42` |
| Model | backend `FusionClassifier`: EffNet-B0 (1280) ‖ ResNet-34 (512) → Linear(1792,512) → BN → ReLU → Dropout(0.3) → Linear(512,7) |
| Init | torchvision ImageNet `IMAGENET1K_V1` for both backbones |
| Input | `Resize((224,224))` (aspect not preserved), ImageNet mean/std — identical to `backend/app/model.py` (asserted) |
| Train augmentation | random horizontal flip p=0.5, random vertical flip p=0.5 |
| Loss | plain cross-entropy, no class weights, no label smoothing, uniform shuffling |
| Optimiser | AdamW, lr 1e-4, weight decay 1e-4 (all params), betas (0.9, 0.999) |
| Schedule | cosine annealing, T_max 30 epochs, eta_min 1e-6, stepped per epoch |
| Budget | 30 epochs, batch 32, `drop_last=True`, no early stopping, fp32 |
| Selection | minimum validation NLL, earliest epoch on ties |
| Seeds | 42, 123, 2024, 3407, 5001 |

These values were chosen once, before any E0 training, as conventional
fine-tuning defaults. They were not tuned on validation or test data.

## Split

Generated once by `research.data.split`, pinned by SHA-256 in `e0.yaml`.
Training and evaluation fail if the file changes. Every run re-applies these
checks and raises `LeakageError` on failure: lesion overlap, image overlap,
duplicate image IDs, images missing or extra vs. metadata, lesion_id/dx
disagreeing with metadata, a class missing from any split, missing/duplicate/
empty file references.

| split | images | lesions | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|---|---|
| train | 6981 (69.7%) | 5229 | 222 | 361 | 772 | 71 | 773 | 4683 | 99 |
| val | 1532 (15.3%) | 1120 | 53 | 82 | 160 | 24 | 173 | 1018 | 22 |
| test | 1502 (15.0%) | 1121 | 52 | 71 | 167 | 20 | 167 | 1004 | 21 |

Split-method history, for transparency: a first candidate
(`StratifiedGroupKFold`, 20 folds, 3 folds each for val/test) was generated
and rejected **before any training**, because rare classes were unevenly
distributed (df: 23 val vs 11 test images; akiec 4.3% vs 2.5%). It was
replaced by the lesion-level stratified split above. No model output was used
in this decision.

HAM10000 has no patient identifier, so patient-level leakage (different
lesions of one patient in different splits) cannot be excluded. Lesion-level
grouping is the strongest control the dataset allows.

## Outputs

```
outputs/E0/seed_<seed>/
  config.yaml, run_metadata.json   config copy; seed, environment, git commit, hashes, split summary
  history.csv / history.json       per-epoch train loss/acc, val NLL/acc/balanced acc/macro F1/AUROC, lr, time
  best.pt                          selected checkpoint (+ classes, seed, epoch, criterion, config hash, split hash)
  val_metrics_best.json
  predictions/{val,test}.npz       RAW float32 logits, float64 probs, labels, preds, max_prob, image_id, lesion_id, seed
  predictions/{val,test}.csv       same, full precision (%.17g)
  evaluation/                      metrics_{val,test}.json, bootstrap_test.json, reliability/risk-coverage plots
outputs/E0/summary/                e0_summary.json, e0_summary.md, risk_coverage_test_all_seeds.png
```

`best.pt` uses the backend's checkpoint keys (`model_state_dict`,
`diagnosis_classes`, `val_acc`, `epoch`), so it can be served through
`MODEL_PATH` without code changes. Do not replace the demo checkpoint with it
unless you decide to.

## Evaluation definitions

- **Classification:** accuracy, balanced accuracy (mean recall), macro P/R/F1,
  weighted F1, per-class P/R/F1/specificity, confusion matrix (rows = true),
  one-vs-rest ROC-AUC per class and macro. Precision of a never-predicted class is 0.
- **Reliability (raw softmax):** NLL (from logits, float64); Brier = mean Σ_k (p_k − y_k)², range [0, 2];
  top-label ECE/MCE with 15 equal-width bins (i/15, (i+1)/15]; classwise ECE (Kull et al., 2019) = mean of per-class ECEs;
  reliability diagrams.
- **Selective prediction:** confidence = raw max softmax probability; risk = error rate of accepted samples;
  AURC = mean selective risk over k = 1..N; E-AURC = AURC − oracle AURC; risk at fixed coverages.
- **Malignant vs benign:** mapping from `backend/app/class_info.py` (`category`) and asserted equal to the config:
  malignant = akiec, bcc, mel; benign = bkl, df, nv, vasc. Decision = argmax class is malignant; AUROC score = sum of
  malignant-class probabilities.
- **Melanoma operating point:** mel-vs-rest on p(mel); threshold = smallest value reaching specificity ≥ 0.90 on
  **validation**, applied unchanged to test; flagged insufficient if test has < 30 melanoma lesions.
- **Statistics:** seed variation = mean ± SD (ddof = 1) of the five per-seed test estimates. Bootstrap = percentile 95% CI
  per seed from 2,000 resamples of **lesions** (images of one lesion are not independent). The two are reported separately.

## Known limitations

- No patient identifier in HAM10000 (see Split).
- Single-source dataset. There is no external test set; the official ISIC 2018 Task 3 test set is a candidate for later work.
- akiec (actinic keratosis / Bowen's disease) is counted as malignant because that is the project's declared mapping.
  Actinic keratosis is clinically pre-malignant, so state this in the paper.
- GPU runs use deterministic settings with `warn_only=True`, so a few CUDA kernels can still be nondeterministic.
  Results are reproducible up to that, and exactly reproducible only on identical hardware and software.
- `run_metadata.json` records `git_dirty`. Commit `research/` before the real runs so each run maps to a commit.

## STEP 1 — Baseline reliability of the existing checkpoint (evaluation only)

Evaluates the historical `backend/models/best_fusion_model.pt` without retraining or
calibrating it: raw logits, classification, calibration *measurements*, confidence /
entropy / top-2-margin analysis, high-confidence errors, MSP risk–coverage, and
malignant-vs-benign. Config: `configs/step1_reliability.yaml`. Code:
`evaluation/step1_reliability.py`, `reliability_analysis.py`, `reliability_figures.py`.

```bash
# Existing checkpoint on the E0 split -- labelled cross-protocol (see below)
python -m research.evaluation.step1_reliability --e0-split test --data-root <HAM10000 root>
# Existing checkpoint on an explicit manifest: image_id, lesion_id, label, image_path
python -m research.evaluation.step1_reliability --manifest evaluation_manifest.csv
```

Outputs go to `outputs/reliability_step1/` (git-ignored): `metadata/`, `predictions/`
(raw logits `.npz` + CSV), `metrics/`, `figures/` (PNG + PDF), `reports/step1_report.md`.

The historical model's training/validation split cannot be reconstructed, and its
checkpoint arithmetic suggests it used all of HAM10000. Results on any HAM10000 subset,
including the E0 split, are therefore a cross-protocol evaluation on largely
seen images, not a held-out test of that model. Use `--manifest` with images that
model cannot have seen (e.g. the ISIC 2018 Task 3 test set) for a genuinely held-out evaluation.
