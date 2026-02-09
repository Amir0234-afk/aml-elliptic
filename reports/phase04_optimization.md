# Phase 04 — Tabular Optimization (Leakage-Safe Temporal HPO + Threshold Tuning)

**File:** `src/phase04_tuning.py`  
**Outputs (primary):**
- `results/metrics/phase04_tuning_AF_<backend>.json`
- `results/metrics/phase04_tuning_LF_<backend>.json`
- `results/metrics/phase04_tuning_index_<backend>.json`

**Outputs (artifacts & diagnostics):**
- Models: `results/model_artifacts/phase04/{lr,rf}_{AF,LF}_<backend>.joblib`
- LR scalers: `results/model_artifacts/phase04/lr_scaler_{AF,LF}.joblib`
- Thresholds: `results/model_artifacts/phase04/{lr,rf}_threshold_{AF,LF}_<backend>.json`
- Plots: `results/visualizations/phase04/*.png`
- Artifact inventory: `results/logs/phase04_artifacts_<backend>_<timestamp>.json`
- Run manifest: `results/logs/phase04_<timestamp>.json` (via `write_run_manifest`)

---

## 1. Objective

Phase 04 produces **final, leakage-safe optimized tabular baselines** for illicit transaction detection on the Elliptic dataset under a **temporal split**. The phase addresses two core requirements:

1. **Temporal integrity:** model selection must not exploit future information (no “peeking” into later time steps).
2. **Operational decision rule:** probability outputs must be converted into labels using a **threshold** chosen without test leakage, and ideally tuned for the target metric (**F1 on the illicit class**).

Concretely, Phase 04:
- tunes **Logistic Regression (LR)** hyperparameters (regularization strength `C`) under **forward-chaining temporal CV**;
- tunes **Random Forest (RF)** hyperparameters under the same temporal CV regime, using **backend-aware search spaces**;
- performs **threshold tuning** on a strictly-held-out **tail slice inside training** (not on test);
- exports **reproducible model artifacts**, **plots**, and **JSON reports** for downstream evaluation/reporting.

---

## 2. Experimental Design and Data Protocol

### 2.1 Dataset Inputs

Phase 04 operates on the labeled subset exported in Phase 01:
- preferred: `data/processed/elliptic_labeled.parquet`
- fallback: `data/processed/elliptic_labeled.csv`

The dataset is expected to contain:
- `txId` (transaction identifier)
- `time_step` in `[1..49]` (temporal index)
- `class` in `{0,1}` for labeled data (0 = licit, 1 = illicit)
- feature columns (determined by feature mode)

If labeled data is missing, Phase 04 instructs running Phase 01 preprocessing.

### 2.2 Feature Modes

Phase 04 evaluates **two feature modes**, consistent with earlier phases:

- **AF (All Features):** full set of engineered/available features (e.g., local + aggregated).
- **LF (Local Features):** reduced set focusing on local/transaction-level features.

Feature selection is delegated to:
- `get_feature_cols(df, feature_mode)` from `src.data`.

### 2.3 Temporal Train/Test Split

To preserve causality, Phase 04 performs a **time-step split**:

- Let `steps = sorted(unique(time_step))`.
- Let `n_train = floor(len(steps) * train_ratio)`.
- Then:
  - `train_steps = steps[:n_train]`
  - `test_steps  = steps[n_train:]`

This means the model is trained only on early steps and evaluated on later steps, matching real-world “predict forward in time” conditions.

### 2.4 Class Imbalance Handling

The illicit class is typically the minority class. Phase 04 computes class weights on the training window:

- `cw = class_weights_binary(y_train)`

These weights are passed into LR and RF training through the backend interface, enforcing a cost-sensitive objective.

---

## 3. Leakage-Safe Model Selection (Forward-Chaining Temporal CV)

### 3.1 Motivation

Standard random cross-validation is invalid for time-evolving graphs/transactions because it mixes future and past. Phase 04 uses **forward-chaining CV** to enforce temporal order:

- Each fold trains on earlier time steps and validates on the immediately following step(s).

This prevents:
- training on information that occurs after validation,
- hyperparameter selection that benefits from temporal leakage.

### 3.2 CV Split Construction

Given:
- `train_steps` (time steps in training window),
- `row_steps` (the time_step per training row),

Phase 04 constructs `n_splits` folds, each defined by a train end index and a validation window size `val_steps`:

- fold k:
  - train on `steps[:train_end]`
  - validate on `steps[train_end : train_end + val_steps]`

The fold returns row indices into `X_train_raw` / `y_train`.

If the training window is too small to construct splits, Phase 04 fails fast with a clear error.

---

## 4. Threshold Tuning Without Test Leakage

### 4.1 Motivation

Optimizing `F1_illicit` requires selecting a classification threshold `t` for the positive class probability. However, tuning `t` directly on the test set would leak evaluation information.

### 4.2 Tail Split Strategy

Phase 04 creates a **tail slice within training**:

- `tail_steps_count = tabular_cv_val_steps` (typically 1)
- `tail_steps` = last `tail_steps_count` steps from `train_steps`
- `base_steps` = remaining earlier steps in training

Then:
- **base** is used to fit the model for threshold selection,
- **tail** is used exclusively to select `t` maximizing `F1_illicit`.

This yields a threshold tuned on “recent past” while strictly preserving the test window as unseen future.

### 4.3 Threshold Search Method

For a vector of predicted positive probabilities `p_pos`, Phase 04 finds the threshold maximizing illicit F1:

- candidate thresholds are the **unique probability values** (or a quantile-compressed set if too many).
- for each threshold:
  - `pred = threshold_predictions(p_pos, t)`
  - `metrics = compute_metrics(y_true, pred)`
  - choose `t` maximizing `metrics.f1_illicit`

This produces:
- `best_threshold`
- `best_tail_f1_illicit`

---

## 5. Models and Optimization Procedures

Phase 04 optimizes **two model families** per feature mode (AF and LF), using the configured tabular backend.

### 5.1 Backend Abstraction

All model training and inference is executed through:

- `backend = get_tabular_backend()`
- interface: `TabularBackend`

This allows switching between:
- CPU sklearn-like training
- GPU-accelerated training (e.g., `cuml`)

The backend exposes:
- `train_lr(...)`, `train_rf(...)`
- `predict_proba_positive(model, X)`

Phase 04 is explicitly **backend-aware**, especially for RF hyperparameter grids (see §5.3).

---

## 5.2 Logistic Regression (LR) Optimization

#### 5.2.1 Preprocessing
LR uses standardization:
- fit `mu, sd` on training fold
- transform validation fold with those parameters

This prevents leakage from validation into scaling statistics.

#### 5.2.2 Hyperparameter Search Space
LR tunes only:
- `C` ∈ logspace(-3, 2) (24 values)

Trials:
- `tabular_tune_trials` random samples from the grid (without replacement)

#### 5.2.3 CV Objective
For each candidate `C`:
- for each temporal fold:
  - fit LR on fold-train (standardized)
  - predict probs on fold-val
  - compute the best threshold on fold-val (maximizing `F1_illicit`)
  - record fold F1
- average fold F1 → `cv_f1_illicit`

This yields:
- best `C` by mean CV F1
- a ranked list `trials_top10`

#### 5.2.4 Threshold Tuning (Tail)
After selecting best `C`:
- fit scaler on `base_steps`
- train LR on standardized base
- predict on standardized tail
- choose threshold maximizing tail `F1_illicit`

#### 5.2.5 Final Training and Test Evaluation
- fit scaler on **all training steps**
- train LR on all training rows
- predict probabilities on test rows
- apply the tail-tuned threshold
- compute test metrics

LR artifacts persisted:
- model: `lr_<mode>_<backend>.joblib`
- scaler: `lr_scaler_<mode>.joblib`
- threshold: `lr_threshold_<mode>_<backend>.json`

---

## 5.3 Random Forest (RF) Optimization

RF optimization is performed under the same **forward-chaining temporal CV**, but the search space differs by backend.

### 5.3.1 Backend-Aware Search Space

**If backend is `cuml`:**
- `max_features` ∈ {"sqrt", 0.3, 0.5}
- `max_depth` ∈ {10, 16, 20, 30, 40} (avoid `None` for stability)
- `min_samples_leaf` fixed to 1
- `min_samples_split` fixed to 2
- `max_samples` fixed to `None`

**Else (sklearn-like):**
- `max_features` ∈ {"sqrt", "log2", 0.3, 0.5, 0.7}
- `max_depth` ∈ {None, 10, 20, 30, 40}
- `min_samples_leaf` ∈ {1, 2, 5, 10}
- `min_samples_split` ∈ {2, 5, 10}
- `max_samples` ∈ {None, 0.7, 0.85, 0.95}

Always:
- `n_estimators` ∈ {200, 400, 600, 800, 1000}

Trials:
- `tabular_tune_trials` random draws from this parameter space.

### 5.3.2 CV Objective
For each sampled parameter set:
- for each temporal fold:
  - train RF on fold-train
  - predict probs on fold-val
  - pick best threshold on fold-val (max F1)
  - record fold F1
- average fold F1 → `cv_f1_illicit`

Select the best RF parameter set by mean CV illicit F1.

### 5.3.3 Threshold Tuning (Tail)
After selecting best RF parameters:
- train RF on `base_steps`
- predict on tail
- tune threshold maximizing tail `F1_illicit`

### 5.3.4 Final Training and Test Evaluation
- train RF on **all training steps**
- predict probabilities on test steps
- apply tuned threshold
- compute test metrics

RF artifacts persisted:
- model: `rf_<mode>_<backend>.joblib`
- threshold: `rf_threshold_<mode>_<backend>.json`

---

## 6. Diagnostics and Visualizations

For each (mode, model) pair Phase 04 generates:

1. **Precision–Recall curve** on test:
   - `pr_lr_<mode>_<backend>.png`
   - `pr_rf_<mode>_<backend>.png`

2. **Threshold sweep** on tail steps:
   - F1 / Precision / Recall vs threshold
   - `thr_sweep_lr_<mode>_<backend>.png`
   - `thr_sweep_rf_<mode>_<backend>.png`

3. **Calibration curve** on test (quantile binning):
   - `calib_lr_<mode>_<backend>.png`
   - `calib_rf_<mode>_<backend>.png`

4. **Confusion matrix (raw + normalized)** on test:
   - `cm_lr_<mode>_<backend>.png` and `_norm.png`
   - `cm_rf_<mode>_<backend>.png` and `_norm.png`

These plots support:
- threshold interpretability (operating point selection),
- precision/recall tradeoff inspection (PR),
- probability reliability assessment (calibration),
- error decomposition (confusion matrices).

---

## 7. Reproducibility and Artifact Accounting

### 7.1 Determinism and Seeds
Phase 04 sets:
- global seed via `set_seed(cfg.seed, deterministic_torch=False)`
- local RNG via `np.random.default_rng(cfg.seed)`

Randomness sources include:
- hyperparameter sampling for LR and RF
- (backend-dependent) RF training randomness

### 7.2 Run Manifest
At the beginning of the run, Phase 04 writes a manifest including:
- run timestamp (UTC)
- phase identifier
- config snapshot
- existence/metadata of required input files
- backend name and a phase note

This is written to `results/logs/` via `write_run_manifest(...)`.

### 7.3 Artifact Inventory
After completion, Phase 04 scans key output directories and records:
- path
- byte size
- SHA-256 hash

This produces:
- `results/logs/phase04_artifacts_<backend>_<timestamp>.json`

The inventory enables:
- auditing which files were produced,
- content-addressable verification that files did not change,
- robust comparison across runs.

---

## 8. Outputs and Report Schema

### 8.1 Per-Mode Metric Reports
For each feature mode:
- `results/metrics/phase04_tuning_<mode>_<backend>.json`

Contains:
- split definition (`train_steps`, `test_steps`)
- CV details (`n_splits`, `val_steps`, number of folds)
- class weights
- LR results:
  - best `C`, best CV illicit F1
  - tuned threshold, tail F1
  - test metrics (precision/recall/F1, micro F1, confusion matrix)
  - top trials
- RF results:
  - best RF params, best CV illicit F1
  - tuned threshold, tail F1
  - test metrics
  - top trials

### 8.2 Combined Index Report
- `results/metrics/phase04_tuning_index_<backend>.json`

Contains:
- config snapshot (as dict)
- backend name
- nested reports for AF and LF

This file serves as the single entry point for downstream summaries.

---

## 9. Summary of Scientific Rationale

Phase 04 is designed to produce **credible tabular baselines** under a strict temporal regime:

- **Forward-chaining CV** ensures hyperparameter selection respects time ordering.
- **Tail-step threshold tuning** gives an operational decision threshold without test leakage.
- **Backend abstraction** makes the pipeline portable across CPU and GPU implementations while keeping the optimization logic consistent.
- **Model artifacts + plots + inventories** make results reproducible, auditable, and report-ready.

As a result, Phase 04 yields:
- optimized LR and RF configurations for AF and LF feature modes,
- tuned thresholds aligned with the project’s target metric (illicit F1),
- a complete set of diagnostics enabling rigorous interpretation and comparison with graph-based models in later phases.
