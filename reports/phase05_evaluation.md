# Phase 05 — Evaluation & Inference Report (`phase05_evaluation.md`)

**Project:** AML Elliptic Bitcoin Transaction Classification  
**Phase:** 05 (Final evaluation + inference exports)  
**Primary script:** `src/phase05_eval_infer.py`  
**Key outputs:** `results/metrics/phase05_eval_report.json`, prediction exports in `results/predictions/`, summary plot in `results/visualizations/phase05/`

---

## 1) Abstract

Phase 05 performs the **final, reproducible evaluation** of trained models and exports **prediction artifacts** required for downstream analysis/reporting. It evaluates:

1. **Phase 04 tuned tabular models** (Logistic Regression + Random Forest) using their **saved thresholds** and (for LR) their **saved normalization statistics**.
2. **Phase 03 GNN checkpoints** (GCN, SkipGCN) by rebuilding the graph **exactly as in Phase 03** and loading the best hyperparameters from `results/metrics/gnn_*.json` to avoid checkpoint shape mismatches.

Phase 05 writes:
- a unified JSON report (`phase05_eval_report.json`)
- per-model prediction tables (Parquet/CSV)
- a final **summary bar plot** comparing **test illicit-F1** across models.

---

## 2) Goals and Scope

### 2.1 Goals
- Produce **final test-set metrics** (precision/recall/F1 for illicit + micro-F1 + confusion matrix) for:
  - tuned tabular models from Phase 04 (AF + LF)
  - trained GNN checkpoints from Phase 03 (AF)
- Export prediction files to enable:
  - threshold-based decision auditing
  - per-time-step or per-subset diagnostics
  - inference on **all nodes** (including unlabeled `class=-1`) for GNN models

### 2.2 Non-goals
- No model retraining or hyperparameter search.
- No new feature engineering.
- No new split logic (splits are reconstructed deterministically from Phase 03/04 rules).

---

## 3) Inputs, Prerequisites, and Artifacts

### 3.1 Required data inputs
- `data/processed/elliptic_full.(parquet|csv)`  
  Full dataset, includes `class ∈ {0,1,-1}`.
- `data/processed/elliptic_labeled.(parquet|csv)`  
  Labeled-only dataset, includes `class ∈ {0,1}`.
- `data/raw/elliptic_txs_edgelist.csv`  
  Transaction graph edges.

### 3.2 Required model artifacts
**Tabular (Phase 04 artifacts):**
- `results/model_artifacts/phase04/lr_{AF,LF}_{backend}.joblib`
- `results/model_artifacts/phase04/lr_scaler_{AF,LF}.joblib` *(stored as mu/sd stats)*
- `results/model_artifacts/phase04/lr_threshold_{AF,LF}_{backend}.json`
- `results/model_artifacts/phase04/rf_{AF,LF}_{backend}.joblib`
- `results/model_artifacts/phase04/rf_threshold_{AF,LF}_{backend}.json`

**GNN (Phase 03 artifacts):**
- `results/model_artifacts/gcn_AF.pt`
- `results/model_artifacts/skip_gcn_AF.pt`
- `results/metrics/gnn_AF.json` *(contains best_hp used to instantiate models before loading checkpoints)*

### 3.3 Run logging / reproducibility
- `src/runlog.py` writes a run manifest to `results/logs/phase05_*.json` capturing:
  - git commit hash
  - command, platform, python
  - config snapshot (`ExperimentConfig`)
  - required file existence + sizes + mtimes
  - environment variables (`AML_DEVICE`, etc.)

---

## 4) Evaluation Protocol

### 4.1 Temporal split (leakage-safe)
Phase 05 uses the same temporal split strategy as prior phases:

- **Tabular evaluation (Phase 04 parity):**
  - Train steps: first `floor(train_ratio * T)` timesteps
  - Test steps: remaining timesteps
  - With `train_ratio=0.70` and Elliptic timesteps 1..49:
    - Train: 1..34
    - Test: 35..49

- **GNN evaluation (Phase 03 parity):**
  - Uses `df_labeled` only to derive `train_steps/test_steps` (temporal).
  - Rebuilds full graph using `df_full` (includes unlabeled nodes) and creates:
    - `train_mask`: labeled nodes in train-only steps
    - `val_mask`: labeled nodes in last `val_ratio_within_train` portion of train steps
    - `test_mask`: labeled nodes in test steps
  - With `train_ratio=0.70`, `val_ratio_within_train=0.10`:
    - Train-only steps: 1..31
    - Val steps: 32..34
    - Test steps: 35..49

### 4.2 Metrics
Computed via `src/eval.py::compute_metrics`:
- `precision_illicit`, `recall_illicit`, `f1_illicit` (positive label = illicit = 1)
- `f1_micro`
- `cm` = confusion matrix with labels `[0,1]` (rows=true, cols=pred)

### 4.3 Thresholding (tabular)
Tabular models are evaluated using:
- **probability of illicit** (`p_illicit = predict_proba[:,1]`)
- **saved operating threshold** from Phase 04:
  - `y_pred = (p_illicit >= threshold).astype(int)`
This ensures Phase 05 reproduces the selected operating point rather than defaulting to `0.5`.

---

## 5) Implementation Summary

### 5.1 Main script responsibilities (`src/phase05_eval_infer.py`)
Phase 05 implements:

1. **Load configuration and enforce determinism**
   - `ExperimentConfig()` + `set_seed(cfg.seed, deterministic_torch=False)`
   - `AML_DEVICE` and/or CLI decide runtime device; `_pick_device` maps to `torch.device`.

2. **Load processed datasets**
   - `_load_processed_full(paths)` for full graph + inference alignment
   - `_load_processed_labeled(paths)` for labeled-only split derivation and tabular test set

3. **Evaluate Phase 04 tuned tabular artifacts**
   - Detect backend name from `results/metrics/phase04_tuning_index_*.json`
   - For each feature mode (AF, LF):
     - Load LR + scaler stats (mu/sd) + threshold, evaluate
     - Load RF + threshold, evaluate
   - Export predictions to `results/predictions/phase05_tabular_*.parquet`

4. **Evaluate Phase 03 GNN checkpoints**
   - Load best hyperparameters from `results/metrics/gnn_AF.json`
     - Required to instantiate correct hidden sizes (prevents state_dict shape mismatch)
   - Rebuild the graph via `src/graph_data.py::build_graph` with identical settings to Phase 03:
     - normalize using **train labeled nodes only**
     - optionally make undirected edges
   - Load checkpoints (`.pt`) and compute test metrics using `test_mask`.

5. **Optional inference export (GNN all nodes)**
   - Compute `p_illicit` for **all graph nodes** (including unlabeled `class=-1`)
   - Export aligned predictions (sorted by `txId` to match graph node ordering)

6. **Visualization (summary plot)**
   - Generates a single summary bar plot (Phase 05) comparing **test illicit-F1** for:
     - Tabular: LR/RF (AF/LF) using tuned thresholds
     - GNN: GCN/SkipGCN (AF)
   - Writes to `results/visualizations/phase05/phase05_final_f1_summary.png`

7. **Write unified evaluation report**
   - `results/metrics/phase05_eval_report.json`

---

## 6) Outputs

### 6.1 Primary report
- `results/metrics/phase05_eval_report.json`

Contains:
- config snapshot, device
- tabular evaluation results (if Phase 04 artifacts exist)
- GNN evaluation results
- paths to exported prediction tables
- path to final summary plot

### 6.2 Prediction exports
- Tabular:
  - `results/predictions/phase05_tabular_lr_AF_{backend}.parquet`
  - `results/predictions/phase05_tabular_rf_AF_{backend}.parquet`
  - `results/predictions/phase05_tabular_lr_LF_{backend}.parquet`
  - `results/predictions/phase05_tabular_rf_LF_{backend}.parquet`

Each includes: `txId, time_step, class, p_illicit, y_pred`

- GNN:
  - `results/predictions/phase05_gnn_gcn_AF.parquet`
  - `results/predictions/phase05_gnn_skip_gcn_AF.parquet`

Each includes: `txId, time_step, class, p_illicit, y_pred` for **all nodes** (including `class=-1`).

### 6.3 Visualizations
- `results/visualizations/phase05/phase05_final_f1_summary.png`

---

## 7) Results (from `phase05_eval_report.json`)

### 7.1 Tabular tuned models (Phase 04 artifacts, backend=`cuml`)

**Split:** Train steps 1..34, Test steps 35..49

| Model | Features | Threshold | Precision (illicit) | Recall (illicit) | F1 (illicit) | Micro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| LR | AF | 0.9471 | 0.3709 | 0.5282 | 0.4358 | 0.9112 |
| RF | AF | 0.4133 | 0.7968 | 0.7276 | **0.7606** | **0.9702** |
| LR | LF | 0.7973 | 0.3902 | 0.7285 | 0.5082 | 0.9084 |
| RF | LF | 0.4017 | 0.7649 | 0.7211 | 0.7424 | 0.9675 |

Confusion matrices are stored in the JSON report under each model entry.

### 7.2 GNN checkpoints (Phase 03 artifacts, evaluated in Phase 05)

**Graph:** 203,769 nodes; 468,710 edges  
**Split:** Train-only 1..31, Val 32..34, Test 35..49

| Model | Features | Precision (illicit) | Recall (illicit) | F1 (illicit) | Micro-F1 |
|---|---:|---:|---:|---:|---:|
| GCN | AF | 0.5069 | 0.5115 | 0.5092 | 0.9359 |
| SkipGCN | AF | 0.4729 | 0.5789 | 0.5205 | 0.9307 |

### 7.3 Best-performing model (by illicit-F1 on test)
- **Random Forest (AF, tuned threshold)** achieves the highest illicit-F1 (**0.7606**) and micro-F1 (**0.9702**) among evaluated models in this run.

---

## 8) Interpretation and Discussion

### 8.1 Thresholding impact (tabular)
Phase 05 evaluates tabular models using **tuned thresholds** from Phase 04 rather than a fixed `0.5`. This is essential under class imbalance: the chosen operating point trades off precision vs recall according to validation-derived objectives (illicit-F1).

### 8.2 GNN vs tabular in this pipeline
- GNN test illicit-F1 (~0.51–0.52) is substantially below tuned RF (~0.74–0.76) in this run.
- This result is consistent with:
  - the difficulty of learning from sparse neighborhood structure
  - potential sensitivity to temporal split, masking, and normalization
  - limited GNN tuning budget (`gnn_tune_trials=10`)

### 8.3 Reproducibility guarantees
- Temporal splitting is deterministic.
- Graph node ordering is deterministic (`txId` sort).
- Normalization for GNN uses **train labeled nodes only** (explicit anti-leakage).
- Full run provenance is recorded in `results/logs/phase05_*.json`.

---

## 9) How to Run Phase 05

### 9.1 Minimal run
From repository root:
```bash
python -m src.main --phase 5
```

### 9.2 Force GPU

Either:
```bash
python -m src.main --phase 5 --device cuda
```
or:
```bash
AML_DEVICE=cuda python -m src.main --phase 5
```

### 9.3 Preconditions checklist
Before running Phase 05, ensure:
- Phase 01 has produced `data/processed/elliptic_full.*` and `elliptic_labeled.*`
- Phase 03 has produced:
- - `results/model_artifacts/gcn_AF.pt`, `skip_gcn_AF.pt`
- - `results/metrics/gnn_AF.json`
- Phase 04 has produced tuned artifacts under:
- - `results/model_artifacts/phase04/`
- - `results/metrics/phase04_tuning_index_*.json`
If Phase 04 artifacts are missing, Phase 05 will still run GNN evaluation and export those results, but tabular tuned evaluation may be absent or partial.

---

## 10)Known Limitations and Next Improvements

More Phase 05 diagnostics plots
1. Current Phase 05 visualization is a summary illicit-F1 bar plot. Additional recommended plots (optional):
- - confusion matrices (raw + normalized) per model
- - illicit precision/recall/F1 over time_step (temporal stability)
- - distribution of `p_illicit` for labeled licit vs illicit (separation)
2. Calibration analysis
If probabilities are used operationally, add calibration curves and Brier score.
3. Consistent operating-point selection policy
Document whether thresholds optimize illicit-F1, recall constraints, or cost-weighted utility.

---

## 11) File-Level Summary (Phase 05)

- `src/phase05_eval_infer.py`
Orchestrates final evaluation for tabular (Phase 04 artifacts) + GNN (Phase 03 checkpoints), exports predictions, writes unified JSON report, and triggers a final summary plot.
- `src/viz.py`
Provides plotting utilities. Phase 05 adds/uses a final summary bar plot for illicit-F1 comparison.
- `results/metrics/phase05_eval_report.json`
Single source of truth for Phase 05 results + artifact paths.
- `results/predictions/*`
Prediction tables for each evaluated model (and GNN inference on all nodes).
- `results/visualizations/phase05/*`
Phase 05 plots (currently: final summary bar plot).