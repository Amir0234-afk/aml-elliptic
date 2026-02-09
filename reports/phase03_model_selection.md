# Phase 03 — Model Selection (Scientific Report)

**Project:** AML Elliptic (Bitcoin transaction classification)  
**Phase:** 03 — Model Selection (Tabular Baselines + Graph Neural Networks)  
**Run ID:** `phase03_20260208T230132Z`  
**Git commit:** `01ca07a9807f1ec48e5f6935a5c30de2b45d13c7`  
**Command:** `python -m src.main --phase 3 --device cuda --tabular-backend cuml`  
**UTC timestamp:** 2026-02-08T23:01:32Z  
**Execution:** WSL2 Linux 6.6 (x86_64), Python 3.10.19 (conda-forge)  
**Hardware:** NVIDIA GeForce RTX 3060 Laptop GPU  
**Software:** torch 2.6.0+cu124 (CUDA available), cuDNN enabled, cuDNN deterministic

---

## Abstract

This phase establishes leakage-safe, reproducible model-selection baselines for illicit transaction detection on the Elliptic Bitcoin dataset. We implement (i) tabular classifiers trained on engineered features with temporal splits and forward-chaining cross-validation, and (ii) graph neural networks trained on the transaction graph using the full node set (including unlabeled nodes as structural context) while supervising only on labeled nodes. We report illicit-class precision/recall/F1 and confusion matrices under a strict time-based evaluation protocol. Results show that a tuned Random Forest on all features achieves the best illicit-class F1 (0.8027), while GNNs (GCN/SkipGCN) reach moderate test illicit-class F1 (~0.51–0.52). An embedding-augmentation experiment (RF on feature ⊕ GNN embedding) attains very high precision (0.9929) but does not exceed the best tabular baseline in illicit F1.

---

## 1. Research Objective

### 1.1 Problem statement
Given Bitcoin transactions represented by:
- a **feature vector** per transaction (time-dependent engineered statistics), and
- a **directed transaction graph** (edges between transactions),

predict whether a transaction is **illicit** (positive class) or **licit** (negative class), with a substantial portion of transactions **unlabeled**.

### 1.2 Phase 03 goals
1. Produce **strong baselines** under a realistic **time-based generalization** setting.
2. Ensure **reproducibility** and **auditability** (deterministic seeds, logged artifacts, hashes).
3. Compare:
   - tabular learning (feature-only) vs.
   - graph learning (structure + features),
   under the same temporal protocol.
4. Test whether GNN representations help a strong tabular model (embedding augmentation).

---

## 2. Data and Label Semantics

### 2.1 Data sources
Inputs are loaded from:
- `elliptic_txs_features.csv` (node features + time_step)
- `elliptic_txs_edgelist.csv` (graph edges)
- `elliptic_txs_classes.csv` (labels: 1, 2, unknown)

Processed datasets produced by Phase 01:
- `data/processed/elliptic_full.csv` (all nodes; includes unlabeled)
- `data/processed/elliptic_labeled.csv` (labeled-only subset)

### 2.2 Label mapping
Raw labels are mapped to:
- illicit → `1`
- licit → `0`
- unlabeled/unknown → `-1`

This mapping is implemented in `src/data.py` (`map_class_to_int`, `build_full_dataset`).

---

## 3. Experimental Design

### 3.1 Temporal evaluation protocol (primary design choice)
All models are evaluated using a **chronological split** by `time_step`:

- **Train steps:** 1–34  
- **Test steps:** 35–49

**Rationale:** Random splitting would leak future patterns into training, inflating performance estimates and reducing external validity. Temporal splitting approximates deployment: training on past, predicting on future.

### 3.2 Target metric: illicit-class performance
Because illicit detection is the operationally important class, we optimize and report:
- **Precision(illicit)** = TP / (TP + FP)  
- **Recall(illicit)** = TP / (TP + FN)  
- **F1(illicit)** = 2PR / (P + R)

We additionally report micro-F1 and confusion matrices for completeness.

### 3.3 Class imbalance handling
Class weights are computed on the **training labeled nodes only**:

\[
w_c = \frac{N}{2N_c}
\]

This discourages the model from defaulting to majority-class prediction.

---

## 4. Methods

Phase 03 consists of three experimental blocks:

1. **Tabular baselines** (LR, RF) under temporal split + forward-chaining CV.
2. **Graph baselines** (GCN, SkipGCN) under full-graph construction + leakage-safe masks.
3. **Embedding augmentation**: RF trained on concatenated raw features and learned GNN node embeddings.

### 4.1 Tabular baselines (feature-only)

#### 4.1.1 Feature modes
Two feature sets are evaluated (Elliptic conventions):
- **AF (All Features):** all columns `feat_*`
- **LF (Local Features):** first 94 `feat_*` columns

Selected by `src/data.py:get_feature_cols`.

#### 4.1.2 Leakage-safe hyperparameter tuning (forward-chaining CV)
Within training steps (1–34), we use **forward-chaining** splits:
- training fold uses early steps,
- validation fold uses the next step(s) immediately after the training window.

This prevents temporally “future” examples from influencing model selection.

Implemented in `src/phase03_models.py:_forward_chaining_cv_splits`.

#### 4.1.3 Logistic Regression (LR)
- Standardization: `StandardScaler` fit on training fold only, applied to validation/test.
- Tuning: random subset of `C ∈ logspace(1e-3, 1e2)`.
- Backend: `cuml` (GPU) when available; otherwise falls back to sklearn.
- Training/evaluation uses the same temporal test split.

#### 4.1.4 Random Forest (RF)
- Input: raw features (no scaling).
- Tuning: randomized trials over:
  - `n_estimators ∈ {200,400,600,800,1000}`
  - `max_features ∈ {"sqrt", 0.3, 0.5, ...}`
  - `max_depth ∈ {10,16,20,30,40}` (cuML stability)
- Backend: cuML RF on GPU (with recorded capability caveats).

### 4.2 Graph Neural Networks (GNN)

#### 4.2.1 Full-graph construction with unlabeled nodes as context
We build a single PyTorch Geometric `Data` object:
- **Nodes:** all transactions from `elliptic_full` (includes unlabeled nodes)
- **Edges:** from `elliptic_txs_edgelist.csv`, filtered to nodes present
- Optionally **undirected:** edge list mirrored (enabled)

Implemented in `src/graph_data.py:build_graph`.

**Rationale:** GNN message passing can exploit unlabeled neighbors as structural context, but supervision must remain confined to labeled nodes.

#### 4.2.2 Supervision masks and leakage controls
- `train_mask`: labeled nodes in **train-only steps** (train steps excluding the last 10% reserved for validation)
- `val_mask`: labeled nodes in the **held-out tail of training steps**
- `test_mask`: labeled nodes in **test steps**
- unlabeled nodes (`y=-1`) are excluded from all masks.

#### 4.2.3 Feature normalization without label leakage
We normalize features using mean/std computed **only on train-labeled nodes**, then apply to all nodes:

\[
x' = \frac{x - \mu_{train}}{\sigma_{train} + \epsilon}
\]

Implemented in `src/graph_data.py` (`normalize=True` with `idx=train_mask`).

**Rationale:** Computing normalization statistics using test or future nodes would leak information.

#### 4.2.4 Architectures
- **GCN:** standard 2-layer graph convolution
- **SkipGCN:** GCN with skip connection from input features to output

Models are defined in `src/gnn.py`, trained with early stopping (`epochs=200`, `patience=20`).

#### 4.2.5 GNN hyperparameter tuning
Random search (`gnn_tune_trials=10`) over:
- hidden dimension, dropout, learning rate, weight decay
- illicit weight scaling factor ∈ {0.5,1,2,4} (applied only to illicit class)

Selection criterion: best validation F1(illicit).

### 4.3 Embedding augmentation experiment
We test whether learned GNN representations improve tabular classification:
1. Train best GCN/SkipGCN on graph supervision.
2. Extract node embeddings `h = encode(x, edge_index)` for all nodes.
3. Concatenate features: \( X_{aug} = [x \oplus h] \)
4. Train **sklearn RandomForestClassifier** (CPU) on labeled train steps only.
5. Evaluate on labeled test steps.

Implemented in `src/phase03_models.py` (`enable_embedding_aug=True`).

**Rationale:** If the GNN learns meaningful structural representations, embeddings can act as additional features for a high-capacity classifier.

---

## 5. Results

### 5.1 Tabular models (AF, All Features)

**Train steps:** 1–34  
**Test steps:** 35–49  
**Class weights:** `w0=0.5655`, `w1=4.3174`  
**Tuning:** enabled, forward-chaining CV (5 splits)

#### 5.1.1 Logistic Regression (AF)
- Precision(illicit): **0.2192**
- Recall(illicit): **0.8236**
- F1(illicit): **0.3463**
- Micro-F1: **0.7980**
- Confusion matrix:
  - `[[12410, 3177], [191, 892]]`
- Best `C`: **100.0**

**Interpretation:** LR achieves high recall but produces many false positives, yielding poor precision and modest illicit F1.

#### 5.1.2 Random Forest (AF)
- Precision(illicit): **0.8992**
- Recall(illicit): **0.7248**
- F1(illicit): **0.8027**
- Micro-F1: **0.9768**
- Confusion matrix:
  - `[[15499, 88], [298, 785]]`
- Best params:
  - `n_estimators=1000`
  - `max_features=0.3`
  - `max_depth=20`
  - `min_samples_leaf=1`

**Interpretation:** RF(AF) is the strongest overall model in illicit F1, combining high precision and strong recall.

---

### 5.2 Tabular models (LF, Local Features)

#### 5.2.1 Logistic Regression (LF)
- Precision(illicit): **0.1912**
- Recall(illicit): **0.8329**
- F1(illicit): **0.3110**
- Micro-F1: **0.7602**
- Confusion matrix:
  - `[[11771, 3816], [181, 902]]`

#### 5.2.2 Random Forest (LF)
- Precision(illicit): **0.8879**
- Recall(illicit): **0.7018**
- F1(illicit): **0.7839**
- Micro-F1: **0.9749**
- Confusion matrix:
  - `[[15491, 96], [323, 760]]`

**Interpretation:** AF consistently outperforms LF; global/aggregate features are beneficial for illicit detection.

---

### 5.3 GNN models (AF)

**Train steps:** 1–31  
**Val steps:** 32–34  
**Test steps:** 35–49  
**Graph:** 203,769 nodes; 468,710 edges (undirected expansion)  
**Labeled nodes:** train=27,615, val=2,279, test=16,670  
**Unlabeled nodes:** 157,205  
**Class weights (train labeled):** `w0=0.5623`, `w1=4.5123`

#### 5.3.1 GCN (AF)
- Best validation F1(illicit): **0.8134**
- Best epoch: **198**
- Best HP: hidden=200, dropout=0.0, lr=0.002, wd=1e-6, illicit_wscale=0.5
- Test:
  - Precision(illicit): **0.5069**
  - Recall(illicit): **0.5115**
  - F1(illicit): **0.5092**
  - Micro-F1: **0.9359**
  - CM: `[[15048, 539], [529, 554]]`

#### 5.3.2 SkipGCN (AF)
- Best validation F1(illicit): **0.8089**
- Best epoch: **199**
- Best HP: hidden=100, dropout=0.0, lr=0.002, wd=1e-4, illicit_wscale=0.5
- Test:
  - Precision(illicit): **0.4729**
  - Recall(illicit): **0.5789**
  - F1(illicit): **0.5205**
  - Micro-F1: **0.9307**
  - CM: `[[14888, 699], [456, 627]]`

**Interpretation:** Despite strong validation scores, both GNN variants generalize to test steps with only moderate illicit F1 (~0.51–0.52), substantially below RF(AF)=0.80.

---

### 5.4 Embedding augmentation (RF on feature ⊕ embedding)

**Embedding source:** GCN  
**RF estimators:** 500  
**Test:**
- Precision(illicit): **0.9929**
- Recall(illicit): **0.6445**
- F1(illicit): **0.7816**
- Micro-F1: **0.9766**
- CM: `[[15582, 5], [385, 698]]`

**Interpretation:** Embeddings yield extremely high precision and competitive micro-F1, but illicit F1 remains below the best RF(AF) baseline.

---

## 6. Discussion

### 6.1 Why RF(AF) dominates
RF(AF) likely benefits from:
- strong engineered features capturing transaction behavior patterns,
- robustness to heterogeneous feature scales and non-linear interactions,
- effective handling of imbalance via class weights.

### 6.2 Why GNNs underperform on test
Observed gap (val F1 high, test F1 moderate) suggests:
- **temporal distribution shift** between steps 1–34 and 35–49,
- limited architecture capacity (2-layer GCN family),
- supervision sparsity (most nodes unlabeled),
- graph homophily assumptions not holding strongly for illicit behavior,
- potential mismatch between the validation window (steps 32–34) and the true test regime.

### 6.3 Embedding augmentation outcome
The augmentation experiment indicates embeddings add useful discriminative information (precision near 0.993), but the base RF(AF) already captures most of the usable signal for illicit F1 under this protocol.

---

## 7. Conclusion (Model Selection)

Under a leakage-safe temporal evaluation protocol:

- **Best overall illicit F1:** **Random Forest (AF)** with **F1=0.8027**  
- **High-recall baseline:** Logistic Regression (AF/LF) with **Recall≈0.82–0.83**, low precision  
- **Graph baselines (AF):** GCN/SkipGCN achieve **F1≈0.51–0.52** on test  
- **Embedding augmentation:** strong precision (0.993) but **F1=0.7816**, not exceeding RF(AF)

Therefore, Phase 03 selects **RF(AF)** as the primary baseline model for downstream phases.

---

## 8. Reproducibility, Artifacts, and Freeze

### 8.1 Determinism and logging
- Seed: `42`
- Deterministic torch behavior enabled (noting CUDA determinism constraints)
- Full run manifest written to `results/logs/phase03_20260208T230132Z.json`

### 8.2 Persisted metrics
- `results/metrics/baselines_AF.json`
- `results/metrics/baselines_LF.json`
- `results/metrics/gnn_AF.json`

### 8.3 Persisted models
- `results/model_artifacts/rf_AF_cuml.joblib` (selected baseline)
- `results/model_artifacts/lr_AF_cuml.joblib`
- `results/model_artifacts/scaler_AF.joblib`
- `results/model_artifacts/rf_LF_cuml.joblib`
- `results/model_artifacts/lr_LF_cuml.joblib`
- `results/model_artifacts/scaler_LF.joblib`
- `results/model_artifacts/gcn_AF.pt`
- `results/model_artifacts/skip_gcn_AF.pt`
- `results/model_artifacts/rf_aug_AF_gcn.joblib`

### 8.4 Visualizations
- `results/visualizations/model_comparison_f1.png`

### 8.5 Artifact inventory (hashes)
- `results/logs/phase03_artifacts_*.json` records SHA256 for each artifact.

### 8.6 Freeze statement
Phase 03 is frozen: code paths, metrics, and serialized artifacts define the fixed baseline and graph reference points for later phases. Any further changes must be versioned as a new phase or an explicitly new experimental branch.

---
