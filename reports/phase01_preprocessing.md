# Phase 01 — Preprocessing (Elliptic Bitcoin Transaction Graph)

**Implementation:** `src/phase01_preprocessing.py`  
**Schema:** `phase01_v3`  
**Primary artifacts:** `data/processed/*` and `results/metrics/phase01_data_summary.json`  
**Run manifest:** `results/logs/phase01_*.json` (written by `src/runlog.py::write_run_manifest`)

---

## 1. Abstract

Phase 01 implements a deterministic preprocessing and validation pipeline for the Elliptic transaction graph dataset. It ingests the three raw CSV inputs (features, edge list, and classes), constructs a canonical “full” node table aligned by `txId`, derives the labeled-only subset, produces a node index mapping to enforce stable node ordering across phases, and exports both raw and node-filtered edge lists. The phase also computes integrity metadata (SHA256, sizes, timestamps), enforces dataset contract checks (schema, time-step range, numeric finiteness), and writes a structured metrics summary to support reproducibility and downstream debugging.

---

## 2. Goals and Rationale

### 2.1 Goals
1. **Create a canonical node table** with one row per `txId` that merges features and class labels.
2. **Standardize label encoding** for downstream ML (binary classification with unknown/unlabeled handling).
3. **Ensure deterministic ordering** to prevent subtle nondeterminism in graph construction and modeling.
4. **Export stable, portable artifacts** (Parquet preferred, CSV fallback) used by later phases.
5. **Validate data integrity** early (schema, time, numeric values, edge sanity).
6. **Record provenance** (file hashes, sizes, timestamps, run metadata) for scientific reproducibility.

### 2.2 Why these steps are necessary
- The Elliptic dataset is used across multiple phases (EDA, tabular baselines, GNN training). Any inconsistency in node ordering, label mapping, or graph edges produces silent downstream errors (misaligned labels, incorrect edges, data leakage).
- Deterministic exports enable byte-identical artifacts across runs given identical raw inputs.
- Early contract checks prevent expensive failures later during model training.

---

## 3. Inputs

### 3.1 Raw files (expected location: `data/raw/`)
1. **`elliptic_txs_features.csv`**  
   - Headerless (read with `header=None` in `src/data.py`)  
   - Expected shape: `(203,769 rows, 167 columns)`  
   - Semantics: `[txId, time_step, 165 numeric features]`

2. **`elliptic_txs_edgelist.csv`**  
   - Two columns: `txId1, txId2`  
   - Directed transaction graph edges (as provided by dataset)

3. **`elliptic_txs_classes.csv`**  
   - Two columns: `txId, class`  
   - Class values: `"1"`, `"2"`, `"unknown"`

### 3.2 Label mapping (via `src/data.py`)
- Raw dataset convention:
  - `"1"` = illicit
  - `"2"` = licit
  - `"unknown"` = unlabeled
- Internal encoding used throughout the project:
  - `1` = illicit (positive class)
  - `0` = licit (negative class)
  - `-1` = unknown/unlabeled

This mapping is applied by `build_full_dataset()` (which calls `map_class_to_int()` inside `src/data.py`).

---

## 4. Methods (Implementation Details)

### 4.1 Directory setup
`_ensure_dirs(Paths())` ensures the pipeline’s output directories exist:
- `data/processed/`
- `results/`
- `results/metrics/`
- `results/logs/`

### 4.2 Run provenance: manifest logging
At the beginning of each run, the script calls:
- `write_run_manifest(...)`

This writes a JSON log under `results/logs/` that captures:
- run id, UTC datetime, git commit, command, cwd, python/platform
- raw file existence/size/mtime
- selected environment values (e.g., `AML_STRICT_ELLIPTIC`)
- schema version (`phase01_v3`) in `extra`

This is the authoritative provenance record for Phase 01 runs.

### 4.3 Strict dataset validation (optional)
Environment-controlled strict mode:
- `AML_STRICT_ELLIPTIC` (default: enabled via `_truthy_env("AML_STRICT_ELLIPTIC","1")`)

If strict mode is enabled:
- enforce `loaded.features.shape[1] == 167`

Rationale:
- Ensures we are processing the canonical Elliptic feature matrix (2 identifier columns + 165 features).
- Prevents accidental ingestion of malformed/corrupted files (wrong delimiter, headers inserted, truncation).

### 4.4 Full dataset construction (node table)
Steps:
1. Load raw inputs: `loaded = load_elliptic(paths.raw_dir)`
2. Build full dataset: `df_full = build_full_dataset(loaded.features, loaded.classes)`
3. Enforce stable dtypes: `_coerce_dtypes_full(df_full)`

`_coerce_dtypes_full` enforces:
- `txId`: `int64`
- `time_step`: `int64`
- `class`: `int64` in `{-1, 0, 1}`
- `feat_*`: `float32`

Rationale:
- Improves performance and memory footprint (float32 features).
- Eliminates dtype drift between runs/environments (reproducibility).
- Avoids downstream issues in PyTorch/PyG where dtype mismatches can cause implicit casting or errors.

### 4.5 Canonical node ordering + node index artifact
The canonical node ordering is the sorted `txId` sequence from `df_full`:
- `tx_ids = df_full["txId"].to_numpy(...)`

Validation:
- `_assert_node_index(tx_ids)` ensures:
  - 1D array, non-empty
  - strictly increasing order (deterministic)
  - unique txIds

Export:
- `_save_node_index(paths, tx_ids)` produces:
  - `data/processed/node_index.parquet` (or CSV fallback)
  - columns: `txId`, `node_idx` (`0..N-1`)

Rationale:
- Graph frameworks require a stable mapping from `txId → node_idx`.
- This artifact prevents accidental reordering differences across scripts/phases.
- Downstream graph building can rely on this canonical ordering to ensure node feature matrices and label arrays are aligned.

### 4.6 Data contract validation
`_assert_data_contract(df_full, feat_cols)` enforces:
- required columns exist: `txId`, `time_step`, `class`
- exactly one row per `txId`
- canonical time steps: `1..49` with 49 unique steps
- presence of feature columns `feat_*`
- numeric finiteness: no NaN/inf in feature matrix

Rationale:
- Prevents training-time failures.
- Guards against silent corruption (NaNs introduced by parsing or upstream edits).
- Enforces the temporal structure needed for later temporal splits.

### 4.7 Labeled-only dataset
Derived as:
- `df_labeled = df_full[df_full["class"] != -1]`

Rationale:
- Supervised training uses labeled nodes only.
- Unlabeled nodes remain present in the full graph for message passing (GNN), but must be excluded from supervised loss/metrics.

### 4.8 Edge processing and determinization
The script exports:
1. **Raw edges (as loaded)**  
   - `data/processed/elliptic_edges.parquet`

2. **Kept edges (node-filtered + determinized)**  
   `edges_kept = _filter_edges_to_nodes(loaded.edges, df_full["txId"])`:
   - removes edges whose endpoints are not present as nodes
   - selects only `["txId1","txId2"]`
   - drops duplicates
   - sorts by `(txId1, txId2)` for stable exports

Rationale:
- Later phases should typically use `elliptic_edges_kept` to avoid referencing missing nodes.
- Sorting and duplicate removal ensure deterministic serialization and stable diagnostics.
- In the canonical Elliptic dataset, kept edges should match raw edges (no drops). The kept artifact still protects against malformed inputs.

### 4.9 Graph diagnostics (quality checks)
The phase computes multiple edge diagnostics restricted to the node set:

**Undirected degree diagnostics (induced subgraph)**
- `_edge_stats(edges, node_ids)` treats edges as undirected only for degree magnitude reporting.

**Directed diagnostics**
- `_edge_stats_directed(edges, node_ids)` reports:
  - in-degree/out-degree distributions (min/max/mean/median)
  - count of nodes with zero in-degree and/or zero out-degree

**Coverage**
- `_edge_coverage(edges, node_ids)` reports:
  - number of unique nodes appearing in edges
  - number missing from edges
  - coverage ratio

Rationale:
- Quickly detects disconnected/isolated nodes, unexpected self-loops, or duplicate edges.
- Directed stats distinguish “no incoming edges” vs “no outgoing edges”, which matters for temporal/directed interpretations.

### 4.10 Raw input integrity metadata
`_file_meta()` records for each raw file:
- existence, bytes, mtime (UTC), SHA256 hash

Rationale:
- Provides a cryptographic fingerprint for data provenance.
- Enables reproducible reporting and supports verification that multiple experiments used identical inputs.

---

## 5. Outputs (Artifacts)

### 5.1 Processed datasets (`data/processed/`)
Primary outputs (Parquet preferred; CSV written only if Parquet fails):
- `elliptic_full.parquet`  
  Full node table: `txId`, `time_step`, `feat_*`, `class_raw`, `class`
- `elliptic_labeled.parquet`  
  Labeled node subset only (`class` in `{0,1}`)
- `elliptic_edges.parquet`  
  Raw edges as loaded
- `elliptic_edges_kept.parquet`  
  Node-filtered, deduplicated, sorted edge list
- `node_index.parquet`  
  Canonical mapping: `txId → node_idx`

### 5.2 Metrics summary (`results/metrics/`)
- `phase01_data_summary.json`  
  Contains:
  - schema version
  - saved artifact locations and formats
  - raw input metadata (hashes)
  - dataset sizes, label distributions
  - contract check indicators
  - edge diagnostics and coverage

### 5.3 Run logs (`results/logs/`)
- `phase01_<timestamp>.json` (produced by `write_run_manifest`)  
  Contains run configuration and environment/provenance.

---

## 6. Observed Results (from latest Phase 01 run)

### 6.1 Dataset sizes
- Raw features: `203,769` rows × `167` cols  
- Raw classes: `203,769` rows  
- Full dataset: `203,769` rows  
- Labeled subset: `46,564` rows  
- Unknown/unlabeled: `157,205` rows  
- Feature columns: `165` (`feat_0 .. feat_164`)  
- Time steps: `min=1`, `max=49`, `unique=49`

### 6.2 Label distribution
**Raw labels (`class_raw`):**
- `unknown`: `157,205`
- `2`: `42,019` (licit)
- `1`: `4,545` (illicit)

**Mapped labels (`class`):**
- `-1` (unknown): `157,205`
- `0` (licit): `42,019`
- `1` (illicit): `4,545`

### 6.3 Edge statistics
- Raw edges rows: `234,355`
- Kept edges rows: `234,355`
- Dropped edges: `0`

**Undirected degree diagnostics (induced):**
- self-loops: `0`
- duplicate directed edges: `0`
- degree min: `1`
- degree max: `473`
- degree mean: `2.3002026804862368`
- degree median: `2`
- isolated nodes: `0`

**Directed diagnostics (induced):**
- out-degree max: `472`, mean: `1.4088490787219334`, median: `1`
- in-degree max: `284`, mean: `1.578711594036929`, median: `1`
- zero out-degree nodes: `37,424`
- zero in-degree nodes: `55,322`

**Coverage:**
- unique nodes in induced edges: `203,769`
- missing nodes from edges: `0`
- coverage ratio: `1.0`

Interpretation:
- Every node participates in at least one edge (coverage 1.0, isolated 0).
- Many nodes have zero in-degree or zero out-degree, which is expected in a directed transaction graph even if the undirected projection shows connectivity.

---

## 7. Reproducibility and Determinism Guarantees

Phase 01 is designed to be repeatable and deterministic given identical raw inputs:
- Node ordering is canonicalized and exported (`node_index`).
- Edge exports are determinized (`drop_duplicates` + sorted `(txId1, txId2)`).
- Feature dtypes are forced to stable types (`float32`), avoiding platform-dependent inference.
- Raw input hashes (SHA256) are stored in the summary JSON to verify identical inputs across experiments.
- Run manifests store git commit and runtime context for auditability.

---

## 8. Configuration and Execution

### 8.1 How to run
From project root:
```bash
python -m src.main --phase 1
```

### 8.2 Environment controls

- `AML_STRICT_ELLIPTIC` (default `1`)

- - `1`: enforce canonical raw feature column count of 167

- - `0`: bypass this check (only recommended for nonstandard datasets)

---

## 9. Scope Boundaries (What Phase 01 does NOT do)

- No feature normalization/standardization for modeling.

- - Normalization is performed later in a split-aware manner to avoid leakage.

- No temporal splitting into train/val/test.

- - Splits are performed in later phases using `time_step`.

- No graph object construction (PyG `Data`) or adjacency operations.

- - Graph building is handled later (e.g., `src/graph_data.py`) using the exported node index and edge list.

---

## 10. Transition to Phase 02 (EDA)

Phase 01 outputs are the single source of truth for all downstream phases:

- Use `elliptic_full.parquet` for graph-based work (includes unlabeled nodes).
- Use `elliptic_labeled.parquet` for supervised tabular baselines.
- Prefer `elliptic_edges_kept.parquet` for any graph construction.
- Use `node_index.parquet` to guarantee consistent node ordering across all computations.
Phase 02 should treat Phase 01 artifacts as immutable unless raw inputs change.

```makefile
::contentReference[oaicite:0]{index=0}
```
