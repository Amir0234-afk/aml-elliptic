# Phase 02 — Exploratory Data Analysis (EDA) & Visualization Export

## 1. Purpose and Scope

Phase 02 performs a **descriptive, non-modeling** analysis of the Elliptic Bitcoin transaction dataset after Phase 01 preprocessing. The goals of this phase are:

1. **Validate data integrity** of the processed artifacts (schema, label domain, time steps, graph coverage).
2. **Characterize the classification task** (class imbalance and temporal behavior of labels).
3. **Characterize graph structure** (degree distribution and structural differences across labeled/unlabeled and licit/illicit subsets).
4. **Surface feature-level signal** using simple, interpretable proxies.
5. Export **standardized visualizations and metrics** to support later modeling decisions (Phases 03–05).

Phase 02 intentionally avoids any model training, data splitting, or feature normalization. It is strictly exploratory.

---

## 2. Inputs and Artifacts

### 2.1 Processed Node Tables (Parquet-first)

Phase 02 consumes the processed datasets produced in Phase 01:

- `data/processed/elliptic_full.{parquet|csv}`  
  Contains all nodes (labeled and unlabeled), with  
  `class ∈ {-1, 0, 1}` where `-1=unknown`, `0=licit`, `1=illicit`.

- `data/processed/elliptic_labeled.{parquet|csv}`  
  Contains only labeled nodes, with  
  `class ∈ {0, 1}`.

**Parquet-first loading.**  
Because Phase 01 v3 writes Parquet when possible, Phase 02 always attempts to load Parquet first and falls back to CSV only if Parquet is missing. This prevents silent reuse of stale CSV artifacts from earlier runs.

### 2.2 Graph Edges (Deterministic, Processed First)

Edges are loaded in the following priority order:

1. `data/processed/elliptic_edges_kept.parquet`  
2. `data/processed/elliptic_edges_kept.csv`  
3. `data/processed/elliptic_edges.parquet`  
4. `data/processed/elliptic_edges.csv`  
5. Fallback: raw edges from `data/raw/elliptic_txs_edgelist.csv`

The “kept” edge set is deterministic and aligned with the canonical node set, ensuring consistency with later graph-based modeling.

---

## 3. Reproducibility and Run Logging

Each Phase 02 execution records a run manifest including:

- Run identifier, timestamp, commit hash, and execution command.
- Configuration parameters (seed, etc.).
- Exact input files consumed (raw and processed).
- Metadata describing which formats and edge sources were used.

This ensures Phase 02 outputs are fully auditable and reproducible.

---

## 4. Phase 02 Contract Checks (Data Integrity)

Before any analysis, Phase 02 enforces a strict contract:

- Required columns (`txId`, `time_step`, `class`) exist.
- Feature columns (`feat_*`) are present.
- One row per `txId` in both full and labeled tables.
- Valid class domains:
  - Full: `{-1, 0, 1}`
  - Labeled: `{0, 1}`
- Canonical temporal structure:
  - `time_step` spans exactly 1–49 with 49 unique values.
- Edge coverage:
  - All edge endpoints belong to the node set.

Violations cause immediate failure to prevent misleading analysis.

---

## 5. Analyses and Visualizations

All figures are written to:

```
results/visualizations/phase02_eda/
```


### 5.1 Label Distribution

- **File:** `class_distribution.png`  
- **Purpose:** Quantifies class imbalance between licit and illicit transactions.

---

### 5.2 Nodes per Time Step

- **Files:**  
  - `nodes_per_timestep_full.png`  
  - `nodes_per_timestep_labeled.png`  

- **Purpose:** Confirms temporal coverage and stability across time steps.

---

### 5.3 Label Counts Over Time

- **File:** `label_counts_over_time.png`  
- **Purpose:** Shows how licit and illicit labeled nodes are distributed across time.

---

### 5.4 Illicit Ratio Over Time

- **File:** `illicit_ratio_over_time.png`  
- **Purpose:** Tracks temporal variation in illicit prevalence among labeled nodes.

---

### 5.5 Feature-Level Signal (Mean Differences)

- **File:** `top_feature_mean_diff.png`  
- **Purpose:** Ranks features by absolute difference in mean between illicit and licit classes as a simple discriminative proxy.

---

### 5.6 Graph Degree Distribution

- **Files:**  
  - `degree_distribution.png`  
  - `degree_distribution_log.png`  

- **Purpose:** Characterizes graph sparsity and heavy-tailed connectivity.

---

### 5.7 Additional Structural Analyses

**Counts over time**
- `licit_count_over_time.png`
- `illicit_count_over_time.png`

**Labeled fraction over time**
- `labeled_fraction_over_time.png`

**Degree distributions by subgroup**
- `degree_labeled.png`, `degree_labeled_log.png`
- `degree_unlabeled.png`, `degree_unlabeled_log.png`
- `degree_licit.png`, `degree_licit_log.png`
- `degree_illicit.png`, `degree_illicit_log.png`

These plots assess selection bias and structural differences between node subsets.

---

## 6. Metrics Summary

Phase 02 exports a machine-readable summary to:

```
results/metrics/phase02_eda_summary.json
```


Key statistics from the current run include:

- Total nodes: 203,769  
- Labeled nodes: 46,564  
  - Licit: 42,019  
  - Illicit: 4,545  
- Unlabeled nodes: 157,205  
- Time steps: 49 (full and labeled)
- Edges: 234,355 (100% node coverage)
- Degree percentiles (undirected):
  - p50 = 2  
  - p90 = 3  
  - p99 = 13  
  - max = 473  

---

## 7. Scientific Takeaways

- **Severe class imbalance** motivates class-weighted learning and illicit-focused metrics.
- **Stable temporal coverage** supports forward temporal validation.
- **Sparse, heavy-tailed graph structure** justifies shallow and regularized GNN architectures.
- **Structural differences between labeled and unlabeled nodes** indicate potential selection bias.
- **Feature signal is distributed**, motivating models that capture interactions and graph context.

---

## 8. Explicit Non-Goals

Phase 02 deliberately excludes:

- Feature normalization or scaling.
- Train/test splits or temporal cross-validation.
- Any model training or predictive performance metrics.

These belong to later phases.

---

## 9. Outputs

- Visualizations: `results/visualizations/phase02_eda/*.png`
- Metrics summary: `results/metrics/phase02_eda_summary.json`
- Run manifest: `results/logs/phase02_<timestamp>.json`

---

## 10. Conclusion

Phase 02 validates dataset integrity, characterizes class imbalance, temporal dynamics, and graph structure, and produces reproducible artifacts for reporting. The findings directly motivate and justify the modeling strategies applied in subsequent phases (03–05).
