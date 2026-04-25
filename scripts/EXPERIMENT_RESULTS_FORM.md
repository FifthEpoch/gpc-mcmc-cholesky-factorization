# Experiment Results Form

Use this document so everyone records runs in the same shape. You can copy the **blank row** in the table for each new job, or append rows to `experiment_results_runs.csv` (same columns).

---

## How to fill a row

1. Run your experiment; note Slurm `job_id`, account/partition, and git commit if possible.
2. Copy metrics from `data/exp3_<dataset>_results.json` or `data/exp4_<dataset>_results.json` (or your own JSON).
3. Set `embedding_source` to `self_extracted`, `partner_hg`, or `other`.
4. Use a unique `record_id` (e.g. `wc3013-2026-04-24-exp3-pcam-a100`).

---

## Field glossary

| Field | Meaning |
|-------|---------|
| `record_id` | Unique label for this row |
| `experiment` | `exp3`, `exp4`, or `other` |
| `method_name` | e.g. `mlp`, `tabpfn` |
| `dataset` | `pcam`, `camelyon17`, `embed` |
| `embedding_source` | Where embeddings came from |
| `embedding_root` | Path passed as `EMBEDDING_DIR` or default `data/embeddings` |
| `embedding_variant` | e.g. `densenet121`, `dinov2_vitl14`, `projected_512` |
| `feature_dim` | Embedding width (e.g. 512, 1024) |
| `threshold` | Classification threshold (we use 0.5 in code) |
| `fit_or_train_time_sec` | Exp3: `train_time_sec`; Exp4: `fit_time_sec` |
| `inference_time_sec` | Same key in both JSON files |
| `total_pipeline_time_sec` | Optional: include embedding extraction if you timed full pipeline |

---

## Results table (copy blank row below)

| record_id | experiment | method_name | dataset | embedding_source | embedding_root | embedding_variant | feature_dim | threshold | seed | job_id | account_partition | node_gpu | code_ref | run_timestamp_utc | auroc | auprc | accuracy | sensitivity_tpr | specificity_tnr | false_positive_rate | false_negative_rate | brier | ece | tp | tn | fp | fn | fit_or_train_time_sec | inference_time_sec | total_pipeline_time_sec | n_train | n_val | n_test | notes |
|-----------|--------------|-------------|---------|------------------|----------------|-------------------|-------------|-----------|------|--------|-------------------|----------|----------|-------------------|-------|-------|----------|-----------------|-----------------|----------------------|----------------------|-------|-----|----|----|----|----|------------------------|---------------------|-------------------------|---------|-------|--------|-------|
| | exp3 | mlp | pcam | self_extracted | data/embeddings | densenet121 | 1024 | 0.5 | 42 | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

---

## JSON mapping (Exp3 / Exp4)

From `exp3_*_results.json` / `exp4_*_results.json` after recent code updates:

- `auroc`, `auprc`, `accuracy`, `brier`, `ece`
- `sensitivity` → `sensitivity_tpr`
- `false_negative_rate`, `false_positive_rate`
- `tp`, `tn`, `fp`, `fn`
- `true_positive_rate`, `true_negative_rate` → optional; `specificity_tnr` = `true_negative_rate`
- Exp3: `train_time_sec` → `fit_or_train_time_sec`
- Exp4: `fit_time_sec` → `fit_or_train_time_sec`
- `n_train`, `n_test`; add `n_val` manually if you log it

---

## Sanity checks before saving

- `false_negative_rate ≈ fn / (tp + fn)` (floating rounding OK)
- `false_positive_rate ≈ fp / (fp + tn)`
- All counts non-negative integers; `tp + tn + fp + fn = n_test` at the evaluated split
