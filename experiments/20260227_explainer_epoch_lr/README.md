# Explainer Epoch & Learning Rate Sweep (2026-02-27)

## Experiment

Grid search over GNNExplainer hyperparameters (learning rate and number of
epochs) using a fixed, pre-trained CDCRNN checkpoint. The goal was to determine
how explainer optimization settings affect the sparsity of the learned edge
importance masks (i.e., the number of edges exceeding the 0.5 importance
threshold).

All 18 jobs shared the same frozen model checkpoint
(`hospital_advan_plus_full_cdcrnn_x_log1p_y_log1p_h1_lr1e-05_hs16_epochs20000`)
and only varied the explainer parameters:

| Parameter          | Values                                  |
|--------------------|-----------------------------------------|
| `explain.lr`       | 0.01, 0.001, 0.0001                     |
| `explain.epochs`   | 200, 500, 1000, 2000, 5000, 10000       |


## Outputs

- **Per-run plots & data**: `plots/explainer_lr{LR}_epochs{EP}/`
  - `{version}.pdf` — loss curves, aggregate forecast, CBSA forecasts,
    explainer loss
  - `{version}_explain.pdf` — explainable edge count vs aggregate target
  - `eval_metrics.csv` — train/test RMSE & MAE
  - `explain_count.csv` — explainable edge count per snapshot
  - `explain_edges.csv` — list of all edges above the importance threshold
  - `importance_masks.parquet` — full per-edge importance scores
- **Combined PDF**: `plots/explainer_sweep_combined.pdf`
- **SLURM logs**: `experiments/20260227_explainer_epoch_lr/logs/`

## Findings

Higher learning rates and more epochs both drive the explainer toward sparser
masks (fewer edges above the 0.5 threshold):

| LR     | Epochs | Mean explainable edges/snapshot |
|--------|--------|---------------------------------|
| 0.0001 |    200 |                          18,718 |
| 0.0001 |    500 |                          18,512 |
| 0.0001 |  1,000 |                          18,144 |
| 0.0001 |  2,000 |                          17,381 |
| 0.0001 |  5,000 |                          15,002 |
| 0.0001 | 10,000 |                          11,884 |
| 0.001  |    200 |                          17,380 |
| 0.001  |    500 |                          14,783 |
| 0.001  |  1,000 |                          11,159 |
| 0.001  |  2,000 |                           9,939 |
| 0.001  |  5,000 |                           9,233 |
| 0.001  | 10,000 |                           9,094 |
| 0.01   |    200 |                          10,048 |
| 0.01   |    500 |                           9,357 |
| 0.01   |  1,000 |                           9,064 |
| 0.01   |  2,000 |                           8,983 |
| 0.01   |  5,000 |                           8,982 |
| 0.01   | 10,000 |                           8,985 |

At `lr=0.01`, the mask converges quickly — there is little difference between
2,000 and 10,000 epochs (8,983 vs 8,985 edges). At `lr=0.0001`, the mask is
still noticeably denser even after 10,000 epochs (11,884 edges), suggesting the
optimizer hasn't fully converged.

For practical use, `lr=0.01` with 1,000–2,000 epochs appears sufficient:
the mask has converged and additional epochs provide negligible sparsity gains.
