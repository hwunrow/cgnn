# Rerun Explainer — Hospital Data (2026-03-13)

## Experiment

Re-run GNNExplainer on a pre-trained CDCRNN checkpoint using case data
(`caes_advan_plus_cdcrnn`) since this original experiment
did not run the GNNExplainer for the test period. Training is skipped
(`+skip_training=True`); only the explanation step is executed against the
existing checkpoint.

The explainer settings used were selected based on the convergence analysis
from `20260227_explainer_epoch_lr`:

| Parameter           | Value                                             |
|---------------------|---------------------------------------------------|
| `model`             | `cdcrnn_case_hs16`.                               |
| `checkpoint_version`| `case_advan_plus_cdcrnn_h1_lr1e-05_hs16`          |
| `data.end_date`     | `12/31/2022`                                      |
| `explain.lr`        | 0.01                                              |
| `explain.epochs`    | 200, 2000                                         |

## Outputs

Outputs are written into the existing checkpoint directory:
`plots/case_advan_plus_cdcrnn_h1_lr1e-05_hs16_rerun_explainer_lr0.01_epochs_{N_EPOCHS}/`

- `{version}.pdf` — loss curves, aggregate forecast, CBSA forecasts, explainer loss
- `{version}_explain.pdf` — explainable edge count vs aggregate target
- `eval_metrics.csv` — train/test RMSE & MAE
- `explain_count.csv` — explainable edge count per snapshot
- `explain_edges.csv` — list of all edges above the importance threshold
- `importance_masks.parquet` — full per-edge importance scores

**SLURM logs**: `experiments/20260313_rerun_explainer_case_data/logs/`
