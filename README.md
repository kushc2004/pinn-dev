# Physics-Guided Chiller Power Prediction

Leakage-safe regression benchmark for industrial chiller power using a
train-only symbolic surrogate as a neural-network regularizer.

The project predicts **Motor KW** from seven operating channels in 18,987
snapshots of one water-cooled chiller. It began as a Siemens exploration; the
original TensorFlow prototype remains under `legacy/`, while the active code is
a reproducible PyTorch/PySR rebuild.

## What changed in the clean protocol

The earlier experimental version had two evaluation problems: symbolic
regression sampled rows from the complete dataset, and the physics weight
`lambda` was compared directly on the high-load test set. Those results are
preserved in git history but are **not used as current evidence**.

The clean protocol enforces:

1. **Split before every learned statistic.**
2. **Train-only scaling.** The same feature/target scalers are used by PySR and
   the neural models.
3. **Train-only symbolic regression.** PySR never sees validation/test rows or
   targets.
4. **Validation-only lambda selection.** `lambda` is selected over
   `{0, .02, .05, .10, .20, .50, 1.0}` using mean validation MAE across three
   seeds. Allowing `lambda=0` prevents the experiment from forcing a physics
   prior when validation does not support it.
5. **Final test touched only after model selection.** The frozen baseline and
   selected physics-guided model are evaluated across five seeds.
6. **Paired uncertainty.** A hierarchical bootstrap reports a 95% confidence
   interval for the paired MAE improvement.
7. **Practical tabular references.** Fixed Ridge and histogram-gradient-
   boosting baselines are reported alongside the controlled MLP comparison.

## Data and target

Inputs:

- Evaporator inlet water temperature
- Evaporator outlet water temperature
- Evaporator flow rate
- Condenser refrigerant pressure
- Condenser inlet water temperature
- Condenser outlet water temperature
- Condenser flow rate

Target: **Motor KW**.

## Evaluation protocols

### 1. Random interpolation

Deterministic **64/16/20** train/validation/test split. This measures ordinary
interpolation under the empirical sample distribution; it is not presented as
future-time generalization.

### 2. High-load target-tail stress test

`Motor KW` quantiles define three non-overlapping partitions:

- bottom 80% -> train,
- 80th--90th percentile -> validation,
- top 10% -> final test.

On the current dataset this is approximately:

| Partition | Rows | Motor KW range |
|---|---:|---:|
| Train | 15,250 | 139.1--249.1 |
| Validation | 1,882 | 249.2--252.2 |
| Final test | 1,855 | 252.3--260.8 |

This is intentionally described as **high-load target-tail generalization**.
It is not called full physical operating-envelope extrapolation: the evaluator
also records, feature by feature, how much of the final test lies outside the
training marginal ranges.

## Symbolic surrogate

For each protocol separately, PySR receives a seeded 5,000-row subsample of
that protocol's **training split only**, already transformed by the train-only
scalers. It searches expressions over `+ - * /` with 20 outer iterations,
8 populations using PySR multithreading, 512-row mutation batches, and a
15-minute hard timeout. Hall-of-fame candidates are still evaluated on the
full 5,000-row discovery sample. The chosen op-tree and exact search settings
are stored in:

```text
results/equation_random.json
results/equation_high_load.json
```

The expression is best described as a **data-discovered symbolic surrogate
with physically interpretable feature dependence**, not a first-principles
thermodynamic law or conventional dimensional power law.

## Neural experiment

Both controlled neural models use exactly the same MLP:

```text
7 -> 64 -> 32 -> 16 -> 1
```

The baseline minimizes data MSE. The physics-guided model minimizes:

```text
MSE(y, y_hat) + lambda * MSE(f_SR(x), y_hat)
```

Because PySR and the network now share the same train-fitted coordinates,
`f_SR(x)` already lives in standardized target space; no second ad-hoc
standardization of the symbolic output is applied.

## Reproduce locally

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m unittest discover -s tests -v
sh scripts/run_full_experiment.sh
```

The pipeline stages are:

```text
sr_random
sr_high_load
tune_random
tune_high_load
final_random
final_high_load
evaluate
```

Every stage is cached in `results/state.json` with both the dataset SHA-256 and
`PROTOCOL_VERSION`. A dataset or evaluation-protocol change invalidates stale
state automatically.

## Outputs

After a complete run:

```text
results/
  equation_random.json
  equation_high_load.json
  lambda_selection_random.json
  lambda_selection_high_load.json
  final_summary.json
  metrics.csv
  validation/
  checkpoints/
  figures/
    parity_clean_protocol.png
    lambda_validation_sweep.png
```

`final_summary.json` is the source of truth for any CV metric. Do not reuse the
pre-clean 17.5% number unless the clean run independently reproduces it.

## Kaggle P100 run

The notebook in `kaggle_pinn/` clones the current GitHub `main`, verifies the
code/tests, stages the private Chiller-3 dataset, restores only compatible
artifacts, runs the full clean protocol, and exports `pinn_artifacts.tar.gz`.

```sh
kaggle kernels push -p kaggle_pinn
python3 scripts/watch_kaggle_kernel.py \
  --slug kushchaudhari/pinn-chiller-pipeline \
  --output outputs/logs/kaggle_pinn.log
```

## Claim boundary

This repository contains one chiller from one site. It is suitable for a
controlled study of symbolic regularization and high-load target-tail
generalization. It does **not** establish cross-chiller, cross-site, or
production-control performance.

## Provenance

Developed from an exploration carried out with Siemens resources
(Climatix/water-cooled chiller telemetry). The cleared Chiller-3 dataset and
derived artifacts are used here with permission; raw Climatix circuit data is
not redistributed.
