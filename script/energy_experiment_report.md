# Energy Benchmark Report

本轮修改参考了用户提供的论文 `Learning_to_Route_Models_for_Causal_Intervention_Prediction_in_Energy_Systems.pdf` 的核心思路：能源系统任务不能只停留在单一被动预测，还需要在多数据源、多缺失强度、多模型之间比较并选择合适模型。当前项目被扩展为一个可继续放大的能源时间序列插补/模型选择实验框架。

## Added Datasets

| Dataset | Source | Role |
| --- | --- | --- |
| `appliances_energy` | [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) | 室内环境、天气与能耗的多变量能源数据 |
| `household_power` | [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) | 家庭用电负荷与分项功率数据 |
| `citylearn_zone5` | [CityLearn Climate Zone 5](https://github.com/citylearn-project/CityLearn/tree/v1.0.0/data/Climate_Zone_5) | 多建筑能源需求、天气、太阳能与碳强度数据 |
| `opsd_germany` | [Open Power System Data time series](https://data.open-power-system-data.org/time_series/) | 德国负荷、风电、光伏与价格相关时间序列 |
| `physionet_2012` | Existing project dataset | 非能源强基准，用于观察跨领域稳健性 |

## Experiment Matrix

| Model | Datasets | Missing rates | Current scale |
| --- | --- | --- | --- |
| `mean` / `median` / `locf` | 5 datasets | 0.1, 0.3, 0.5 | Non-trainable baselines |
| `saits` | 5 datasets | 0.1, 0.3, 0.5 | 50 epochs, max 1000 samples |
| `tefn` | 5 datasets | 0.1, 0.3, 0.5 | 30 epochs, max 1000 samples, `n_fod=2` |
| `timemixerpp` | 5 datasets | 0.1, 0.3, 0.5 | 30 epochs, max 200 samples, compact model |
| `uniformtsv` | 3 energy datasets | 0.1, 0.3, 0.5 | 20 epochs, max 100 samples, T5-small backbone |

The aggregated metrics are saved at:

`output/imputation/mps/energy_benchmark/metrics_summary.csv`

The learned routing artifacts are saved at:

`output/imputation/mps/energy_benchmark/routing/`

The C-TCAR feature and routing artifacts are saved at:

`output/imputation/mps/energy_benchmark/ctcar_features.csv`

`output/imputation/mps/energy_benchmark/ctcar_routing/`

Reproducibility note: the completed benchmark rows are stored under `seed42` because the data split and missingness masks used seed 42. During the run, `main.py` still fixed the model training seed at 2025; this has now been corrected so future multi-seed runs make both data masking and model initialization honor `--random_seed`.

## Current Winners By MAE

| Dataset | Missing rate | Best model | MAE | MSE | RMSE | MRE |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| appliances_energy | 0.1 | locf | 0.1128 | 0.1550 | 0.3938 | 0.1119 |
| appliances_energy | 0.3 | locf | 0.1195 | 0.1609 | 0.4011 | 0.1188 |
| appliances_energy | 0.5 | locf | 0.1320 | 0.1720 | 0.4147 | 0.1309 |
| citylearn_zone5 | 0.1 | saits | 0.1333 | 0.0514 | 0.2267 | 0.1638 |
| citylearn_zone5 | 0.3 | saits | 0.1519 | 0.0660 | 0.2569 | 0.1868 |
| citylearn_zone5 | 0.5 | saits | 0.2122 | 0.1117 | 0.3342 | 0.2607 |
| household_power | 0.1 | saits | 0.1795 | 0.2085 | 0.4566 | 0.2481 |
| household_power | 0.3 | saits | 0.2056 | 0.2445 | 0.4944 | 0.2823 |
| household_power | 0.5 | saits | 0.2565 | 0.3035 | 0.5510 | 0.3524 |
| opsd_germany | 0.1 | saits | 0.1050 | 0.0276 | 0.1660 | 0.1151 |
| opsd_germany | 0.3 | saits | 0.1255 | 0.0389 | 0.1971 | 0.1376 |
| opsd_germany | 0.5 | saits | 0.1840 | 0.0775 | 0.2784 | 0.2023 |
| physionet_2012 | 0.1 | saits | 0.2694 | 0.2976 | 0.5455 | 0.3820 |
| physionet_2012 | 0.3 | saits | 0.3004 | 0.3624 | 0.6020 | 0.4243 |
| physionet_2012 | 0.5 | saits | 0.3485 | 0.4173 | 0.6460 | 0.4925 |

## Interpretation

The current results are more useful than a single-model run because they show heterogeneous winners across regimes:

- `locf` is strongest on `appliances_energy` for all three missing rates, beating neural models by a large margin. This is an important baseline result: simple temporal persistence is very strong for smooth appliance-energy series.
- `saits` is strongest on CityLearn, household power, OPSD, and PhysioNet, showing better robustness when the series are multivariate, heterogeneous, or sparse.
- `timemixerpp` remains competitive on some energy settings but does not dominate once LOCF is included.
- `tefn` can run locally after reducing `n_fod` from 16 to 2. The original `n_fod=16` caused exponential feature expansion and MPS buffer requests above 13 GiB on long windows.
- `uniformtsv` runs successfully with a T5-small backbone, but this small local experiment does not yet outperform the lighter baselines. This supports the paper-style claim that model routing/selection is necessary; a larger model is not automatically better for every energy-series regime.

## Router Diagnostics

`script/model_routing_analysis.py` builds a compact router from the metrics table. Current diagnostics:

| Diagnostic | Value |
| --- | ---: |
| Training accuracy | 1.000 |
| Leave-one-scenario-out accuracy | 1.000 |
| Leave-one-dataset-out accuracy | 0.600 |

The dataset-held-out score is intentionally harder and shows why future work should add more energy datasets and seeds before claiming a general router. The current router is still useful as a reproducible model-selection artifact over the evaluated benchmark matrix.

## C-TCAR Router

`script/ctcar_features.py` now extracts an 80-dimensional C-TCAR representation:

| Group | Dimensions |
| --- | ---: |
| Statistical | 20 |
| Temporal ACF | 24 |
| Spectral FFT | 19 |
| Causal structural proxy | 10 |
| Confounding proxy | 7 |

`script/ctcar_routing_analysis.py` trains a RandomForest router using those features plus missing rate. Current diagnostics:

| Diagnostic | Value |
| --- | ---: |
| Training accuracy | 1.000 |
| Leave-one-scenario-out accuracy | 1.000 |
| Leave-one-dataset-out accuracy | 0.800 |

This is still a causal-aware proxy, not a full intervention-effect estimator: the project does not yet contain true intervention labels, ATE ground truth, or do-calculus evaluation. However, it is now much closer to the paper method than the previous metadata-only router.

## Reproduce Or Scale Up

Fast local benchmark:

```bash
EPOCH=50 PATIENCE=10 MAX_SAMPLES=1000 WINDOW_STRIDE=8 SEEDS="42" ./run_energy_benchmark.sh
```

Larger multi-seed benchmark:

```bash
EPOCH=50 PATIENCE=10 MAX_SAMPLES=20000 WINDOW_STRIDE=8 SEEDS="42 123 456" ./run_energy_benchmark.sh
```

Aggregate metrics:

```bash
.venv/bin/python script/collect_metrics.py \
  --root output/imputation/mps/energy_benchmark \
  --out output/imputation/mps/energy_benchmark/metrics_summary.csv
```

Build routing table and router:

```bash
.venv/bin/python script/model_routing_analysis.py \
  --metrics output/imputation/mps/energy_benchmark/metrics_summary.csv \
  --out-dir output/imputation/mps/energy_benchmark/routing
```

Extract C-TCAR features and train the C-TCAR router:

```bash
.venv/bin/python script/ctcar_features.py \
  --datasets "physionet_2012 appliances_energy household_power citylearn_zone5 opsd_germany" \
  --missing-rates "0.1 0.3 0.5" \
  --max-samples 1000 \
  --window-stride 8 \
  --seed 42 \
  --out output/imputation/mps/energy_benchmark/ctcar_features.csv

.venv/bin/python script/ctcar_routing_analysis.py \
  --metrics output/imputation/mps/energy_benchmark/metrics_summary.csv \
  --ctcar output/imputation/mps/energy_benchmark/ctcar_features.csv \
  --out-dir output/imputation/mps/energy_benchmark/ctcar_routing
```
