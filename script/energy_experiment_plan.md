# Energy Imputation Benchmark Plan

This benchmark is designed to make the UniFormTSV project more convincing for
energy-system experiments inspired by causal intervention prediction work.

## Dataset Scope

- `physionet_2012`: retained as a non-energy sanity baseline.
- `appliances_energy`: UCI low-energy house data with appliance load, lights,
  indoor sensors, and weather variables.
- `household_power`: UCI individual household electric power consumption,
  hourly resampled from minute-level measurements.
- `citylearn_zone5`: CityLearn multi-building district data with end-use loads,
  solar/weather/carbon variables, and storage/control semantics in the source
  environment.
- `opsd_germany`: Open Power System Data Germany grid time series with load,
  wind, solar, price, and calendar features.

## Stronger Experimental Matrix

Run:

```bash
./run_energy_benchmark.sh
```

The script evaluates:

- models: `mean`, `median`, `locf`, `saits`, `tefn`, `timemixerpp`, `uniformtsv`
- missing rates: `0.1`, `0.3`, `0.5`
- seeds: `42`, `123`, `456`
- training: `50` epochs with patience `10`
- model width: `d_model=64`, `d_ffn=128`, `n_heads=4`, `n_layers=2`
- TEFN complexity: `N_FOD=4` by default; use `N_FOD=2` for long windows on MPS.

After training, collect all metrics:

```bash
python script/collect_metrics.py --root output/imputation/cuda/energy_benchmark --out output/imputation/cuda/energy_benchmark/metrics_summary.csv
```

Use the CPU path instead of `cuda` if the runs were executed without GPUs.

Then build the routing table and lightweight router:

```bash
python script/model_routing_analysis.py --metrics output/imputation/cuda/energy_benchmark/metrics_summary.csv --out-dir output/imputation/cuda/energy_benchmark/routing
```

For the C-TCAR version of the routing method, first extract the 80-dimensional
representation, then train the RandomForest router:

```bash
python script/ctcar_features.py \
  --datasets "physionet_2012 appliances_energy household_power citylearn_zone5 opsd_germany" \
  --missing-rates "0.1 0.3 0.5" \
  --max-samples 1000 \
  --window-stride 8 \
  --seed 42 \
  --out output/imputation/cuda/energy_benchmark/ctcar_features.csv

python script/ctcar_routing_analysis.py \
  --metrics output/imputation/cuda/energy_benchmark/metrics_summary.csv \
  --ctcar output/imputation/cuda/energy_benchmark/ctcar_features.csv \
  --out-dir output/imputation/cuda/energy_benchmark/ctcar_routing
```
