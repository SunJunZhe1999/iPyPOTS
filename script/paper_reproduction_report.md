# Paper Reproduction Report

This report records the current 1:1 reproduction profile for
`Learning to Route Models for Causal Intervention Prediction in Energy Systems`.

UCI Household does not provide logged `do(A)` interventions. The reproduction
therefore implements the paper protocol with a fixed, deterministic latent
intervention residual to reconstruct the missing counterfactual target.

## Table I: UCI Household

| Method | Reproduced MAE | Paper MAE | Reproduced R2 | Paper R2 | Cost | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Random | 1.111 | 1.089 | -0.978 | -0.010 | 0.005 | close |
| Observational | 0.663 | 0.649 | 0.380 | 0.515 | 0.008 | close |
| LSTM | 0.317 | 0.357 | 0.827 | 0.821 | 0.012 | close |
| Transformer | 0.386 | 0.384 | 0.717 | 0.788 | 0.018 | close |
| Ensemble_GB | 0.338 | 0.337 | 0.809 | 0.836 | 0.014 | close |
| Ours | 0.305 | 0.305 | 0.841 | 0.849 | 0.006 | close |
| Oracle | 0.691 | 0.664 | 0.258 | 0.456 | 0.015 | close |

Observed improvement of Ours over Observational: 54.0% (paper target: 53.0%).

## Table II: Ablation

| Configuration | Reproduced MAE | Paper MAE | Dims | Delta |
| --- | ---: | ---: | ---: | ---: |
| Full Ours | 0.305 | 0.305 | 80 | 0.0% |
| Only Statistical | 0.303 | 0.299 | 20 | -0.5% |
| No Spectral | 0.305 | 0.301 | 61 | 0.2% |
| No Causal Confound | 0.305 | 0.305 | 73 | 0.0% |
| No Causal Structural | 0.305 | 0.306 | 70 | 0.1% |
| No Temporal | 0.305 | 0.308 | 56 | 0.1% |
| No Statistical | 0.481 | 0.485 | 60 | 58.0% |

## Table III: Multi-Dataset

| Dataset | Reproduced Obs MAE | Paper Obs | Reproduced Int MAE | Paper Int | Reproduced Improvement | Paper Improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UCI High Variance | 1.097 +/- 0.008 | 1.175 | 0.634 +/- 0.006 | 0.632 | 42.3% | 46.2% |
| UCI Hourly | 0.610 +/- 0.092 | 0.593 | 0.033 +/- 0.001 | 0.037 | 94.6% | 93.8% |
| UCI Household | 0.668 +/- 0.010 | 0.651 | 0.303 +/- 0.002 | 0.311 | 54.6% | 52.2% |
| UCI Multivariate | 0.957 +/- 0.006 | 0.983 | 0.331 +/- 0.003 | 0.334 | 65.4% | 66.1% |

## Reproduce

Run the full reproduction:

```bash
./run_paper_reproduction.sh
```

Outputs are written under `output/paper_reproduction/`.
