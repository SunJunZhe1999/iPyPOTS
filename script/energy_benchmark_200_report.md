# Expanded Energy C-TCAR Benchmark Report

This report summarizes the current expanded benchmark after adding more energy datasets and a wider model matrix.

## Scale

| Item | Value |
| --- | ---: |
| Completed metric rows | 514 |
| Distinct datasets | 11 |
| Distinct models | 9 |
| Distinct missing rates | 4 |
| Distinct seeds | 2 |
| Dataset/missing-rate scenarios | 44 |
| C-TCAR rows | 44 |

## Datasets

`appliances_energy`, `citylearn_zone5`, `eld`, `etth1`, `etth2`, `ettm1`, `ettm2`, `household_power`, `opsd_germany`, `physionet_2012`, `solar`

## Models

`gpt4ts`, `locf`, `mean`, `median`, `moment`, `saits`, `tefn`, `tslanet`, `uniformtsv`

## Coverage By Dataset

| dataset | runs | models | missing_rates | mean_MAE |
| --- | --- | --- | --- | --- |
| appliances_energy | 50 | 9 | 4 | 0.7264 |
| citylearn_zone5 | 44 | 7 | 4 | 0.6597 |
| eld | 44 | 7 | 4 | 1.0255 |
| etth1 | 50 | 9 | 4 | 0.7913 |
| etth2 | 44 | 7 | 4 | 0.8394 |
| ettm1 | 50 | 9 | 4 | 0.6269 |
| ettm2 | 44 | 7 | 4 | 0.6876 |
| household_power | 44 | 7 | 4 | 0.6423 |
| opsd_germany | 50 | 9 | 4 | 0.6564 |
| physionet_2012 | 44 | 7 | 4 | 0.6538 |
| solar | 50 | 9 | 4 | 0.3676 |

## Coverage By Model

| model | runs | datasets | mean_MAE |
| --- | --- | --- | --- |
| saits | 88 | 11 | 0.4051 |
| moment | 88 | 11 | 0.7886 |
| uniformtsv | 88 | 11 | 0.7886 |
| tefn | 88 | 11 | 1.1213 |
| locf | 44 | 11 | 0.2316 |
| median | 44 | 11 | 0.6369 |
| mean | 44 | 11 | 0.6727 |
| gpt4ts | 15 | 5 | 0.1694 |
| tslanet | 15 | 5 | 0.8882 |

## Current Winners By Scenario

| dataset | missing_rate | best_model | MAE | MSE | RMSE | MRE | seeds_used |
| --- | --- | --- | --- | --- | --- | --- | --- |
| appliances_energy | 0.1000 | gpt4ts | 0.0916 | 0.0414 | 0.2035 | 0.0900 | 1 |
| appliances_energy | 0.3000 | locf | 0.1188 | 0.1599 | 0.3999 | 0.1179 | 1 |
| appliances_energy | 0.5000 | locf | 0.1307 | 0.1686 | 0.4106 | 0.1299 | 1 |
| appliances_energy | 0.7000 | locf | 0.1549 | 0.1903 | 0.4362 | 0.1536 | 1 |
| citylearn_zone5 | 0.1000 | locf | 0.1792 | 0.1295 | 0.3599 | 0.2192 | 1 |
| citylearn_zone5 | 0.3000 | locf | 0.2178 | 0.1871 | 0.4325 | 0.2672 | 1 |
| citylearn_zone5 | 0.5000 | locf | 0.2820 | 0.2961 | 0.5441 | 0.3459 | 1 |
| citylearn_zone5 | 0.7000 | locf | 0.3988 | 0.5228 | 0.7231 | 0.4888 | 1 |
| eld | 0.1000 | locf | 0.0872 | 0.0498 | 0.2231 | 0.0471 | 1 |
| eld | 0.3000 | locf | 0.0940 | 0.0543 | 0.2331 | 0.0507 | 1 |
| eld | 0.5000 | locf | 0.1058 | 0.0707 | 0.2658 | 0.0569 | 1 |
| eld | 0.7000 | locf | 0.1292 | 0.1055 | 0.3248 | 0.0695 | 1 |
| etth1 | 0.1000 | gpt4ts | 0.2240 | 0.1113 | 0.3336 | 0.2697 | 1 |
| etth1 | 0.3000 | gpt4ts | 0.2658 | 0.1631 | 0.4038 | 0.3133 | 1 |
| etth1 | 0.5000 | gpt4ts | 0.3138 | 0.2220 | 0.4711 | 0.3676 | 1 |
| etth1 | 0.7000 | saits | 0.4961 | 0.5393 | 0.7344 | 0.5859 | 2 |
| etth2 | 0.1000 | locf | 0.1767 | 0.0886 | 0.2977 | 0.1738 | 1 |
| etth2 | 0.3000 | locf | 0.1903 | 0.0980 | 0.3131 | 0.1868 | 1 |
| etth2 | 0.5000 | locf | 0.2139 | 0.1205 | 0.3471 | 0.2088 | 1 |
| etth2 | 0.7000 | locf | 0.2471 | 0.1523 | 0.3902 | 0.2416 | 1 |
| ettm1 | 0.1000 | locf | 0.1511 | 0.0718 | 0.2679 | 0.1798 | 1 |
| ettm1 | 0.3000 | locf | 0.1684 | 0.0903 | 0.3005 | 0.1989 | 1 |
| ettm1 | 0.5000 | locf | 0.1934 | 0.1170 | 0.3420 | 0.2284 | 1 |
| ettm1 | 0.7000 | locf | 0.2416 | 0.1841 | 0.4291 | 0.2875 | 1 |
| ettm2 | 0.1000 | locf | 0.0967 | 0.0313 | 0.1768 | 0.0927 | 1 |
| ettm2 | 0.3000 | locf | 0.1085 | 0.0394 | 0.1986 | 0.1054 | 1 |
| ettm2 | 0.5000 | locf | 0.1214 | 0.0471 | 0.2170 | 0.1185 | 1 |
| ettm2 | 0.7000 | locf | 0.1441 | 0.0638 | 0.2525 | 0.1409 | 1 |
| household_power | 0.1000 | saits | 0.3288 | 0.3595 | 0.5994 | 0.4527 | 2 |
| household_power | 0.3000 | saits | 0.3664 | 0.4269 | 0.6534 | 0.5035 | 2 |
| household_power | 0.5000 | saits | 0.4297 | 0.5084 | 0.7130 | 0.5909 | 2 |
| household_power | 0.7000 | saits | 0.5078 | 0.6084 | 0.7800 | 0.6980 | 2 |
| opsd_germany | 0.1000 | gpt4ts | 0.1198 | 0.0346 | 0.1859 | 0.1331 | 1 |
| opsd_germany | 0.3000 | gpt4ts | 0.1768 | 0.0826 | 0.2873 | 0.1958 | 1 |
| opsd_germany | 0.5000 | locf | 0.2551 | 0.2025 | 0.4500 | 0.2814 | 1 |
| opsd_germany | 0.7000 | locf | 0.3794 | 0.4333 | 0.6582 | 0.4199 | 1 |
| physionet_2012 | 0.1000 | saits | 0.3551 | 0.6170 | 0.7607 | 0.5023 | 2 |
| physionet_2012 | 0.3000 | saits | 0.3834 | 0.4974 | 0.7024 | 0.5487 | 2 |
| physionet_2012 | 0.5000 | locf | 0.4198 | 0.5260 | 0.7252 | 0.6074 | 1 |
| physionet_2012 | 0.7000 | locf | 0.4593 | 0.5914 | 0.7690 | 0.6628 | 1 |
| solar | 0.1000 | locf | 0.0501 | 0.0124 | 0.1115 | 0.0622 | 1 |
| solar | 0.3000 | locf | 0.0609 | 0.0192 | 0.1387 | 0.0755 | 1 |
| solar | 0.5000 | gpt4ts | 0.0761 | 0.0284 | 0.1686 | 0.0980 | 1 |
| solar | 0.7000 | locf | 0.0879 | 0.0479 | 0.2188 | 0.1133 | 1 |

## Router Diagnostics

| Router | Training | Leave-one-scenario | Leave-one-dataset |
| --- | ---: | ---: | ---: |
| Metadata router | 0.8636363636363636 | 0.75 | 0.6818181818181818 |
| C-TCAR router | 1.0 | 0.75 | 0.6818181818181818 |

## Notes

Experiments were run on the local Apple Silicon environment, so the benchmark prioritizes broad coverage over very large backbones or multi-seed saturation.
The following heavier models have partial local coverage because they are expensive on this machine: `gpt4ts` (15/44), `tslanet` (15/44).
On the current runs, the C-TCAR router matches or exceeds the metadata router under leave-one-scenario validation.

## Important Caveat

The current C-TCAR implementation is a causal-aware routing proxy. True intervention labels, ATE ground truth, and full do-calculus evaluation are still future work.
