# Expanded Energy C-TCAR Benchmark Report

This report summarizes the current expanded benchmark after adding more energy datasets and a wider model matrix.

## Scale

| Item | Value |
| --- | ---: |
| Completed metric rows | 1262 |
| Distinct datasets | 11 |
| Distinct models | 9 |
| Distinct missing rates | 4 |
| Distinct seeds | 4 |
| Dataset/missing-rate scenarios | 44 |
| C-TCAR rows | 44 |

## Datasets

`appliances_energy`, `citylearn_zone5`, `eld`, `etth1`, `etth2`, `ettm1`, `ettm2`, `household_power`, `opsd_germany`, `physionet_2012`, `solar`

## Models

`gpt4ts`, `locf`, `mean`, `median`, `moment`, `saits`, `tefn`, `tslanet`, `uniformtsv`

## Coverage By Dataset

| dataset | runs | models | missing_rates | mean_MAE |
| --- | --- | --- | --- | --- |
| appliances_energy | 118 | 9 | 4 | 0.6644 |
| citylearn_zone5 | 112 | 7 | 4 | 0.6067 |
| eld | 112 | 7 | 4 | 0.8077 |
| etth1 | 118 | 9 | 4 | 0.7187 |
| etth2 | 112 | 7 | 4 | 0.7046 |
| ettm1 | 118 | 9 | 4 | 0.5974 |
| ettm2 | 112 | 7 | 4 | 0.5896 |
| household_power | 112 | 7 | 4 | 0.6038 |
| opsd_germany | 118 | 9 | 4 | 0.6436 |
| physionet_2012 | 112 | 7 | 4 | 0.6068 |
| solar | 118 | 9 | 4 | 0.3306 |

## Coverage By Model

| model | runs | datasets | mean_MAE |
| --- | --- | --- | --- |
| locf | 176 | 11 | 0.2322 |
| saits | 176 | 11 | 0.3763 |
| median | 176 | 11 | 0.6325 |
| mean | 176 | 11 | 0.6683 |
| moment | 176 | 11 | 0.7704 |
| uniformtsv | 176 | 11 | 0.7704 |
| tefn | 176 | 11 | 0.9348 |
| gpt4ts | 15 | 5 | 0.1694 |
| tslanet | 15 | 5 | 0.8882 |

## Current Winners By Scenario

| dataset | missing_rate | best_model | MAE | MSE | RMSE | MRE | seeds_used |
| --- | --- | --- | --- | --- | --- | --- | --- |
| appliances_energy | 0.1000 | gpt4ts | 0.0916 | 0.0414 | 0.2035 | 0.0900 | 1 |
| appliances_energy | 0.3000 | locf | 0.1190 | 0.1595 | 0.3994 | 0.1180 | 4 |
| appliances_energy | 0.5000 | locf | 0.1310 | 0.1697 | 0.4119 | 0.1299 | 4 |
| appliances_energy | 0.7000 | locf | 0.1558 | 0.1922 | 0.4384 | 0.1545 | 4 |
| citylearn_zone5 | 0.1000 | saits | 0.1643 | 0.0782 | 0.2733 | 0.2012 | 4 |
| citylearn_zone5 | 0.3000 | saits | 0.1887 | 0.1014 | 0.3111 | 0.2314 | 4 |
| citylearn_zone5 | 0.5000 | saits | 0.2464 | 0.1548 | 0.3855 | 0.3022 | 4 |
| citylearn_zone5 | 0.7000 | saits | 0.3694 | 0.2919 | 0.5334 | 0.4531 | 4 |
| eld | 0.1000 | locf | 0.0846 | 0.0429 | 0.2069 | 0.0453 | 4 |
| eld | 0.3000 | locf | 0.0917 | 0.0502 | 0.2240 | 0.0491 | 4 |
| eld | 0.5000 | locf | 0.1032 | 0.0655 | 0.2558 | 0.0552 | 4 |
| eld | 0.7000 | locf | 0.1246 | 0.0967 | 0.3109 | 0.0667 | 4 |
| etth1 | 0.1000 | gpt4ts | 0.2240 | 0.1113 | 0.3336 | 0.2697 | 1 |
| etth1 | 0.3000 | gpt4ts | 0.2658 | 0.1631 | 0.4038 | 0.3133 | 1 |
| etth1 | 0.5000 | gpt4ts | 0.3138 | 0.2220 | 0.4711 | 0.3676 | 1 |
| etth1 | 0.7000 | saits | 0.4598 | 0.4561 | 0.6724 | 0.5420 | 4 |
| etth2 | 0.1000 | locf | 0.1735 | 0.0836 | 0.2891 | 0.1705 | 4 |
| etth2 | 0.3000 | locf | 0.1884 | 0.0967 | 0.3109 | 0.1845 | 4 |
| etth2 | 0.5000 | locf | 0.2086 | 0.1135 | 0.3368 | 0.2040 | 4 |
| etth2 | 0.7000 | locf | 0.2448 | 0.1490 | 0.3859 | 0.2396 | 4 |
| ettm1 | 0.1000 | saits | 0.1470 | 0.0497 | 0.2209 | 0.1737 | 4 |
| ettm1 | 0.3000 | locf | 0.1694 | 0.0915 | 0.3024 | 0.1992 | 4 |
| ettm1 | 0.5000 | locf | 0.1940 | 0.1185 | 0.3443 | 0.2280 | 4 |
| ettm1 | 0.7000 | locf | 0.2427 | 0.1875 | 0.4329 | 0.2861 | 4 |
| ettm2 | 0.1000 | locf | 0.0994 | 0.0333 | 0.1824 | 0.0973 | 4 |
| ettm2 | 0.3000 | locf | 0.1099 | 0.0401 | 0.2003 | 0.1079 | 4 |
| ettm2 | 0.5000 | locf | 0.1230 | 0.0484 | 0.2200 | 0.1207 | 4 |
| ettm2 | 0.7000 | locf | 0.1454 | 0.0644 | 0.2538 | 0.1429 | 4 |
| household_power | 0.1000 | saits | 0.2470 | 0.2599 | 0.4998 | 0.3398 | 4 |
| household_power | 0.3000 | saits | 0.2769 | 0.3083 | 0.5444 | 0.3806 | 4 |
| household_power | 0.5000 | saits | 0.3339 | 0.3852 | 0.6124 | 0.4590 | 4 |
| household_power | 0.7000 | saits | 0.4186 | 0.4993 | 0.7023 | 0.5752 | 4 |
| opsd_germany | 0.1000 | gpt4ts | 0.1198 | 0.0346 | 0.1859 | 0.1331 | 1 |
| opsd_germany | 0.3000 | saits | 0.1593 | 0.0775 | 0.2504 | 0.1760 | 4 |
| opsd_germany | 0.5000 | saits | 0.2200 | 0.1307 | 0.3339 | 0.2428 | 4 |
| opsd_germany | 0.7000 | locf | 0.3778 | 0.4314 | 0.6568 | 0.4170 | 4 |
| physionet_2012 | 0.1000 | saits | 0.3325 | 0.4629 | 0.6577 | 0.4800 | 4 |
| physionet_2012 | 0.3000 | saits | 0.3597 | 0.4506 | 0.6658 | 0.5224 | 4 |
| physionet_2012 | 0.5000 | saits | 0.4131 | 0.5195 | 0.7167 | 0.5982 | 4 |
| physionet_2012 | 0.7000 | locf | 0.4655 | 0.7227 | 0.8473 | 0.6779 | 4 |
| solar | 0.1000 | locf | 0.0500 | 0.0129 | 0.1138 | 0.0645 | 4 |
| solar | 0.3000 | locf | 0.0601 | 0.0194 | 0.1392 | 0.0775 | 4 |
| solar | 0.5000 | locf | 0.0758 | 0.0309 | 0.1759 | 0.0977 | 4 |
| solar | 0.7000 | locf | 0.0995 | 0.0544 | 0.2330 | 0.1295 | 4 |

## Router Diagnostics

| Router | Training | Leave-one-scenario | Leave-one-dataset |
| --- | ---: | ---: | ---: |
| Metadata router | 0.9090909090909091 | 0.8181818181818182 | 0.75 |
| C-TCAR router | 1.0 | 0.8409090909090909 | 0.5909090909090909 |

## Notes

Experiments were run on the local Apple Silicon environment, so the benchmark prioritizes broad coverage over very large backbones or multi-seed saturation.
The following heavier models have partial local coverage because they are expensive on this machine: `gpt4ts` (15/44), `tslanet` (15/44).
On the current runs, the C-TCAR router matches or exceeds the metadata router under leave-one-scenario validation.

## Important Caveat

The current C-TCAR implementation is a causal-aware routing proxy. True intervention labels, ATE ground truth, and full do-calculus evaluation are still future work.
