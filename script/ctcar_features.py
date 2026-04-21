#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C-TCAR feature extraction for causal-aware model routing experiments.

This module implements an 80-dimensional Causal Time-series
Characteristic-Aware Representation inspired by the uploaded routing paper.
The causal features are pragmatic proxies built from lagged dependency graphs,
because the current project does not yet contain interventional labels or a
full DYNOTEARS dependency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pypots.data.dataset.load_prepare_dataset import DatasetPreparator


DATASET_DIMS = {
    "physionet_2012": {"n_steps": 48, "n_features": 37},
    "appliances_energy": {"n_steps": 72, "n_features": 32},
    "household_power": {"n_steps": 168, "n_features": 11},
    "citylearn_zone5": {"n_steps": 168, "n_features": 20},
    "opsd_germany": {"n_steps": 168, "n_features": 10},
}

STAT_FEATURE_NAMES = [
    "stat_mean",
    "stat_median",
    "stat_std",
    "stat_variance",
    "stat_iqr",
    "stat_min",
    "stat_max",
    "stat_range",
    "stat_q05",
    "stat_q25",
    "stat_q75",
    "stat_q95",
    "stat_skewness",
    "stat_kurtosis",
    "stat_mad",
    "stat_rms",
    "stat_abs_mean",
    "stat_coeff_variation",
    "stat_missing_fraction",
    "stat_observed_fraction",
]

TEMPORAL_FEATURE_NAMES = [f"temporal_acf_lag_{lag:02d}" for lag in range(1, 25)]

SPECTRAL_FEATURE_NAMES = (
    [f"spectral_fft_coeff_{idx:02d}" for idx in range(1, 11)]
    + [
        "spectral_entropy",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff_85",
        "spectral_dominant_freq",
        "spectral_dominant_power_share",
        "spectral_low_energy_ratio",
        "spectral_mid_energy_ratio",
        "spectral_high_energy_ratio",
    ]
)

CAUSAL_STRUCTURAL_FEATURE_NAMES = [
    "causal_dag_density",
    "causal_max_indegree",
    "causal_mean_indegree",
    "causal_max_outdegree",
    "causal_mean_outdegree",
    "causal_mean_abs_edge_weight",
    "causal_max_abs_edge_weight",
    "causal_graph_asymmetry",
    "causal_action_to_outcome_path_strength",
    "causal_action_outcome_direct_weight",
]

CONFOUNDING_FEATURE_NAMES = [
    "confounding_count",
    "confounding_fraction",
    "confounding_mean_strength",
    "confounding_max_strength",
    "confounding_backdoor_path_proxy",
    "confounding_adjustment_set_size",
    "confounding_bias_proxy",
]

CTCAR_FEATURE_NAMES = (
    STAT_FEATURE_NAMES
    + TEMPORAL_FEATURE_NAMES
    + SPECTRAL_FEATURE_NAMES
    + CAUSAL_STRUCTURAL_FEATURE_NAMES
    + CONFOUNDING_FEATURE_NAMES
)

assert len(CTCAR_FEATURE_NAMES) == 80


def _safe_float(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(value)


def _nan_flatten(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _standardize_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return np.zeros_like(values, dtype=np.float64)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    values = np.where(finite, values, mean)
    return (values - mean) / std


def _fill_nan_by_feature(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    filled = X.copy()
    for feature_idx in range(filled.shape[-1]):
        feature = filled[:, :, feature_idx]
        if np.isfinite(feature).any():
            value = np.nanmean(feature)
        else:
            value = 0.0
        feature[~np.isfinite(feature)] = value
        filled[:, :, feature_idx] = feature
    return filled


def _statistical_features(X: np.ndarray) -> List[float]:
    flat_all = np.asarray(X, dtype=np.float64).reshape(-1)
    flat = _nan_flatten(X)
    if flat.size == 0:
        flat = np.zeros(1, dtype=np.float64)

    mean = np.mean(flat)
    median = np.median(flat)
    std = np.std(flat)
    var = np.var(flat)
    q05, q25, q75, q95 = np.quantile(flat, [0.05, 0.25, 0.75, 0.95])
    centered = flat - mean
    std_safe = std if std > 1e-12 else 1.0
    skewness = np.mean((centered / std_safe) ** 3)
    kurtosis = np.mean((centered / std_safe) ** 4) - 3.0
    mad = np.mean(np.abs(centered))
    rms = np.sqrt(np.mean(flat**2))
    abs_mean = np.mean(np.abs(flat))
    coeff_variation = std / (abs(mean) + 1e-8)
    missing_fraction = 1.0 - np.isfinite(flat_all).mean()

    return [
        mean,
        median,
        std,
        var,
        q75 - q25,
        np.min(flat),
        np.max(flat),
        np.max(flat) - np.min(flat),
        q05,
        q25,
        q75,
        q95,
        skewness,
        kurtosis,
        mad,
        rms,
        abs_mean,
        coeff_variation,
        missing_fraction,
        1.0 - missing_fraction,
    ]


def _acf_1d(values: np.ndarray, lag: int) -> float:
    if values.size <= lag:
        return 0.0
    left = values[:-lag]
    right = values[lag:]
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return 0.0
    left = left[mask]
    right = right[mask]
    left_std = np.std(left)
    right_std = np.std(right)
    if left_std < 1e-12 or right_std < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _temporal_features(X: np.ndarray) -> List[float]:
    features = []
    max_lag = 24
    for lag in range(1, max_lag + 1):
        correlations = []
        for feature_idx in range(X.shape[-1]):
            series = X[:, :, feature_idx].reshape(-1)
            correlations.append(_acf_1d(series, lag))
        features.append(np.nanmean(np.abs(correlations)))
    return features


def _spectral_features(X: np.ndarray) -> List[float]:
    filled = _fill_nan_by_feature(X)
    series = np.nanmean(filled, axis=0)
    if series.ndim == 1:
        series = series[:, None]

    magnitudes = []
    for feature_idx in range(series.shape[-1]):
        feature = _standardize_vector(series[:, feature_idx])
        spectrum = np.abs(np.fft.rfft(feature))
        if spectrum.size <= 1:
            spectrum = np.zeros(2, dtype=np.float64)
        magnitudes.append(spectrum[1:])

    max_len = max(len(m) for m in magnitudes)
    padded = np.zeros((len(magnitudes), max_len), dtype=np.float64)
    for idx, mag in enumerate(magnitudes):
        padded[idx, : len(mag)] = mag
    mean_mag = padded.mean(axis=0)
    total = mean_mag.sum()
    prob = mean_mag / (total + 1e-12)

    coeffs = np.zeros(10, dtype=np.float64)
    coeffs[: min(10, len(prob))] = prob[:10]

    freq = np.linspace(0.0, 0.5, len(prob), endpoint=True) if len(prob) else np.zeros(1)
    entropy = -np.sum(prob * np.log(prob + 1e-12)) / np.log(len(prob) + 1e-12) if len(prob) > 1 else 0.0
    centroid = np.sum(freq * prob) if len(prob) else 0.0
    bandwidth = np.sqrt(np.sum(((freq - centroid) ** 2) * prob)) if len(prob) else 0.0
    cumulative = np.cumsum(prob)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85)) if len(cumulative) else 0
    rolloff = freq[min(rolloff_idx, len(freq) - 1)] if len(freq) else 0.0
    dominant_idx = int(np.argmax(prob)) if len(prob) else 0
    dominant_freq = freq[dominant_idx] if len(freq) else 0.0
    dominant_share = prob[dominant_idx] if len(prob) else 0.0

    split1 = max(1, len(prob) // 3)
    split2 = max(split1 + 1, 2 * len(prob) // 3)
    low_ratio = prob[:split1].sum()
    mid_ratio = prob[split1:split2].sum()
    high_ratio = prob[split2:].sum()

    return list(coeffs) + [
        entropy,
        centroid,
        bandwidth,
        rolloff,
        dominant_freq,
        dominant_share,
        low_ratio,
        mid_ratio,
        high_ratio,
    ]


def _lagged_dependency_matrix(X: np.ndarray) -> np.ndarray:
    filled = _fill_nan_by_feature(X)
    n_features = filled.shape[-1]
    previous_values = filled[:, :-1, :].reshape(-1, n_features)
    current_values = filled[:, 1:, :].reshape(-1, n_features)

    previous_values = np.apply_along_axis(_standardize_vector, 0, previous_values)
    current_values = np.apply_along_axis(_standardize_vector, 0, current_values)

    denom = max(1, previous_values.shape[0] - 1)
    weights = previous_values.T @ current_values / denom
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(weights, 0.0)
    return weights


def _select_action_outcome(
    X: np.ndarray,
    feature_names: Optional[List[str]],
    weights: np.ndarray,
) -> Tuple[int, int, str, str]:
    n_features = X.shape[-1]
    names = feature_names or [f"feature_{idx}" for idx in range(n_features)]
    lower_names = [name.lower() for name in names]

    outcome_keywords = ["load", "energy", "power", "consumption", "demand", "appliances"]
    action_keywords = ["price", "temperature", "solar", "wind", "hour", "month", "weather", "carbon", "humidity"]

    variances = np.nanvar(X, axis=(0, 1))
    outcome_candidates = [
        idx
        for idx, name in enumerate(lower_names)
        if any(keyword in name for keyword in outcome_keywords)
        and not any(time_key in name for time_key in ["sin", "cos", "hour", "month", "dow"])
    ]
    if outcome_candidates:
        outcome_idx = max(outcome_candidates, key=lambda idx: variances[idx])
    else:
        outcome_idx = int(np.nanargmax(variances)) if len(variances) else 0

    action_candidates = [
        idx
        for idx, name in enumerate(lower_names)
        if idx != outcome_idx and any(keyword in name for keyword in action_keywords)
    ]
    if action_candidates:
        action_idx = max(action_candidates, key=lambda idx: abs(weights[idx, outcome_idx]))
    else:
        candidate_scores = np.abs(weights[:, outcome_idx])
        candidate_scores[outcome_idx] = -np.inf
        action_idx = int(np.nanargmax(candidate_scores)) if n_features > 1 else outcome_idx

    return action_idx, outcome_idx, names[action_idx], names[outcome_idx]


def _threshold_graph(weights: np.ndarray) -> np.ndarray:
    abs_weights = np.abs(weights)
    non_zero = abs_weights[abs_weights > 1e-12]
    if non_zero.size == 0:
        return np.zeros_like(weights, dtype=bool)
    threshold = max(0.10, float(np.quantile(non_zero, 0.80)))
    graph = abs_weights >= threshold
    np.fill_diagonal(graph, False)
    return graph


def _path_strength(graph: np.ndarray, weights: np.ndarray, source: int, target: int) -> float:
    if source == target:
        return 0.0
    n_nodes = graph.shape[0]
    queue = [(source, 1.0, 0)]
    visited = {source}
    while queue:
        node, strength, depth = queue.pop(0)
        if depth >= n_nodes:
            continue
        neighbors = np.where(graph[node])[0]
        for neighbor in neighbors:
            edge_strength = abs(weights[node, neighbor])
            next_strength = min(strength, edge_strength)
            if neighbor == target:
                return float(next_strength / (depth + 1))
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, next_strength, depth + 1))
    return 0.0


def _partial_correlation(y: np.ndarray, x: np.ndarray, z: Optional[np.ndarray]) -> float:
    mask = np.isfinite(y) & np.isfinite(x)
    if z is not None:
        mask &= np.isfinite(z).all(axis=1)
    if mask.sum() < 5:
        return 0.0
    y = y[mask]
    x = x[mask]
    if z is None or z.shape[1] == 0:
        if np.std(y) < 1e-12 or np.std(x) < 1e-12:
            return 0.0
        return float(np.corrcoef(y, x)[0, 1])

    z = z[mask]
    design = np.column_stack([np.ones(z.shape[0]), z])
    y_res = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    x_res = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    if np.std(y_res) < 1e-12 or np.std(x_res) < 1e-12:
        return 0.0
    return float(np.corrcoef(y_res, x_res)[0, 1])


def _causal_and_confounding_features(
    X: np.ndarray,
    feature_names: Optional[List[str]],
) -> Tuple[List[float], List[float], Dict[str, object]]:
    weights = _lagged_dependency_matrix(X)
    graph = _threshold_graph(weights)
    n_features = X.shape[-1]
    action_idx, outcome_idx, action_name, outcome_name = _select_action_outcome(X, feature_names, weights)

    indegree = graph.sum(axis=0)
    outdegree = graph.sum(axis=1)
    edge_weights = np.abs(weights[graph])
    if edge_weights.size == 0:
        edge_weights = np.zeros(1, dtype=np.float64)

    density = graph.sum() / max(1, n_features * (n_features - 1))
    graph_asymmetry = np.mean(np.abs(weights - weights.T))
    path_strength = _path_strength(graph, weights, action_idx, outcome_idx)
    direct_weight = abs(weights[action_idx, outcome_idx])

    causal_features = [
        density,
        indegree.max(initial=0),
        indegree.mean() if indegree.size else 0.0,
        outdegree.max(initial=0),
        outdegree.mean() if outdegree.size else 0.0,
        edge_weights.mean(),
        edge_weights.max(initial=0),
        graph_asymmetry,
        path_strength,
        direct_weight,
    ]

    parents_action = set(np.where(graph[:, action_idx])[0])
    parents_outcome = set(np.where(graph[:, outcome_idx])[0])
    confounders = sorted((parents_action & parents_outcome) - {action_idx, outcome_idx})

    # Add strong common-association proxies when the thresholded graph is sparse.
    if not confounders:
        association_strength = np.abs(weights[:, action_idx]) * np.abs(weights[:, outcome_idx])
        association_strength[[action_idx, outcome_idx]] = 0.0
        proxy_candidates = np.where(association_strength >= np.quantile(association_strength, 0.90))[0]
        confounders = [int(idx) for idx in proxy_candidates if association_strength[idx] > 0]

    confounder_strengths = [
        abs(weights[idx, action_idx]) + abs(weights[idx, outcome_idx])
        for idx in confounders
    ]
    confounder_count = len(confounders)
    confounder_fraction = confounder_count / max(1, n_features - 2)
    mean_strength = float(np.mean(confounder_strengths)) if confounder_strengths else 0.0
    max_strength = float(np.max(confounder_strengths)) if confounder_strengths else 0.0

    flattened = _fill_nan_by_feature(X).reshape(-1, n_features)
    action_values = flattened[:, action_idx]
    outcome_values = flattened[:, outcome_idx]
    top_confounders = confounders[: min(5, confounder_count)]
    adjustment = flattened[:, top_confounders] if top_confounders else np.empty((flattened.shape[0], 0))
    unadjusted = _partial_correlation(outcome_values, action_values, None)
    adjusted = _partial_correlation(outcome_values, action_values, adjustment)
    bias_proxy = abs(unadjusted - adjusted) / (abs(unadjusted) + 1e-8)

    confounding_features = [
        confounder_count,
        confounder_fraction,
        mean_strength,
        max_strength,
        confounder_count,
        len(top_confounders),
        bias_proxy,
    ]

    metadata = {
        "action_feature_index": int(action_idx),
        "outcome_feature_index": int(outcome_idx),
        "action_feature_name": action_name,
        "outcome_feature_name": outcome_name,
        "ctcar_proxy_graph_edges": int(graph.sum()),
        "ctcar_proxy_confounders": confounder_count,
    }
    return causal_features, confounding_features, metadata


def extract_ctcar_features(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Extract the 80-dimensional C-TCAR representation from windows."""

    X = np.asarray(X, dtype=np.float64)
    stat = _statistical_features(X)
    temporal = _temporal_features(X)
    spectral = _spectral_features(X)
    causal, confounding, metadata = _causal_and_confounding_features(X, feature_names)

    values = stat + temporal + spectral + causal + confounding
    if len(values) != len(CTCAR_FEATURE_NAMES):
        raise RuntimeError(f"C-TCAR expected 80 features, got {len(values)}")

    feature_dict = {
        f"ctcar_{name}": _safe_float(value)
        for name, value in zip(CTCAR_FEATURE_NAMES, values)
    }
    return feature_dict, metadata


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def _prepare_dataset(
    preparator: DatasetPreparator,
    dataset_name: str,
    missing_rate: float,
    max_samples: Optional[int],
    window_stride: int,
    seed: int,
):
    dims = DATASET_DIMS[dataset_name]
    args = SimpleNamespace(
        dataset_name=dataset_name,
        missing_rate=missing_rate,
        n_steps=dims["n_steps"],
        n_features=dims["n_features"],
        max_samples=max_samples,
        window_stride=window_stride,
        random_seed=seed,
    )
    return preparator.prepare(args)


def build_ctcar_table(
    datasets: Iterable[str],
    missing_rates: Iterable[float],
    max_samples: Optional[int],
    window_stride: int,
    seed: int,
    cache_dir: str,
) -> pd.DataFrame:
    preparator = DatasetPreparator(cache_dir=cache_dir)
    rows = []
    for dataset_name in datasets:
        if dataset_name not in DATASET_DIMS:
            raise ValueError(f"Unknown dataset for C-TCAR extraction: {dataset_name}")
        for missing_rate in missing_rates:
            dataset = _prepare_dataset(
                preparator=preparator,
                dataset_name=dataset_name,
                missing_rate=float(missing_rate),
                max_samples=max_samples,
                window_stride=window_stride,
                seed=seed,
            )
            X = dataset.get("train_X_ori", dataset["train_X"])
            feature_names = dataset.get("feature_names")
            features, metadata = extract_ctcar_features(X, feature_names=feature_names)
            rows.append(
                {
                    "dataset": dataset_name,
                    "missing_rate": float(missing_rate),
                    "seed": seed,
                    "n_steps": int(dataset["n_steps"]),
                    "n_features": int(dataset["n_features"]),
                    **metadata,
                    **features,
                }
            )
            print(f"Extracted C-TCAR: {dataset_name} mr={missing_rate}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 80-dimensional C-TCAR features.")
    parser.add_argument(
        "--datasets",
        default="physionet_2012 appliances_energy household_power citylearn_zone5 opsd_germany",
        help="Whitespace/comma-separated dataset names",
    )
    parser.add_argument("--missing-rates", default="0.1 0.3 0.5", help="Whitespace/comma-separated missing rates")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="./datasets/")
    parser.add_argument("--out", default="output/imputation/mps/energy_benchmark/ctcar_features.csv")
    args = parser.parse_args()

    datasets = _parse_list(args.datasets)
    missing_rates = [float(item) for item in _parse_list(args.missing_rates)]
    table = build_ctcar_table(
        datasets=datasets,
        missing_rates=missing_rates,
        max_samples=args.max_samples,
        window_stride=args.window_stride,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"Wrote {len(table)} C-TCAR rows with {len(CTCAR_FEATURE_NAMES)} features to {out_path}")


if __name__ == "__main__":
    main()
