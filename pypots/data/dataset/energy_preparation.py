#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for preparing public energy-system datasets for imputation experiments."""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


UCI_APPLIANCES_URL = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
UCI_HOUSEHOLD_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
OPSD_60MIN_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
CITYLEARN_BASE_URL = "https://raw.githubusercontent.com/citylearn-project/CityLearn/v1.0.0/data/Climate_Zone_5"

CUSTOM_ENERGY_DATASET_NAMES = {
    "appliances_energy",
    "uci_appliances_energy",
    "household_power",
    "uci_household_power",
    "opsd_germany",
    "open_power_germany",
    "citylearn_zone5",
    "citylearn_climate_zone_5",
}


def prepare_custom_energy_dataset(
    name: str,
    cache_dir: str,
    rate: float,
    n_steps: int,
    window_stride: int = 1,
    max_samples: Optional[int] = None,
    random_state: int = 2025,
) -> Dict:
    """Load, window, split, and mask a supported public energy dataset."""

    normalized_name = name.lower()
    loaders: Dict[str, Callable[[Path], Tuple[pd.DataFrame, str]]] = {
        "appliances_energy": _load_appliances_energy,
        "uci_appliances_energy": _load_appliances_energy,
        "household_power": _load_household_power,
        "uci_household_power": _load_household_power,
        "opsd_germany": _load_opsd_germany,
        "open_power_germany": _load_opsd_germany,
        "citylearn_zone5": _load_citylearn_zone5,
        "citylearn_climate_zone_5": _load_citylearn_zone5,
    }
    if normalized_name not in loaders:
        raise ValueError(f"Unsupported custom energy dataset: {name}")

    dataset_dir = Path(cache_dir).expanduser().resolve() / normalized_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    frame, canonical_name = loaders[normalized_name](dataset_dir)
    values, feature_names = _clean_numeric_frame(frame)

    windows = _make_windows(values, n_steps=n_steps, stride=window_stride, max_samples=max_samples)
    windows = _standardize_windows(windows)
    train_X_ori, val_X_ori, test_X_ori = _split_windows(windows)

    rng = np.random.default_rng(random_state)
    train_X = train_X_ori.copy()
    val_X = _inject_mcar_missingness(val_X_ori, rate=rate, rng=rng)
    test_X = _inject_mcar_missingness(test_X_ori, rate=rate, rng=rng)

    return {
        "train_X": train_X.astype(np.float32),
        "val_X": val_X.astype(np.float32),
        "val_X_ori": val_X_ori.astype(np.float32),
        "test_X": test_X.astype(np.float32),
        "test_X_ori": test_X_ori.astype(np.float32),
        "n_steps": n_steps,
        "n_features": int(windows.shape[-1]),
        "feature_names": feature_names,
        "dataset_name": canonical_name,
        "missing_rate": rate,
        "window_stride": window_stride,
    }


def _load_appliances_energy(dataset_dir: Path) -> Tuple[pd.DataFrame, str]:
    zip_path = dataset_dir / "appliances_energy_prediction.zip"
    csv_path = dataset_dir / "energydata_complete.csv"
    _download_and_extract(UCI_APPLIANCES_URL, zip_path, dataset_dir, csv_path)

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df["hour_sin"] = np.sin(2 * np.pi * df["date"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["date"].dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)
    df = df.drop(columns=["date"])
    return df, "appliances_energy"


def _load_household_power(dataset_dir: Path) -> Tuple[pd.DataFrame, str]:
    zip_path = dataset_dir / "household_power.zip"
    txt_path = dataset_dir / "household_power_consumption.txt"
    _download_and_extract(UCI_HOUSEHOLD_URL, zip_path, dataset_dir, txt_path)

    df = pd.read_csv(txt_path, sep=";", na_values="?", low_memory=False)
    timestamp = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    df = df.drop(columns=["Date", "Time"])
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = timestamp
    df = df.loc[df.index.notna()].sort_index()
    df = df.resample("60min").mean()
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    return df.reset_index(drop=True), "household_power"


def _load_opsd_germany(dataset_dir: Path) -> Tuple[pd.DataFrame, str]:
    csv_path = dataset_dir / "time_series_60min_singleindex.csv"
    _download_file(OPSD_60MIN_URL, csv_path)

    desired_columns = [
        "DE_load_actual_entsoe_transparency",
        "DE_load_forecast_entsoe_transparency",
        "DE_solar_generation_actual",
        "DE_wind_generation_actual",
        "DE_wind_onshore_generation_actual",
        "DE_wind_offshore_generation_actual",
        "DE_price_day_ahead",
    ]
    header = pd.read_csv(csv_path, nrows=0).columns
    usecols = ["utc_timestamp"] + [col for col in desired_columns if col in header]
    df = pd.read_csv(csv_path, usecols=usecols, parse_dates=["utc_timestamp"])
    df["hour_sin"] = np.sin(2 * np.pi * df["utc_timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["utc_timestamp"].dt.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["utc_timestamp"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["utc_timestamp"].dt.month / 12)
    df = df.drop(columns=["utc_timestamp"])
    return df, "opsd_germany"


def _load_citylearn_zone5(dataset_dir: Path) -> Tuple[pd.DataFrame, str]:
    files = [f"Building_{i}.csv" for i in range(1, 10)] + [
        "weather_data.csv",
        "carbon_intensity.csv",
        "solar_generation_1kW.csv",
        "building_attributes.json",
    ]
    for file_name in files:
        _download_file(f"{CITYLEARN_BASE_URL}/{file_name}", dataset_dir / file_name)

    building_frames = [pd.read_csv(dataset_dir / f"Building_{i}.csv") for i in range(1, 10)]
    combined = pd.DataFrame(index=building_frames[0].index)
    sum_cols = {
        "district_equipment_power": ["Equipment Electric Power [kWh]", "non_shiftable_load"],
        "district_dhw_heating": ["DHW Heating [kWh]", "dhw_demand"],
        "district_cooling_load": ["Cooling Load [kWh]", "cooling_demand"],
        "district_heating_load": ["Heating Load [kWh]", "heating_demand"],
        "district_solar_generation": ["Solar Generation [W/kW]", "solar_generation"],
    }
    mean_cols = {
        "mean_indoor_temperature": ["Indoor Temperature [C]", "indoor_dry_bulb_temperature"],
        "mean_indoor_humidity": ["Indoor Relative Humidity [%]", "indoor_relative_humidity"],
        "mean_unmet_cooling_setpoint": [
            "Average Unmet Cooling Setpoint Difference [C]",
            "average_unmet_cooling_setpoint_difference",
        ],
        "mean_occupant_count": ["Occupant Count", "occupant_count"],
    }

    for output_col, aliases in sum_cols.items():
        available = [
            frame[col]
            for frame in building_frames
            for col in aliases
            if col in frame.columns
        ]
        if available:
            combined[output_col] = np.sum(available, axis=0)

    for output_col, aliases in mean_cols.items():
        available = [
            frame[col]
            for frame in building_frames
            for col in aliases
            if col in frame.columns
        ]
        if available:
            combined[output_col] = np.mean(available, axis=0)

    first = building_frames[0]
    calendar_cols = {
        "month": ["Month", "month"],
        "hour": ["Hour", "hour"],
        "day_type": ["Day Type", "day_type"],
        "daylight_savings_status": ["Daylight Savings Status", "daylight_savings_status"],
    }
    for output_col, aliases in calendar_cols.items():
        for col in aliases:
            if col in first.columns:
                combined[output_col] = first[col]
                break

    if "hour" in combined.columns:
        combined["hour_sin"] = np.sin(2 * np.pi * combined["hour"] / 24)
        combined["hour_cos"] = np.cos(2 * np.pi * combined["hour"] / 24)
    if "month" in combined.columns:
        combined["month_sin"] = np.sin(2 * np.pi * combined["month"] / 12)
        combined["month_cos"] = np.cos(2 * np.pi * combined["month"] / 12)

    weather = pd.read_csv(dataset_dir / "weather_data.csv")
    weather_cols = [
        "Outdoor Drybulb Temperature [C]",
        "Outdoor Relative Humidity [%]",
        "Diffuse Solar Radiation [W/m2]",
        "Direct Solar Radiation [W/m2]",
        "Wind Speed [m/s]",
    ]
    for col in weather_cols:
        if col in weather.columns:
            combined[col] = weather[col].to_numpy()[: len(combined)]

    carbon = pd.read_csv(dataset_dir / "carbon_intensity.csv")
    for col in carbon.columns:
        normalized_col = col.lower()
        if normalized_col.startswith("carbon") or "co2" in normalized_col:
            combined["carbon_intensity"] = carbon[col].to_numpy()[: len(combined)]
            break

    solar = pd.read_csv(dataset_dir / "solar_generation_1kW.csv")
    if solar.shape[1] > 0:
        combined["solar_generation_1kw_reference"] = solar.iloc[: len(combined), -1].to_numpy()

    return combined, "citylearn_zone5"


def _download_and_extract(url: str, zip_path: Path, extract_dir: Path, expected_file: Path) -> None:
    if not expected_file.exists():
        _download_file(url, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)


def _download_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    print(f"Downloading {url} -> {destination}")
    urlretrieve(url, tmp_path)
    os.replace(tmp_path, destination)


def _clean_numeric_frame(frame: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    numeric = frame.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.interpolate(limit_direction="both").ffill().bfill()
    numeric = numeric.dropna(axis=0, how="any")
    if numeric.empty:
        raise ValueError("No usable numeric data remained after cleaning.")
    return numeric.to_numpy(dtype=np.float32), list(numeric.columns)


def _make_windows(values: np.ndarray, n_steps: int, stride: int, max_samples: Optional[int]) -> np.ndarray:
    if values.shape[0] < n_steps:
        raise ValueError(f"Dataset has only {values.shape[0]} rows, less than n_steps={n_steps}.")
    total = (values.shape[0] - n_steps) // stride + 1
    starts = np.arange(total, dtype=np.int64) * stride
    if max_samples is not None and max_samples > 0 and len(starts) > max_samples:
        starts = starts[np.linspace(0, len(starts) - 1, max_samples, dtype=np.int64)]
    windows = np.stack([values[start : start + n_steps] for start in starts], axis=0)
    return windows


def _standardize_windows(windows: np.ndarray) -> np.ndarray:
    flat = windows.reshape(-1, windows.shape[-1])
    mean = np.nanmean(flat, axis=0, keepdims=True)
    std = np.nanstd(flat, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (windows - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)


def _split_windows(windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = windows.shape[0]
    train_end = max(1, int(n_samples * 0.7))
    val_end = max(train_end + 1, int(n_samples * 0.8))
    if val_end >= n_samples:
        val_end = n_samples - 1
    return windows[:train_end], windows[train_end:val_end], windows[val_end:]


def _inject_mcar_missingness(windows: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    masked = windows.copy()
    observed = np.isfinite(masked)
    missing = rng.random(masked.shape) < rate
    masked[observed & missing] = np.nan
    return masked


def _parse_dataset_list(raw: str) -> Iterable[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare custom energy datasets.")
    parser.add_argument("--datasets", default="appliances_energy,household_power,citylearn_zone5")
    parser.add_argument("--cache_dir", default="./datasets")
    parser.add_argument("--missing_rate", type=float, default=0.3)
    parser.add_argument("--n_steps", type=int, default=48)
    parser.add_argument("--window_stride", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--random_seed", type=int, default=2025)
    args = parser.parse_args()

    for dataset_name in _parse_dataset_list(args.datasets):
        dataset = prepare_custom_energy_dataset(
            dataset_name,
            cache_dir=args.cache_dir,
            rate=args.missing_rate,
            n_steps=args.n_steps,
            window_stride=args.window_stride,
            max_samples=args.max_samples,
            random_state=args.random_seed,
        )
        print(
            f"{dataset_name}: train={dataset['train_X'].shape}, val={dataset['val_X'].shape}, "
            f"test={dataset['test_X'].shape}, features={dataset['n_features']}"
        )


if __name__ == "__main__":
    main()
