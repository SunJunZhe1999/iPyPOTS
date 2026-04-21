#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025/07/24

🚀 Welcome to the Awesome Python Script 🚀

User: Messou Franck Junior Aboya
Email: messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University - IIST - (Tokyo, Japan)
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import os
from typing import Dict
import benchpots
import numpy as np
import tsdb

from .energy_preparation import CUSTOM_ENERGY_DATASET_NAMES, prepare_custom_energy_dataset


class DatasetPreparator:
    def __init__(self, cache_dir: str = "./datasets/"):
        self.base_cache_dir = os.path.abspath(cache_dir)
        if not os.path.exists(os.path.join(self.base_cache_dir, "physionet_2012")):
            try:
                tsdb.migrate_cache(self.base_cache_dir)
            except Exception as e:
                print(f"⚠️ TSDB migration skipped due to error: {e}")
        else:
            print(f"ℹ️ TSDB migration not needed, directory exists: {self.base_cache_dir}")

    def prepare(self, args) -> Dict:
        """
        Prepare and return a preprocessed dataset for imputation.

        Args:
            args: argparse.Namespace with at least:
                  - args.dataset_name
                  - args.missing_rate
                  - args.n_steps (required by ETT)

        Returns:
            Dict: Dataset dictionary with keys like 'train', 'val', 'test'.
        """
        name = args.dataset_name.lower()
        dataset_name = args.dataset_name
        rate = args.missing_rate
        n_steps = getattr(args, "n_steps", 48)
        window_stride = getattr(args, "window_stride", 1)
        max_samples = getattr(args, "max_samples", None)
        random_state = getattr(args, "random_seed", 2025)

        # Construct rate-specific path
        rate_cache_dir = os.path.join(self.base_cache_dir, f"rate_{rate}")
        os.makedirs(rate_cache_dir, exist_ok=True)

        if name in ["physionet", "physionet_2012"]:
            dataset = benchpots.datasets.preprocess_physionet2012(
                subset="set-a", rate=rate, data_path=rate_cache_dir
            )

        elif name in ["italy", "italy_air_quality"]:
            dataset = benchpots.datasets.preprocess_italy_air_quality(
                rate=rate, n_steps=n_steps, data_path=rate_cache_dir
            )

        elif name in ["beijing_multisite_air_quality", "beijing", "beijing_air_quality"]:
            dataset = benchpots.datasets.preprocess_beijing_air_quality(
                rate=rate, n_steps=n_steps, data_path=rate_cache_dir
            )

        elif name in ["etth1", "etth2", "ettm1", "ettm2"]:
            file_map = {
                "etth1": "ETTh1.csv",
                "etth2": "ETTh2.csv",
                "ettm1": "ETTm1.csv",
                "ettm2": "ETTm2.csv",
            }
            file_name = file_map[name]
            subset = file_name.removesuffix(".csv")
            dataset = benchpots.datasets.preprocess_ett(
                data_path=os.path.join(rate_cache_dir, "ETT"),
                file_name=file_name,
                subset=subset,
                n_steps=n_steps,
                rate=rate,
            )

        elif name in ["pems", "pems_traffic"]:
            dataset = benchpots.datasets.preprocess_pems_traffic(
                rate=rate, data_path=rate_cache_dir,  n_steps=n_steps,
            )

        elif name in ["solar", "solar_alabama"]:
            dataset = benchpots.datasets.preprocess_solar_alabama(
                rate=rate, data_path=rate_cache_dir,  n_steps=n_steps,
            )
        elif name in ["eld", "electricity_load_diagrams"]:
            dataset = benchpots.datasets.preprocess_electricity_load_diagrams(
                rate=rate, data_path=rate_cache_dir,  n_steps=n_steps,
            )
        
        elif "ucr_uea_" in name:
            dataset = benchpots.datasets.preprocess_ucr_uea_datasets(
                rate=rate, data_path=rate_cache_dir,  n_steps=n_steps,
                dataset_name=dataset_name,
            )

        elif name in CUSTOM_ENERGY_DATASET_NAMES:
            dataset = prepare_custom_energy_dataset(
                name=name,
                cache_dir=self.base_cache_dir,
                rate=rate,
                n_steps=n_steps,
                window_stride=window_stride,
                max_samples=max_samples,
                random_state=random_state,
            )

        elif tsdb.has(name):
            print(f"📥 Downloading raw dataset '{name}' via TSDB to {rate_cache_dir}")
            tsdb.download_and_extract(name, rate_cache_dir)
            raise NotImplementedError(f"⚠️ Dataset '{name}' is available but preprocessing is not yet implemented.")

        else:
            raise ValueError(f"❌ Unknown or unsupported dataset: {name}")

        dataset = self._limit_samples(dataset, max_samples, random_state)

        print(f"✅ Dataset '{name}' with missing rate {rate} loaded at: {rate_cache_dir}")
        return dataset

    @staticmethod
    def _limit_samples(dataset: Dict, max_samples, random_state: int) -> Dict:
        if max_samples is None:
            return dataset

        max_samples = int(max_samples)
        if max_samples <= 0:
            return dataset

        split_limits = {
            "train": max(1, int(max_samples * 0.7)),
            "val": max(1, int(max_samples * 0.1)),
            "test": max(1, max_samples - int(max_samples * 0.7) - int(max_samples * 0.1)),
        }
        rng = np.random.default_rng(random_state)

        for split, limit in split_limits.items():
            x_key = f"{split}_X"
            if x_key not in dataset:
                continue
            n_samples = dataset[x_key].shape[0]
            if n_samples <= limit:
                continue

            indices = np.sort(rng.choice(n_samples, size=limit, replace=False))
            for key, value in list(dataset.items()):
                if not key.startswith(f"{split}_"):
                    continue
                if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == n_samples:
                    dataset[key] = value[indices]

        return dataset
