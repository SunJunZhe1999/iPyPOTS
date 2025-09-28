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
import shutil
from typing import Dict
import benchpots
import tsdb
import numpy as np


class DatasetPreparator:
    _PHYSIONET_REQUIRED_ENTRIES = (
        "set-a",
        "set-b",
        "set-c",
        "Outcomes-a.txt",
        "Outcomes-b.txt",
        "Outcomes-c.txt",
    )

    def __init__(self, cache_dir: str = "./datasets/"):
        self.base_cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.base_cache_dir, exist_ok=True)

        if not os.path.exists(os.path.join(self.base_cache_dir, "physionet_2012")):
            try:
                tsdb.migrate_cache(self.base_cache_dir)
            except Exception as e:
                print(f"⚠️ TSDB migration skipped due to error: {e}")
        else:
            print(f"ℹ️ TSDB migration not needed, directory exists: {self.base_cache_dir}")

    def _ensure_physionet_raw_data(self) -> None:
        physionet_dir = os.path.join(self.base_cache_dir, "physionet_2012")
        if not os.path.exists(physionet_dir):
            missing_entries = list(self._PHYSIONET_REQUIRED_ENTRIES)
        else:
            missing_entries = [
                entry
                for entry in self._PHYSIONET_REQUIRED_ENTRIES
                if not os.path.exists(os.path.join(physionet_dir, entry))
            ]

        if not missing_entries:
            return

        missing_str = ", ".join(missing_entries)
        print(
            "⚠️ Detected incomplete PhysioNet 2012 raw data:"
            f" {missing_str}. Refreshing via TSDB..."
        )
        shutil.rmtree(physionet_dir, ignore_errors=True)

        try:
            tsdb.download_and_extract("physionet_2012", physionet_dir)
        except Exception as error:
            raise RuntimeError(
                "❌ Auto refresh of PhysioNet 2012 failed. "
                "Please ensure network access and rerun "
                "`tsdb.load('physionet_2012', use_cache=False)` manually."
            ) from error

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

        # Construct rate-specific path
        rate_cache_dir = os.path.join(self.base_cache_dir, f"rate_{rate}")
        os.makedirs(rate_cache_dir, exist_ok=True)

        if name in ["physionet", "physionet_2012"]:
            self._ensure_physionet_raw_data()
            dataset = benchpots.datasets.preprocess_physionet2012(
                subset="set-a", rate=rate, data_path=rate_cache_dir
            )

        elif name in ["italy", "italy_air_quality"]:
            dataset = benchpots.datasets.preprocess_italy_air_quality(
                rate=rate, n_steps=n_steps, data_path=rate_cache_dir
            )

        elif name in ["beijing_multisite_air_quality"]:
            dataset = benchpots.datasets.preprocess_beijing_multisite_air_quality(
                rate=rate, n_steps=n_steps, data_path=rate_cache_dir
            )

        elif name in ["air_quality"]:
            dataset = benchpots.datasets.preprocess_air_quality(
                rate=rate, n_steps=n_steps, data_path=rate_cache_dir
            )

        elif name in ["solar", "solar_alabama"]:
            dataset = benchpots.datasets.preprocess_solar_alabama(
                rate=rate, n_steps=n_steps, data_path=rate_cache_dir
            )

        elif name in ["eld", "electricity_load_diagrams", "electricity"]:
            dataset = benchpots.datasets.preprocess_electricity_load_diagrams(
                rate=rate, n_steps=n_steps, data_path=rate_cache_dir
            )

        elif name in ["pems", "pems_traffic"]:
            dataset = benchpots.datasets.preprocess_pems_traffic(
                rate=rate, data_path=rate_cache_dir, n_steps=n_steps
            )

        elif name in ["ett", "etth1", "etth2", "ettm1", "ettm2"]:
            file_map = {
                "etth1": "ETTh1.csv",
                "etth2": "ETTh2.csv",
                "ettm1": "ETTm1.csv",
                "ettm2": "ETTm2.csv",
            }
            file_name = file_map.get(name, "ETTh1.csv")
            # Python 3.8 兼容：避免使用 str.removesuffix
            subset = file_name[:-4] if file_name.endswith(".csv") else file_name
            dataset = benchpots.datasets.preprocess_ett(
                data_path=os.path.join(rate_cache_dir, "ETT"),
                file_name=file_name,
                subset=subset,
                n_steps=n_steps,
                rate=rate,
            )

        elif name.startswith("ucr_uea_"):
            dataset = benchpots.datasets.preprocess_ucr_uea_datasets(
                rate=rate, data_path=rate_cache_dir, n_steps=n_steps,
                dataset_name=dataset_name,
            )

        elif tsdb.has(name):
            print(f"📥 Downloading raw dataset '{name}' via TSDB to {rate_cache_dir}")
            tsdb.download_and_extract(name, rate_cache_dir)
            raise NotImplementedError(
                f"⚠️ Dataset '{name}' is available but preprocessing is not yet implemented."
            )

        else:
            raise ValueError(f"❌ Unknown or unsupported dataset: {name}")

        print(f"✅ Dataset '{name}' with missing rate {rate} loaded at: {rate_cache_dir}")

        # === Add dataset dimensions for downstream pipelines ===
        try:
            train_X = dataset.get('train_X', None)
            if train_X is not None:
                if not hasattr(train_X, 'shape'):
                    train_X = np.asarray(train_X)
                if getattr(train_X, 'ndim', 0) >= 3:
                    n_steps = int(train_X.shape[1])
                    n_features = int(train_X.shape[2])
                elif getattr(train_X, 'ndim', 0) == 2:
                    # Ambiguous (N, T) case: fall back to args
                    n_steps = int(getattr(args, 'n_steps', 48))
                    n_features = int(getattr(args, 'n_features', 1))
                else:
                    n_steps = int(getattr(args, 'n_steps', 48))
                    n_features = int(getattr(args, 'n_features', 1))
            else:
                n_steps = int(getattr(args, 'n_steps', 48))
                n_features = int(getattr(args, 'n_features', 1))
            dataset['n_steps'] = n_steps
            dataset['n_features'] = n_features
        except Exception as _e:
            print(f"⚠️ Could not determine dataset dims automatically: {_e}")
        return dataset