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
import tsdb


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

        elif tsdb.has(name):
            print(f"📥 Downloading raw dataset '{name}' via TSDB to {rate_cache_dir}")
            tsdb.download_and_extract(name, rate_cache_dir)
            raise NotImplementedError(f"⚠️ Dataset '{name}' is available but preprocessing is not yet implemented.")

        else:
            raise ValueError(f"❌ Unknown or unsupported dataset: {name}")

        print(f"✅ Dataset '{name}' with missing rate {rate} loaded at: {rate_cache_dir}")
        return dataset
