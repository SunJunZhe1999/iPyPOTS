#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025/07/26

🚀 Welcome to the Awesome Python Script 🚀

User: Messou Franck Junior Aboya
Email: messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University - IIST - (Tokyo, Japan)
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

# profiling.py
# Created by Franck Junior Aboya <mesabo18@gmail.com / messouaboya17@gmail.com>
# License: BSD-3-Clause

import os
import time
import csv
import torch
import psutil
from datetime import datetime


def measure_runtime_memory(
    model_forward_fn,
    *inputs,
    save_path="./output/imputation/profiling",
    prefix="model"
):
    """
    Measures runtime (sec) and memory usage (bytes) during model forward pass.
    Saves results incrementally into a CSV file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss

    # CUDA sync before timing
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device)
    else:
        mem_before = process.memory_info().rss

    start_time = time.time()
    _ = model_forward_fn(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end_time = time.time()

    runtime = round(end_time - start_time, 6)

    if device.type == "cuda":
        mem_used = torch.cuda.max_memory_allocated(device=device)
    else:
        mem_after = process.memory_info().rss
        mem_used = mem_after - mem_before

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": device.type,
        "runtime_seconds": runtime,
        "memory_bytes": int(mem_used),
        "memory_MB": round(mem_used / 1024 ** 2, 2),
    }

    # 🔄 Save to CSV (append mode)
    os.makedirs(save_path, exist_ok=True)
    csv_file = os.path.join(save_path, f"{prefix}_profiling.csv")
    write_header = not os.path.exists(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    print(f"📦 Profiling row appended to: {csv_file}")
    return result
