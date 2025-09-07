#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025/07/25

🚀 LLMSAITS Training Pipeline 🚀

User: Messou Franck Junior Aboya
Email: messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University - IIST - (Tokyo, Japan)
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import numpy as np
from pypots.nn.functional import calc_mae, calc_mse, calc_rmse, calc_mre
from pypots.optim import Adam
from pypots.imputation.llmsaits import LLMSAITS


def train_and_evaluate_llmsaits(dataset, args):
    print("🚀 Starting LLMSAITS training pipeline...")

    # 1. Prepare datasets
    train_set = {"X": dataset["train_X"]}
    val_set = {
        "X": dataset["val_X"],
        "X_ori": dataset["val_X_ori"],
    }
    test_set = {"X": dataset["test_X"]}
    test_X_ori = np.nan_to_num(dataset["test_X_ori"])
    test_X_indicating_mask = np.isnan(dataset["test_X_ori"]) ^ np.isnan(dataset["test_X"])

    # 2. Initialize model
    model = LLMSAITS(
        n_steps=dataset["n_steps"],
        n_features=dataset["n_features"],
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        d_ffn=args.d_ffn,
        d_llm=args.d_llm,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        prompt_template=args.prompt_template,
        train_gpt_mlp=args.train_gpt_mlp,
        use_lora=args.use_lora,
        enable_profiling=args.enable_profiling,
        profiling_path=args.profiling_path,
        profiling_prefix=args.profiling_prefix,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        optimizer=Adam(lr=1e-3),
        device=args.device,
        saving_path=args.saving_path,
        model_saving_strategy="best",
        use_hann_window=args.use_hann_window,
        llm_model_type=args.llm_model_type,
    )

    # 3. Train
    model.fit(train_set=train_set, val_set=val_set)

    # 4. Predict
    results = model.predict(test_set)
    imputations = results["imputation"]

    # 5. Evaluate
    mae = calc_mae(imputations, test_X_ori, test_X_indicating_mask)
    mse = calc_mse(imputations, test_X_ori, test_X_indicating_mask)
    rmse = calc_rmse(imputations, test_X_ori, test_X_indicating_mask)
    mre = calc_mre(imputations, test_X_ori, test_X_indicating_mask)
    print(f"[LLMSAITS] Testing —— MAE: {mae:.4f}| MSE: {mse:.4f}| RMSE: {rmse:.4f}| MRE: {mre:.4f}| ")
    return mae, mse, rmse, mre