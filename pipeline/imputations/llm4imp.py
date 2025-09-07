#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025/07/21

🚀 Welcome to the Awesome Python Script 🚀

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
from pypots.imputation.llm4imp import LLM4IMP


def train_and_evaluate_llm4imp(dataset, args):
    # 1. Assemble train/val/test datasets
    dataset_for_IMPU_training = {
        "X": dataset["train_X"],
    }
    dataset_for_IMPU_validating = {
        "X": dataset["val_X"],
        "X_ori": dataset["val_X_ori"],
    }
    dataset_for_IMPU_testing = {
        "X": dataset["test_X"],
    }

    test_X_ori = np.nan_to_num(dataset["test_X_ori"])
    test_X_indicating_mask = np.isnan(dataset["test_X_ori"]) ^ np.isnan(dataset["test_X"])

    # 2. Initialize model
    model = LLM4IMP(
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
        use_prompt=args.use_prompt,
        use_reprogramming=args.use_reprogramming,
    )

    # 3. Train
    model.fit(
        train_set=dataset_for_IMPU_training,
        val_set=dataset_for_IMPU_validating,
    )

    # 4. Test
    results = model.predict(dataset_for_IMPU_testing)
    imputation = results["imputation"]

    # 5. Evaluate
    mae = calc_mae(imputation, test_X_ori, test_X_indicating_mask)
    mse = calc_mse(imputation, test_X_ori, test_X_indicating_mask)
    rmse = calc_rmse(imputation, test_X_ori, test_X_indicating_mask)
    mre = calc_mre(imputation, test_X_ori, test_X_indicating_mask)
    print(f"[LLM4IMP] Testing —— MAE: {mae:.4f}| MSE: {mse:.4f}| RMSE: {rmse:.4f}| MRE: {mre:.4f}| ")
    return mae, mse, rmse, mre
