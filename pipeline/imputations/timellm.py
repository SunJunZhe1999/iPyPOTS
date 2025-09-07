#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025/07/23

🚀 Welcome to the Awesome Python Script 🚀

User: Messou Franck Junior Aboya
Email: messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University - IIST - (Tokyo, Japan)
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

# pipeline/imputations/timellm.py

import numpy as np
from pypots.nn.functional import calc_mae, calc_mse, calc_rmse, calc_mre
from pypots.optim import Adam
from pypots.imputation import TimeLLM


def train_and_evaluate_timellm(dataset: dict, args):
    # 1. Assemble train/val/test datasets
    dataset_for_IMPU_training = {"X": dataset["train_X"]}
    dataset_for_IMPU_validating = {
        "X": dataset["val_X"],
        "X_ori": dataset["val_X_ori"],
    }
    dataset_for_IMPU_testing = {"X": dataset["test_X"]}

    test_X_ori = np.nan_to_num(dataset["test_X_ori"])
    test_X_indicating_mask = (
            np.isnan(dataset["test_X_ori"]) ^ np.isnan(dataset["test_X"])
    )

    # 2. Initialize TimeLLM model
    timellm = TimeLLM(
        n_steps=dataset["n_steps"],
        n_features=dataset["n_features"],
        llm_model_type="GPT2",
        n_layers=args.n_layers,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        d_llm=args.d_llm,
        d_model=args.d_model,
        d_ffn=args.d_ffn,
        n_heads=args.n_heads,
        dropout=args.dropout,
        ORT_weight=args.ORT_weight,
        MIT_weight=args.MIT_weight,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        domain_prompt_content="PhysioNet ICU",
        optimizer=Adam(lr=1e-3),
        num_workers=0,
        device=args.device,
        saving_path=args.saving_path,
        model_saving_strategy="best",
    )

    # 3. Train the model
    timellm.fit(
        train_set=dataset_for_IMPU_training,
        val_set=dataset_for_IMPU_validating,
    )

    # 4. Test the model
    timellm_results = timellm.predict(dataset_for_IMPU_testing)
    imputations = timellm_results["imputation"]

    # 5. Evaluate
    mae = calc_mae(imputations, test_X_ori, test_X_indicating_mask)
    mse = calc_mse(imputations, test_X_ori, test_X_indicating_mask)
    rmse = calc_rmse(imputations, test_X_ori, test_X_indicating_mask)
    mre = calc_mre(imputations, test_X_ori, test_X_indicating_mask)
    print(f"[LLM4IMP] Testing —— MAE: {mae:.4f}| MSE: {mse:.4f}| RMSE: {rmse:.4f}| MRE: {mre:.4f}| ")
    return mae, mse, rmse, mre
