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

import numpy as np
from typing import Tuple
from pypots.optim import Adam
from pypots.imputation.moment import MOMENT
from pypots.nn.functional import calc_mae, calc_mse, calc_rmse, calc_mre


def train_and_evaluate_moment(dataset, args) -> Tuple[float, float, float, float]:
    """Train and evaluate MOMENT on the given dataset."""
    fast_dev = getattr(args, "fast_dev_run", False)
    subset_limit = getattr(args, "train_subset", None)
    if subset_limit is not None:
        subset_limit = max(1, int(subset_limit))
    if fast_dev and subset_limit is None:
        subset_limit = 32

    def _limit(arr, limit):
        if limit is None:
            return arr
        upper = min(len(arr), limit)
        return arr[:upper]

    train_X = dataset["train_X"]
    val_X = dataset["val_X"]
    val_X_ori = dataset["val_X_ori"]
    test_X = dataset["test_X"]
    test_X_ori_raw = dataset["test_X_ori"]

    if subset_limit is not None:
        train_X = _limit(train_X, subset_limit)
        val_X = _limit(val_X, subset_limit)
        val_X_ori = _limit(val_X_ori, subset_limit)
        test_X = _limit(test_X, subset_limit)
        test_X_ori_raw = _limit(test_X_ori_raw, subset_limit)

    train_set = {"X": train_X}
    val_set = {
        "X": val_X,
        "X_ori": val_X_ori,
    }
    test_set = {"X": test_X}

    test_X_indicating_mask = np.isnan(test_X_ori_raw) ^ np.isnan(test_X)
    test_X_ori = np.nan_to_num(test_X_ori_raw)

    epochs = getattr(args, "epochs", 1)
    patience = getattr(args, "patience", 1)
    batch_size = getattr(args, "batch_size", 16)

    train_size = len(train_X)
    batch_size = max(1, min(batch_size, train_size))

    if fast_dev:
        epochs = max(1, min(epochs, 1))
        patience = max(1, min(patience, epochs))
        # Keep the batch_size small to reduce per-epoch cost
        batch_size = max(1, min(batch_size, min(train_size, 8)))
        print(f"[MOMENT] Fast dev run enabled: subset={train_size}, epochs={epochs}, batch_size={batch_size}")

    if subset_limit is not None and not fast_dev:
        batch_size = max(1, min(batch_size, train_size))

    args.epochs = epochs
    args.patience = patience
    args.batch_size = batch_size

    # 2. Initialize model（加兜底，避免 n_steps/n_features 缺失导致 KeyError）
    model = MOMENT(
        n_steps=dataset.get("n_steps", getattr(args, "n_steps", 48)),
        n_features=dataset.get("n_features", getattr(args, "n_features", 1)),
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        transformer_backbone=args.transformer_backbone,
        transformer_type=args.transformer_type,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ffn=args.d_ffn,
        d_model=args.d_model,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        finetuning_mode=args.finetuning_mode,
        revin_affine=args.revin_affine,
        add_positional_embedding=args.add_positional_embedding,
        value_embedding_bias=args.value_embedding_bias,
        orth_gain=args.orth_gain,
        mask_ratio=args.mask_ratio,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        optimizer=Adam(lr=1e-3),
        device=args.device,
        saving_path=args.saving_path,
        model_saving_strategy=args.model_saving_strategy,
        verbose=args.verbose,
    )

    device_str = str(getattr(args, "device", "cpu")).lower()
    if "cuda" not in device_str:
        backbone = getattr(model, "model", None)
        if backbone is not None:
            backbone = getattr(backbone, "backbone", None)
        encoder = getattr(backbone, "encoder", None) if backbone is not None else None
        if encoder is not None and hasattr(encoder, "gradient_checkpointing_disable"):
            try:
                encoder.gradient_checkpointing_disable()
                if hasattr(backbone, "configs"):
                    backbone.configs.enable_gradient_checkpointing = False
                print("[MOMENT] Gradient checkpointing disabled on CPU for faster runs.")
            except Exception as exc:
                print(f"[MOMENT] Warning: could not disable gradient checkpointing ({exc})")

    # 3. Fit model
    model.fit(train_set=train_set, val_set=val_set)

    # 4. Predict
    results = model.predict(test_set)
    imputations = results["imputation"]

    # 5. Evaluate
    mae = calc_mae(imputations, test_X_ori, test_X_indicating_mask)
    mse = calc_mse(imputations, test_X_ori, test_X_indicating_mask)
    rmse = calc_rmse(imputations, test_X_ori, test_X_indicating_mask)
    mre = calc_mre(imputations, test_X_ori, test_X_indicating_mask)

    print(f"[MOMENT] Testing —— MAE: {mae:.4f}| MSE: {mse:.4f}| RMSE: {rmse:.4f}| MRE: {mre:.4f}| ")
    return mae, mse, rmse, mre
