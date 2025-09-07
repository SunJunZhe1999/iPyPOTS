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

import os
import json

from argument_parser import get_args
from pipeline.imputations.llm4imp import train_and_evaluate_llm4imp
from pipeline.imputations.saits import train_and_evaluate_saits
from pipeline.imputations.timellm import train_and_evaluate_timellm
from pipeline.imputations.moment import train_and_evaluate_moment
from pipeline.imputations.tefn import train_and_evaluate_tefn
from pipeline.imputations.tslanet import train_and_evaluate_tslanet
from pipeline.imputations.gpt4ts import train_and_evaluate_gpt4ts
from pipeline.imputations.llmsaits import train_and_evaluate_llmsaits
from pipeline.imputations.timemixerpp import train_and_evaluate_timemixerpp
from pipeline.imputations.moment import train_and_evaluate_moment

from pypots.data.dataset.load_prepare_dataset import DatasetPreparator
from pypots.utils.random import set_random_seed

MODEL_PIPELINES = {
    # LLM (Large Language Model)
    "llm4imp": train_and_evaluate_llm4imp,
    "llmsaits": train_and_evaluate_llmsaits,
    "timellm": train_and_evaluate_timellm,
    "gpt4ts": train_and_evaluate_gpt4ts,
    # TSFM (Time-Series Foundation Model)
    "tefn": train_and_evaluate_tefn,
    
    # NN (Neural Networks)
    "tslanet": train_and_evaluate_tslanet,
    "saits": train_and_evaluate_saits,
    "timemixerpp": train_and_evaluate_timemixerpp,
}



def main(args):
    print(f"🚀 Running imputation pipeline for model: {args.model}")
    dataset = DatasetPreparator().prepare(args)

    model_name = args.model.lower()
    if model_name not in MODEL_PIPELINES:
        raise ValueError(f"❌ Unknown model: {args.model}")

    mae, mse, rmse, mre = MODEL_PIPELINES[model_name](dataset, args)

    # ✅ Save metrics
    metrics_dir = os.path.join(args.saving_path, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, f"{model_name}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(
            {"MAE": mae, "MSE": mse, "RMSE": rmse, "MRE": mre},
            f,
            indent=4
        )
    print(f"📊 Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    args = get_args()
    print(f"Use Train GPT: {args.train_gpt_mlp}")
    print(f"Use Lora: {args.use_lora}")
    print(f"Use Profiling: {args.enable_profiling}")
    set_random_seed(2025)
    main(args)