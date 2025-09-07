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

import argparse
from distutils.util import strtobool


def get_args():
    parser = argparse.ArgumentParser(description="Run imputation models via PyPOTS pipeline")

    # General arguments
    parser.add_argument("--model", type=str, default="llm4imp",
                    help="Model name: llm4imp | saits | timellm | tslanet | moment | tefn | gpt4ts | timemixerpp")
    parser.add_argument("--missing_rate", type=float, default=0.1, help="Artificial missing rate in dataset")
    parser.add_argument("--saving_path", type=str, default="output/imputation", help="Directory to save results")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cpu', 'cuda', or None (auto)")
    parser.add_argument("--dataset_name", type=str, default="physionet_2012",
                        help="Name of the dataset to use: physionet_2012, air_quality, pems_traffic, etth1, etth2,"
                             " ettm1, ettm2, italy_air_quality, beijing_multisite_air_quality")

    # Shared model hyperparameters
    parser.add_argument("--n_steps", type=int, default=48, help="Number of time steps")
    parser.add_argument("--n_features", type=int, default=35, help="Number of input features")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--d_model", type=int, default=64, help="Model hidden size")
    parser.add_argument("--d_ffn", type=int, default=128, help="Feed-forward network size")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # LLM-based models
    parser.add_argument("--patch_size", type=int, default=12, help="Patch length for patch embedding")
    parser.add_argument("--patch_stride", type=int, default=4, help="Patch stride")
    parser.add_argument("--d_llm", type=int, default=768, help="LLM hidden dimension (GPT2: 768, LLaMA: 4096)")
    parser.add_argument("--prompt_template", type=str,
                        default="Impute missing values at time steps where mask=0",
                        help="Instruction prompt template")
    parser.add_argument("--llm_model_type", type=str, default="BERT", help="LLM model type: GPT2 | LLaMA | BERT | ...")
    parser.add_argument("--train_gpt_mlp", type=lambda x: bool(strtobool(x)), default=False,
                        help="Whether to train the MLP layers in GPT2.")
    parser.add_argument("--use_lora", type=lambda x: bool(strtobool(x)), default=False,
                        help="Enable LoRA adaptation for GPT2.")
    parser.add_argument("--enable_profiling", type=lambda x: bool(strtobool(x)), default=False,
                        help="Enable runtime/memory profiling.")
    parser.add_argument("--profiling_path", type=str, default="./output/imputation/profiling",
                        help="Path to save profiling results.")
    parser.add_argument("--profiling_prefix", type=str, default="backbone_llm4imp",
                        help="Prefix for profiling output files.")
    parser.add_argument("--use_hann_window",type=lambda x: bool(strtobool(x)), default=True,
                        help="Apply Hann window before FFT in prompt builder.")
    parser.add_argument("--use_prompt", type=lambda x: bool(strtobool(x)), default=True,
                        help="Enable use of prompt embeddings in LLM4IMP.")
    parser.add_argument("--use_reprogramming", type=lambda x: bool(strtobool(x)), default=True,
                        help="Enable use of reprogramming layer in LLM4IMP.")

    # SAITS-specific
    parser.add_argument("--d_k", type=int, default=16, help="Key/query dimension for SAITS")
    parser.add_argument("--d_v", type=int, default=16, help="Value dimension for SAITS")
    parser.add_argument("--attn_dropout", type=float, default=0.1, help="Attention dropout for SAITS")
    parser.add_argument("--diagonal_attention_mask", action="store_true", help="Use diagonal attention mask in SAITS")

    # TSLANet-specific
    parser.add_argument("--d_embedding", type=int, default=64, help="Embedding dimension for TSLANet")
    parser.add_argument("--mask_ratio", type=float, default=0.15, help="Masking ratio for TSLANet")

    # MOMENT-specific
    parser.add_argument("--transformer_backbone", type=str, default="t5-small",
                        help="Backbone of transformer: t5-small | t5-base | flan-t5-base | ...")
    parser.add_argument("--transformer_type", type=str, default="encoder_decoder",
                        help="Transformer type: encoder_only | decoder_only | encoder_decoder")
    parser.add_argument("--head_dropout", type=float, default=0.0, help="Dropout rate for MOMENT head")
    parser.add_argument("--finetuning_mode", type=str, default="linear-probing",
                        help="Finetuning mode: linear-probing | end-to-end | zero-shot")
    parser.add_argument("--revin_affine", action="store_true", help="Enable RevIn affine transformation")
    parser.add_argument("--add_positional_embedding", action="store_true", help="Enable positional embedding")
    parser.add_argument("--value_embedding_bias", action="store_true", help="Enable bias in value embedding")
    parser.add_argument("--orth_gain", type=float, default=1.0, help="Gain for orthogonal initialization")

    # TEFN-specific
    parser.add_argument("--n_fod", type=int, default=16, help="Number of FODs in TEFN")
    parser.add_argument("--apply_nonstationary_norm", action="store_true",
                        help="Apply nonstationary norm (TEFN)")
    parser.add_argument("--ORT_weight", type=float, default=1.0, help="Weight for ORT loss (TEFN)")
    parser.add_argument("--MIT_weight", type=float, default=1.0, help="Weight for MIT loss (TEFN)")
    
        # TimeMixerPP-specific
    parser.add_argument("--top_k", type=int, default=5, help="Number of top-k frequencies to use")
    parser.add_argument("--n_kernels", type=int, default=6, help="Number of Inception kernels")
    parser.add_argument("--channel_mixing", type=lambda x: bool(strtobool(x)), default=True,
                        help="Enable channel mixing")
    parser.add_argument("--channel_independence", type=lambda x: bool(strtobool(x)), default=True,
                        help="Enable channel independence")
    parser.add_argument("--downsampling_layers", type=int, default=3, help="Number of downsampling layers")
    parser.add_argument("--downsampling_window", type=int, default=2, help="Window size for downsampling")

    # Training-related
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")

    # Optional training behavior
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use (default: adam)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--model_saving_strategy", type=str, default="best",
                        help="Model saving strategy: None | best | better | all")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose training output")

    return parser.parse_args()
