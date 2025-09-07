"""
The core wrapper assembles the submodules of LLM4IMP imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Franck Junior Aboya <mesabo18@gmail.com / messouaboya17@gmail.com>
# License: BSD-3-Clause

import torch
from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.llm4imp.backbone import BackboneLLM4IMP


class _LLM4IMP(ModelCore):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        patch_size: int,
        patch_stride: int,
        d_model: int,
        d_ffn: int,
        d_llm: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        prompt_template: str,
        training_loss: Criterion,
        validation_metric: Criterion,
        train_gpt_mlp: bool = False,
        use_lora: bool = False,
        enable_profiling: bool = False,
        profiling_path: str = "./output/imputation/profiling",
        profiling_prefix: str = "backbone_llm4imp",
        use_hann_window: bool = False,
        llm_model_type: str = "gpt2",
        use_prompt: bool = True,      
        use_reprogramming: bool = True,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.training_loss = training_loss
        if validation_metric.__class__.__name__ == "Criterion":
            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.backbone = BackboneLLM4IMP(
            n_steps=n_steps,
            n_features=n_features,
            patch_size=patch_size,
            patch_stride=patch_stride,
            d_model=d_model,
            d_ffn=d_ffn,
            d_llm=d_llm,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            prompt_template=prompt_template,
            train_gpt_mlp=train_gpt_mlp,
            use_lora=use_lora,
            enable_profiling=enable_profiling,
            profiling_prefix=profiling_prefix,
            profiling_path=profiling_path,
            use_hann_window=use_hann_window,
            llm_model_type=llm_model_type,
            use_prompt=use_prompt,
            use_reprogramming=use_reprogramming,
        )

    def forward(
        self,
        inputs: dict,
        calc_criterion: bool = False,
    ) -> dict:
        X = inputs["X"]  # [B, T, D]
        missing_mask = inputs["missing_mask"]  # [B, T, D]

        reconstruction = self.backbone(X, missing_mask)  # [B, T, D]
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction

        results = {
            "imputation": imputed_data,
            "reconstruction": reconstruction,
        }

        if calc_criterion:
            X_ori = inputs["X_ori"]
            indicating_mask = inputs["indicating_mask"]
            if self.training:
                results["loss"] = self.training_loss(reconstruction, X_ori, indicating_mask)
            else:
                results["metric"] = self.validation_metric(reconstruction, X_ori, indicating_mask)

        return results