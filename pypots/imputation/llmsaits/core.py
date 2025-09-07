"""
The core wrapper assembles the submodules of LLM4IMP imputation model
and takes over the forward progress of the algorithm.
"""

# Created by Franck Junior Aboya <mesabo18@gmail.com / messouaboya17@gmail.com>
# License: BSD-3-Clause

# pypots/imputation/llmsaits/core.py

import torch
from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.llmsaits.backbone import BackboneLLMSAITS


class _LLMSAITS(ModelCore):
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
        profiling_prefix: str = "backbone_llmsaits",
        use_hann_window: bool = False,
        llm_model_type: str = "gpt2",
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.training_loss = training_loss
        self.validation_metric = (
            self.training_loss if validation_metric.__class__.__name__ == "Criterion" else validation_metric
        )

        self.backbone = BackboneLLMSAITS(
            n_steps=n_steps,
            n_features=n_features,
            patch_size=patch_size,
            patch_stride=patch_stride,
            d_model=d_model,
            d_ffn=d_ffn,
            d_llm=d_llm,
            n_heads=n_heads,
            dropout=dropout,
            prompt_template=prompt_template,
            train_gpt_mlp=train_gpt_mlp,
            use_lora=use_lora,
            enable_profiling=enable_profiling,
            profiling_path=profiling_path,
            profiling_prefix=profiling_prefix,
            use_hann_window=use_hann_window,
        )

    def forward(
        self, inputs: dict, calc_criterion: bool = False, diagonal_attention_mask: bool = True  # ✅ Add this
    ) -> dict:
        X = inputs["X"]
        missing_mask = inputs["missing_mask"]

        # 🔁 You can pass diagonal_attention_mask to backbone if supported
        recon_1, recon_2, recon_final = self.backbone(X, missing_mask)  # or add it here if needed

        imputed_data = missing_mask * X + (1 - missing_mask) * recon_final

        results = {
            "X_tilde_1": recon_1,
            "X_tilde_2": recon_2,
            "X_tilde_3": recon_final,
            "reconstruction": recon_final,
            "imputation": imputed_data,
        }

        if calc_criterion:
            X_ori = inputs["X_ori"]
            indicating_mask = inputs["indicating_mask"]
            if self.training:
                ORT = self.training_loss(recon_1, X, missing_mask)
                ORT += self.training_loss(recon_2, X, missing_mask)
                ORT += self.training_loss(recon_final, X, missing_mask)
                ORT /= 3.0
                MIT = self.training_loss(recon_final, X_ori, indicating_mask)
                results["ORT_loss"] = ORT
                results["MIT_loss"] = MIT
                results["loss"] = ORT + MIT
            else:
                results["metric"] = self.validation_metric(recon_final, X_ori, indicating_mask)

        return results
