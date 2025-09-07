# Created by Franck Junior Aboya <mesabo18@gmail.com / messouaboya17@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType
import os
from ..patchtst import PatchEmbedding, FlattenHead
from .layers import ReprogrammingLayer
from ..revin import RevIN
from pypots.utils.profiling import measure_runtime_memory


class BackboneLLM4IMPold(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        patch_size,
        patch_stride,
        d_model,
        d_ffn,
        d_llm,
        n_heads,
        dropout,
        prompt_template: str = "Impute missing values at time steps where mask=0",
        train_gpt_mlp: bool = False,
        use_lora: bool = False,
        enable_profiling: bool = False,
        profiling_prefix: str = "backbone_llm4imp",
        profiling_path: str = "./output/imputation/profiling",
        use_hann_window: bool = False,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.d_llm = d_llm
        self.n_heads = n_heads
        self.prompt_template = prompt_template
        self.train_gpt_mlp = train_gpt_mlp
        self.use_lora = use_lora
        self.enable_profiling = enable_profiling
        self.profiling_prefix = profiling_prefix
        self.profiling_path = profiling_path
        self.use_hann_window = use_hann_window

        self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = GPT2Model.from_pretrained(
            "openai-community/gpt2",
            output_hidden_states=True,
        )

        # 🔒 Always start by freezing all LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False

        # 🔁 If LoRA is enabled
        if self.use_lora:
            print("✅ Applying LoRA to LLM")
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["c_fc", "c_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.llm = get_peft_model(self.llm, lora_config)

        # 🧠 If MLP training is enabled, allow MLP param gradients
        if self.train_gpt_mlp:
            print("✅ Unfreezing GPT MLP layers")
            for block in self.llm.h:
                for name, param in block.mlp.named_parameters():
                    param.requires_grad = True

        self.patch_embedding = PatchEmbedding(
            d_model=d_model,
            patch_size=patch_size,
            patch_stride=patch_stride,
            padding=patch_stride,
            dropout=dropout,
            positional_embedding=True,
        )

        self.reprogramming = ReprogrammingLayer(
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_ffn,
            d_llm=d_llm,
        )

        self.prompt_proj = nn.Linear(self.llm.config.n_embd, d_model)
        self.revin = RevIN(n_features=n_features, affine=False)

        self.output_head = None
        self.output_head_dropout = dropout

    def build_prompts(self, X: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        assert X.dim() == 3, f"Expected [B, T, D], got {X.shape}"
        B, T, D = X.shape
        prompts = []

        for b in range(B):
            x_sample = X[b].cpu()
            mask_sample = missing_mask[b].cpu()

            mean = x_sample.mean(dim=0, keepdim=True)
            std = x_sample.std(dim=0, keepdim=True) + 1e-6
            x_norm = (x_sample - mean) / std

            # window = torch.hann_window(T).unsqueeze(1)
            # x_windowed = x_norm * window
            if self.use_hann_window:
                window = torch.hann_window(T).unsqueeze(1).to(x_norm.device)
                x_windowed = x_norm * window
            else:
                x_windowed = x_norm

            var_per_dim = x_sample.var(dim=0)
            top_dims = torch.topk(var_per_dim, k=min(5, D)).indices
            x_top = x_windowed[:, top_dims]

            fft = torch.fft.rfft(x_top, dim=0)
            power = fft * torch.conj(fft)
            autocorr = torch.fft.irfft(power, dim=0)
            lag_scores = torch.mean(autocorr, dim=1)
            top_lags = torch.topk(lag_scores, k=min(5, T)).indices.tolist()

            min_vals = torch.min(x_sample, dim=0).values
            max_vals = torch.max(x_sample, dim=0).values
            median_vals = torch.median(x_sample, dim=0).values
            trend = torch.sum(x_sample[1:] - x_sample[:-1]).item()
            trend_desc = "upward" if trend > 0 else "downward"

            miss_idx_flat = (missing_mask[b] == 0).nonzero(as_tuple=False)
            miss_idx_str = ", ".join([f"({t},{d})" for t, d in miss_idx_flat.tolist()])

            prompt = (
                f"<|start_prompt|>Task: {self.prompt_template} "
                f"Missing indices: {miss_idx_str}. "
                f"Min values: {min_vals.mean().item():.2f}, "
                f"Max values: {max_vals.mean().item():.2f}, "
                f"Median: {median_vals.mean().item():.2f}, "
                f"Trend is {trend_desc}, "
                f"Top 5 lags: {top_lags}.<|end_prompt|>"
            )

            prompts.append(prompt)

        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        embeds = self.llm.get_input_embeddings()(tokenized.input_ids.to(X.device))
        return embeds

    def forward(self, X, missing_mask):
        if self.enable_profiling:
            result_container = {}

            def forward_wrapper(x, m):
                out = self._forward_internal(x, m)
                result_container["output"] = out
                return out

            profiling_result = measure_runtime_memory(
                forward_wrapper, X, missing_mask, save_path=self.profiling_path, prefix=self.profiling_prefix
            )
            self._last_profiling_result = profiling_result
            return result_container["output"]
        else:
            return self._forward_internal(X, missing_mask)

    def _forward_internal(self, X, missing_mask):
        X = self.revin(X, mode="norm")
        B, T, D = X.shape

        llm_embeds = self.build_prompts(X, missing_mask)
        llm_embeds = llm_embeds.unsqueeze(1).expand(-1, D, -1, -1)
        prompts_proj = self.prompt_proj(llm_embeds)

        X = X.permute(0, 2, 1).contiguous()
        X = X.view(B * D, 1, T)
        X = self.patch_embedding(X)
        P = X.shape[1]
        X = X.view(B, D, P, self.d_model)

        reprogrammed = self.reprogramming(X.flatten(0, 1), llm_embeds.flatten(0, 1), llm_embeds.flatten(0, 1))

        input_repr = reprogrammed.view(B, D, self.d_llm, P).permute(0, 1, 3, 2).contiguous()

        if self.output_head is None:
            self.output_head = FlattenHead(
                d_input=self.d_llm * P,
                d_output=self.n_steps,
                n_features=self.n_features,
                head_dropout=self.output_head_dropout,
            )

        output = self.output_head(input_repr)
        output = output.permute(0, 2, 1).contiguous()
        output = self.revin(output, mode="denorm")
        return output
