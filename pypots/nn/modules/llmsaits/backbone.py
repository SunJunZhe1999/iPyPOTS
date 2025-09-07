# Created by Franck Junior Aboya <mesabo18@gmail.com / messouaboya17@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType

from ..patchtst.layers import PatchEmbedding, FlattenHead
from .layers import ReprogrammingLayer
from ..revin import RevIN
from pypots.utils.profiling import measure_runtime_memory


class BackboneLLMSAITS(nn.Module):
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

        self._device_bound = False  # <-- NEW

        self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = GPT2Model.from_pretrained("openai-community/gpt2", output_hidden_states=True)
        self.llm.eval()

        # freeze by default
        for p in self.llm.parameters():
            p.requires_grad = False

        if self.use_lora:
            lora_config = LoraConfig(
                r=8, lora_alpha=32, target_modules=["c_fc", "c_proj"],
                lora_dropout=0.05, bias="none", task_type=TaskType.FEATURE_EXTRACTION
            )
            self.llm = get_peft_model(self.llm, lora_config)

        if self.train_gpt_mlp:
            for block in self.llm.h:
                for _, param in block.mlp.named_parameters():
                    param.requires_grad = True

        self.patch_embedding_1 = PatchEmbedding(
            d_model=d_model, patch_size=patch_size, patch_stride=patch_stride,
            padding=patch_stride, dropout=dropout, positional_embedding=True,
        )
        self.patch_embedding_2 = PatchEmbedding(
            d_model=d_model, patch_size=patch_size, patch_stride=patch_stride,
            padding=patch_stride, dropout=dropout, positional_embedding=True,
        )

        self.reprogramming_1 = ReprogrammingLayer(d_model=d_model, n_heads=n_heads, d_k=d_ffn, d_llm=d_llm)
        self.reprogramming_2 = ReprogrammingLayer(d_model=d_model, n_heads=n_heads, d_k=d_ffn, d_llm=d_llm)

        self.prompt_proj = nn.Linear(self.llm.config.n_embd, d_model)  # (optional) not strictly needed below
        self.revin = RevIN(n_features=n_features, affine=False)
        self.output_head = None
        self.output_head_dropout = dropout

        self.combiner = nn.Sequential(
            nn.Linear(2 * n_features, n_features),
            nn.Sigmoid()
        )

    # ---- NEW: bind all modules (incl. LLM) to the input device once ----
    def _bind_to_input_device(self, x: torch.Tensor):
        if self._device_bound:
            return
        dev = x.device
        self.llm.to(dev)
        self.patch_embedding_1.to(dev)
        self.patch_embedding_2.to(dev)
        self.reprogramming_1.to(dev)
        self.reprogramming_2.to(dev)
        self.prompt_proj.to(dev)
        self.revin.to(dev)
        self.combiner.to(dev)
        if self.output_head is not None:
            self.output_head.to(dev)
        self._device_bound = True

    def build_prompts(self, X: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = X.shape
        prompts = []
        # keep everything on X.device (no .cpu())
        for b in range(B):
            x_sample = X[b]
            mask_sample = missing_mask[b]
            mean = x_sample.mean(dim=0, keepdim=True)
            std = x_sample.std(dim=0, keepdim=True) + 1e-6
            x_norm = (x_sample - mean) / std

            if self.use_hann_window:
                window = torch.hann_window(T, device=X.device).unsqueeze(1)
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

            miss_idx_flat = (mask_sample == 0).nonzero(as_tuple=False).tolist()
            miss_idx_str = ", ".join([f"({t},{d})" for t, d in miss_idx_flat])

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

        tok = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = tok.input_ids.to(X.device)  # move IDs to the same device
        embeds = self.llm.get_input_embeddings()(input_ids)  # [B, L, n_embd] on X.device
        return embeds

    def forward(self, X, missing_mask):
        if self.enable_profiling:
            outbox = {}
            def wrapper(x, m):
                y = self._forward_internal(x, m)
                outbox["y"] = y
                return y
            _ = measure_runtime_memory(wrapper, X, missing_mask, save_path=self.profiling_path, prefix=self.profiling_prefix)
            return outbox["y"]
        return self._forward_internal(X, missing_mask)

    def _forward_internal(self, X, missing_mask):
        # <<< FIX: ensure *all* modules (incl. LLM) are on X.device >>>
        self._bind_to_input_device(X)

        X = self.revin(X, mode="norm")
        B, T, D = X.shape

        llm_embeds = self.build_prompts(X, missing_mask)               # [B, Lp, n_embd] @ X.device
        llm_embeds = llm_embeds.unsqueeze(1).expand(-1, D, -1, -1)     # [B, D, Lp, n_embd]
        _ = self.prompt_proj(llm_embeds)                                # optional; ensures layer is device-bound

        X_input = X.permute(0, 2, 1).contiguous().view(B * D, 1, T)
        patches_1 = self.patch_embedding_1(X_input).view(B, D, -1, self.d_model)
        reprogrammed_1 = self.reprogramming_1(
            patches_1.flatten(0, 1), llm_embeds.flatten(0, 1), llm_embeds.flatten(0, 1)
        )
        repr_1 = reprogrammed_1.view(B, D, -1, self.d_llm).permute(0, 1, 3, 2).contiguous()

        if self.output_head is None:
            self.output_head = FlattenHead(
                d_input=self.d_llm * repr_1.shape[3],
                d_output=self.n_steps,
                n_features=self.n_features,
                head_dropout=self.output_head_dropout,
            ).to(X.device)

        X_tilde_1 = self.output_head(repr_1).permute(0, 2, 1).contiguous()  # [B, T, D]
        X_prime = missing_mask * X + (1 - missing_mask) * X_tilde_1

        X_prime_input = X_prime.permute(0, 2, 1).contiguous().view(B * D, 1, T)
        patches_2 = self.patch_embedding_2(X_prime_input).view(B, D, -1, self.d_model)
        reprogramming_2 = self.reprogramming_2(
            patches_2.flatten(0, 1), llm_embeds.flatten(0, 1), llm_embeds.flatten(0, 1)
        )
        repr_2 = reprogramming_2.view(B, D, -1, self.d_llm).permute(0, 1, 3, 2).contiguous()
        X_tilde_2 = self.output_head(repr_2).permute(0, 2, 1).contiguous()  # [B, T, D]

        # Make sure mask is float on the same device
        if missing_mask.dtype != torch.float32:
            missing_mask = missing_mask.float()
        missing_mask = missing_mask.to(X.device)

        weights = self.combiner(torch.cat([missing_mask, torch.sigmoid(X_tilde_2)], dim=2))  # [B, T, D]
        X_final = weights * X_tilde_1 + (1 - weights) * X_tilde_2

        X_tilde_1 = self.revin(X_tilde_1, mode="denorm")
        X_tilde_2 = self.revin(X_tilde_2, mode="denorm")
        X_final = self.revin(X_final, mode="denorm")

        return X_tilde_1, X_tilde_2, X_final