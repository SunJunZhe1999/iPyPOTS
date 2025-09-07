# Created by Franck Junior Aboya <mesabo18@gmail.com / messouaboya17@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType

from ..patchtst.layers import PatchEmbedding, FlattenHead
from .layers import ReprogrammingLayer
from ..revin import RevIN
from pypots.utils.profiling import measure_runtime_memory


class BackboneLLM4IMP(nn.Module):
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
        llm_model_type: str = "gpt2",
        n_layers: int = 2,
        num_soft_prompt_tokens: int = 500,
        use_prompt: bool = True,
        use_reprogramming: bool = True,
    ):
        super().__init__()

        # ---------------- core hparams ----------------
        self.n_steps = n_steps
        self.n_features = n_features
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.d_llm = d_llm
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.prompt_template = prompt_template
        self.train_gpt_mlp = train_gpt_mlp
        self.use_lora = use_lora
        self.enable_profiling = enable_profiling
        self.profiling_prefix = profiling_prefix
        self.profiling_path = profiling_path
        self.use_hann_window = use_hann_window
        self.llm_model_type = llm_model_type
        self.num_soft_prompt_tokens = num_soft_prompt_tokens
        self.use_prompt = use_prompt
        self.use_reprogramming = use_reprogramming

        # Track whether we've moved submodules to the input device
        self._device_bound = False

        # ---------------- tokenizer & backbone ----------------
        model_name = "distilgpt2" if llm_model_type == "distilgpt2" else "openai-community/gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.llm.eval()

        # Freeze all LLM params by default
        for p in self.llm.parameters():
            p.requires_grad = False

        # Unfreeze last n_layers blocks if requested
        blocks = getattr(self.llm, "transformer", self.llm).h
        for i, block in enumerate(reversed(blocks)):
            if i < n_layers:
                for _, param in block.named_parameters():
                    param.requires_grad = True

        # Optional LoRA
        if self.use_lora:
            lora_config = LoraConfig(
                r=8, lora_alpha=32, target_modules=["c_attn", "c_proj", "c_fc"], lora_dropout=0.05,
                bias="none", task_type=TaskType.FEATURE_EXTRACTION
            )
            self.llm = get_peft_model(self.llm, lora_config)

        # Optional soft prompt
        if num_soft_prompt_tokens > 0:
            self.soft_prompt = nn.Parameter(
                torch.randn(num_soft_prompt_tokens, self.llm.config.n_embd)
            )
        else:
            self.soft_prompt = None

        # ---------------- numeric path ----------------
        self.patch_embedding = PatchEmbedding(
            d_model=d_model,
            patch_size=patch_size,
            patch_stride=patch_stride,
            padding=patch_stride,
            dropout=dropout,
            positional_embedding=True,
        )

        # Cross-domain attention reprogramming
        self.reprogramming = ReprogrammingLayer(
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_ffn,
            d_llm=d_llm,
        )

        # PROJECTION FIX:
        # Project LLM token embeddings (n_embd) → d_llm for use as K/V in the reprogramming layer
        self.prompt_to_llm = nn.Linear(self.llm.config.n_embd, d_llm)

        # Used only when reprogramming is disabled: project numeric features → d_llm
        self.reprogramming_proj = nn.Linear(d_model, d_llm)

        self.revin = RevIN(n_features=n_features, affine=False)

        self.output_head = None
        self.output_head_dropout = dropout

    # --- util: ensure submodules are on the same device as inputs (first call only) ---
    def _bind_to_input_device(self, x: torch.Tensor):
        if self._device_bound:
            return
        dev = x.device
        # NOTE: .to(dev) will move parameters & buffers; DataParallel wrapping is upstream in the trainer
        self.patch_embedding.to(dev)
        self.reprogramming.to(dev)
        self.prompt_to_llm.to(dev)
        self.reprogramming_proj.to(dev)
        self.revin.to(dev)
        if self.output_head is not None:
            self.output_head.to(dev)
        # move LLM too
        self.llm.to(dev)
        self._device_bound = True

    # ---------------- prompt builder (device-safe) ----------------
    def build_prompts(self, X: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns LLM input embeddings for B prompts: [B, L_prompt, n_embd],
        on the SAME device as X (no CPU detours).
        """
        B, T, D = X.shape

        if not self.use_prompt:
            L = self.num_soft_prompt_tokens if self.num_soft_prompt_tokens > 0 else 1
            E = self.llm.config.n_embd
            return torch.zeros(B, L, E, device=X.device)

        # If soft prompt exists, just tile it (already a Parameter and will be on correct device)
        if self.soft_prompt is not None:
            return self.soft_prompt.unsqueeze(0).expand(B, -1, -1)

        prompts = []
        # All ops stay on X.device (no .cpu())
        for b in range(B):
            x_sample = X[b]           # [T, D]
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
            x_top = x_windowed[:, top_dims]  # [T, d_top]

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

        # Tokenize on CPU (OK), then move IDs to X.device for embedding lookup on-GPU
        tok = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = tok.input_ids.to(X.device)
        embeds = self.llm.get_input_embeddings()(input_ids)  # [B, L_prompt, n_embd] on X.device
        return embeds

    # ---------------- public forward ----------------
    def forward(self, X, missing_mask):
        if self.enable_profiling:
            result_container = {}

            def forward_wrapper(x, m):
                out = self._forward_internal(x, m)
                result_container["output"] = out
                return out

            profiling_result = measure_runtime_memory(
                forward_wrapper, X, missing_mask,
                save_path=self.profiling_path, prefix=self.profiling_prefix
            )
            self._last_profiling_result = profiling_result
            return result_container["output"]
        else:
            return self._forward_internal(X, missing_mask)

    # ---------------- internal forward (device-safe) ----------------
    def _forward_internal(self, X, missing_mask):
        # Lazily move all submodules to the first input's device once
        self._bind_to_input_device(X)

        # RevIN norm
        X = self.revin(X, mode="norm")  # [B, T, D]
        B, T, D = X.shape

        # Build prompts on the same device as X
        llm_embeds = self.build_prompts(X, missing_mask)  # [B, Lp, n_embd] @ X.device
        # Expand per-feature (still on-device)
        llm_embeds = llm_embeds.unsqueeze(1).expand(-1, D, -1, -1)  # [B, D, Lp, n_embd]

        # Numeric path → patch tokens
        Xn = X.permute(0, 2, 1).contiguous()  # [B, D, T]
        Xn = Xn.view(B * D, 1, T)            # [B*D, 1, T]
        Xn = self.patch_embedding(Xn)        # [B*D, P, d_model]
        P = Xn.shape[1]
        Xn = Xn.view(B, D, P, self.d_model)  # [B, D, P, d_model]

        if self.use_reprogramming:
            # Project LLM embeddings to d_llm for KV — FIX
            kv = self.prompt_to_llm(llm_embeds.flatten(0, 1))  # [(B*D), Lp, d_llm]
            q = Xn.flatten(0, 1)                               # [(B*D), P, d_model]
            reprogrammed = self.reprogramming(q, kv, kv)       # [(B*D), P, d_llm]
        else:
            # No cross-attn: map numeric d_model → d_llm
            reprogrammed = self.reprogramming_proj(Xn.flatten(0, 1))  # [(B*D), P, d_llm]

        input_repr = reprogrammed.view(B, D, P, self.d_llm).permute(0, 1, 2, 3).contiguous()
        # FlattenHead expects [B, D, P, d_llm] → outputs [B, D, n_steps]

        if self.output_head is None:
            self.output_head = FlattenHead(
                d_input=self.d_llm * P,
                d_output=self.n_steps,
                n_features=self.n_features,
                head_dropout=self.output_head_dropout,
            ).to(X.device)

        output = self.output_head(input_repr)  # [B, D, n_steps]
        output = output.permute(0, 2, 1).contiguous()  # [B, n_steps, D]

        # RevIN denorm
        output = self.revin(output, mode="denorm")
        return output
    