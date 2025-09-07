"""
The implementation of LLM4IMP for the partially-observed time-series imputation task.

"""

# Created by Franck Junior Aboya <mesabo18@gmail.com>
# License: BSD-3-Clause

from copy import deepcopy
from typing import Union, Optional

import torch
from torch.utils.data import DataLoader

from .core import _LLM4IMP
from ..base import BaseNNImputer
from ..saits.data import DatasetForSAITS
from ...data.checking import key_in_data_set
from ...nn.modules.loss import Criterion, MAE, MSE
from ...optim.adam import Adam
from ...optim.base import Optimizer


class LLM4IMP(BaseNNImputer):
    """
    The PyTorch implementation of the LLM4IMP model for partially observed multivariate
    time-series imputation.

    LLM4IMP reprograms a frozen LLM backbone (e.g., GPT-2) to operate on numeric
    sequences via (i) PatchTST-style patch embeddings, (ii) optional spectral/semantic
    prompts rendered as token embeddings, and (iii) a cross-domain attention
    reprogramming layer that aligns numeric patch tokens with LLM token space. The
    model reconstructs missing values while preserving observed entries.

    Parameters
    ----------
    n_steps : int
        Number of time steps per sample (T).

    n_features : int
        Number of variables (D).

    patch_size : int, default=12
        Temporal length of each patch for patch embedding.

    patch_stride : int, default=6
        Hop size between adjacent patches.

    d_model : int, default=64
        Hidden size of the numeric patch embedding pathway (query dimension).

    d_ffn : int, default=128
        Internal key/query width used by the reprogramming layer (acts as d_k).

    d_llm : int, default=768
        Working dimensionality on the LLM side (projected token embedding size for K/V).

    n_heads : int, default=4
        Number of attention heads in the reprogramming layer. Must divide both
        d_model and d_llm appropriately for multi-head attention.

    n_layers : int, default=2
        Number of final transformer blocks in the LLM to unfreeze (0 keeps LLM fully
        frozen—no backbone fine-tuning).

    dropout : float, default=0.1
        Dropout applied within patch embedding / heads.

    prompt_template : str, default="Impute missing values at time steps where mask=0"
        Natural-language scaffold used when building textual prompts describing
        missingness and summary statistics.

    batch_size : int, default=32
        Mini-batch size for training and evaluation.

    epochs : int, default=5
        Maximum number of training epochs.

    patience : int or None, default=None
        Early stopping patience. If None, early stopping is disabled.

    training_loss : Union[pypots.nn.modules.loss.Criterion, type], default=MAE
        Training loss (class or instance). If a class is provided, it will be instantiated.

    validation_metric : Union[pypots.nn.modules.loss.Criterion, type], default=MSE
        Validation metric for model selection (class or instance).

    optimizer : Union[pypots.optim.base.Optimizer, type], default=Adam
        Optimizer (class or instance) following the PyPOTS optimizer interface.

    num_workers : int, default=0
        Number of workers for PyTorch DataLoader.

    device : Optional[Union[str, torch.device, list]], default=None
        Compute device. If None, uses CUDA if available, else CPU. A list enables
        DataParallel on CUDA devices (e.g., ["cuda:0","cuda:1"]).

    saving_path : Optional[str], default=None
        Directory for checkpoints and TensorBoard logs. Disabled if None.

    model_saving_strategy : {None, "best", "better", "all"}, default="best"
        Checkpoint strategy for automatic saving.

    verbose : bool, default=True
        Controls logging verbosity.

    train_gpt_mlp : bool, default=False
        If True, allows selective fine-tuning of MLP submodules in the unfrozen
        LLM blocks (only effective when n_layers > 0).

    use_lora : bool, default=False
        If True, wraps the LLM with parameter-efficient LoRA adapters. Kept in
        code for research toggles; off by default.

    enable_profiling : bool, default=False
        If True, measures runtime and memory per forward pass and logs to disk.

    use_hann_window : bool, default=False
        If True, applies a Hann window before frequency analysis in prompt building.

    llm_model_type : str, default="gpt2"
        Backbone identifier. Currently supports "gpt2" and "distilgpt2".

    profiling_path : str, default="./output/imputation/profiling"
        Directory to store profiling CSV/JSON artifacts.

    profiling_prefix : str, default="backbone_llm4imp"
        Prefix to tag profiling files.

    use_prompt : bool, default=True
        Enables the prompt path. If False and soft prompts are disabled, the
        model uses a zero prompt embedding.

    use_reprogramming : bool, default=True
        Enables cross-domain attention (numeric→LLM). If False, a linear projection
        is used to map numeric tokens directly to d_llm.

    Notes
    -----
    Input/Output shapes (per batch):
        - Input X: (B, T, D)
        - Mask M: (B, T, D), 1=observed, 0=missing (provided by the dataset API)
        - Output Ŷ: (B, T, D) imputed sequences

    Training objective:
        The model minimizes masked reconstruction loss on missing entries while
        preserving observed values. Early stopping is driven by `validation_metric`.

    Freezing policy:
        By default, all LLM parameters are frozen. Setting `n_layers > 0` unfreezes
        the last `n_layers` transformer blocks; `train_gpt_mlp` further restricts
        updates to MLP submodules if desired.

    Device behavior:
        The backbone, prompt embeddings, and numeric modules are bound to the
        input device at first forward call. If multiple CUDA devices are given,
        DataParallel is used upstream by the framework.

    References
    ----------
    Please see the associated paper and codebase for architectural details and
    ablations.
    """

    def __init__(
            self,
            n_steps: int,
            n_features: int,
            patch_size: int = 12,
            patch_stride: int = 6,
            d_model: int = 64,
            d_ffn: int = 128,
            d_llm: int = 768,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.1,
            prompt_template: str = "Impute missing values at time steps where mask=0",
            batch_size: int = 32,
            epochs: int = 5,
            patience: Optional[int] = None,
            training_loss: Union[Criterion, type] = MAE,
            validation_metric: Union[Criterion, type] = MSE,
            optimizer: Union[Optimizer, type] = Adam,
            num_workers: int = 0,
            device: Optional[Union[str, torch.device, list]] = None,
            saving_path: Optional[str] = None,
            model_saving_strategy: Optional[str] = "best",
            verbose: bool = True,
            train_gpt_mlp: bool = False,
            use_lora: bool = False,
            enable_profiling: bool = False,
            use_hann_window: bool = False,
            llm_model_type: str = "gpt2",
            profiling_path: str = "./output/imputation/profiling",
            profiling_prefix: str = "backbone_llm4imp",
            use_prompt: bool = True,      
            use_reprogramming: bool = True,
    ):
        super().__init__(
            training_loss=training_loss,
            validation_metric=validation_metric,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            device=device,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )

        self.n_steps = n_steps
        self.n_features = n_features

        self.model = _LLM4IMP(
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
            training_loss=self.training_loss,
            validation_metric=self.validation_metric,

            # ✅ Pass through additional arguments
            train_gpt_mlp=train_gpt_mlp,
            use_lora=use_lora,
            enable_profiling=enable_profiling,
            profiling_path=profiling_path,
            profiling_prefix=profiling_prefix,
            use_hann_window=use_hann_window,
            llm_model_type=llm_model_type,
            use_prompt=use_prompt,
            use_reprogramming=use_reprogramming,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        self.optimizer = optimizer() if isinstance(optimizer, type) else optimizer
        self.optimizer.init_optimizer(self.model.parameters())

    def _organize_content_to_save(self):
        from ...version import __version__ as pypots_version

        if isinstance(self.device, list):
            model_state_dict = deepcopy(self.model.module.state_dict())
        else:
            model_state_dict = deepcopy(self.model.state_dict())

        return {
            "model_state_dict": model_state_dict,
            "pypots_version": pypots_version,
        }

    def _assemble_input_for_training(self, data: list) -> dict:
        indices, X, missing_mask, X_ori, indicating_mask = self._send_data_to_given_device(data)
        return {
            "X": X,
            "missing_mask": missing_mask,
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }

    def _assemble_input_for_validating(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: list) -> dict:
        indices, X, missing_mask = self._send_data_to_given_device(data)
        return {
            "X": X,
            "missing_mask": missing_mask,
        }

    def fit(
            self,
            train_set: Union[dict, str],
            val_set: Optional[Union[dict, str]] = None,
            file_type: str = "hdf5",
    ) -> None:
        train_dataset = DatasetForSAITS(train_set, return_X_ori=False, return_y=False, file_type=file_type)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        val_dataloader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for validation.")
            val_dataset = DatasetForSAITS(val_set, return_X_ori=True, return_y=False, file_type=file_type)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        self._train_model(train_dataloader, val_dataloader)
        self.model.load_state_dict(self.best_model_dict)
        self._auto_save_model_if_necessary(confirm_saving=True)
