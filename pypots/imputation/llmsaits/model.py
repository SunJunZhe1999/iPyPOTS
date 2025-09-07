"""
The implementation of LLM4IMP for the partially-observed time-series imputation task.

"""

# Created by Franck Junior Aboya <mesabo18@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import _LLMSAITS
from ..base import BaseNNImputer
from ..saits.data import DatasetForSAITS
from ...data.checking import key_in_data_set
from ...nn.modules.loss import Criterion, MAE, MSE
from ...optim.adam import Adam
from ...optim.base import Optimizer


class LLMSAITS(BaseNNImputer):
    """The PyTorch implementation of the LLMSAITS model combining frozen LLM and SAITS-style refinement."""

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
        # Flags
        train_gpt_mlp: bool = False,
        use_lora: bool = False,
        enable_profiling: bool = False,
        use_hann_window: bool = False,
        llm_model_type: str = "gpt2",
        profiling_path: str = "./output/imputation/profiling",
        profiling_prefix: str = "backbone_llmsaits",
        ORT_weight: int = 1,
        MIT_weight: int = 1,
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

        self.model = _LLMSAITS(
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
            train_gpt_mlp=train_gpt_mlp,
            use_lora=use_lora,
            enable_profiling=enable_profiling,
            profiling_path=profiling_path,
            profiling_prefix=profiling_prefix,
            use_hann_window=use_hann_window,
            llm_model_type=llm_model_type,
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

    def predict(self, test_set: Union[dict, str], file_type: str = "hdf5", diagonal_attention_mask: bool = True) -> dict:
        return super().predict(test_set, file_type, diagonal_attention_mask=diagonal_attention_mask)

    def impute(self, test_set: Union[dict, str], file_type: str = "hdf5", diagonal_attention_mask: bool = True) -> np.ndarray:
        return super().impute(test_set, file_type, diagonal_attention_mask=diagonal_attention_mask)