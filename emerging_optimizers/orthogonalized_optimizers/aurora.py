# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch.optim.optimizer import ParamsT

from emerging_optimizers import registry
from emerging_optimizers.mixin import WeightDecayT
from emerging_optimizers.orthogonalized_optimizers.muon import Muon, MuonScaleT, get_muon_scale_factor
from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import _args_doc
from emerging_optimizers.utils import FP32MatmulPrecT


__all__ = ["Aurora"]


@registry.register_optimizer("aurora")
class Aurora(Muon):
    """Aurora: leverage-uniform Stiefel descent via diagonal preconditioning.

    Aurora extends Muon on tall matrices by finding a positive diagonal
    preconditioner whose polar factor has uniform row norms. Wide and square
    matrices use the standard Muon update.

    Args:
        {_args_doc}
        coefficient_type: Newton-Schulz coefficient set.
        num_ns_steps: Number of Newton-Schulz steps per polar factor.
        scale_mode: Muon update scaling mode.
        extra_scale_factor: Additional multiplicative update scale.
        use_syrk: Whether to use the Triton SYRK Newton-Schulz kernel.
        pp_iterations: Number of diagonal preconditioning iterations.
        pp_beta: EMA coefficient for damping the row-normalization factors.

    References:
        - *Aurora: A Leverage-Aware Optimizer for Rectangular Matrices.* (2026).
          https://blog.tilderesearch.com/blog/aurora
          https://arxiv.org/pdf/2606.27715
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        *,
        nesterov: bool = False,
        weight_decay_method: WeightDecayT = "decoupled",
        fp32_matmul_prec: FP32MatmulPrecT = "medium",
        coefficient_type: NSCoeffT = "quintic",
        num_ns_steps: int = 5,
        scale_mode: MuonScaleT = "spectral",
        extra_scale_factor: float = 1.0,
        use_syrk: bool = False,
        pp_iterations: int = 3,
        pp_beta: float = 0.5,
    ) -> None:
        if pp_iterations < 1:
            raise ValueError(f"pp_iterations must be at least 1, got {pp_iterations}")
        if not 0.0 <= pp_beta <= 1.0:
            raise ValueError(f"pp_beta must be in [0, 1], got {pp_beta}")

        super().__init__(
            params,
            lr,
            momentum,
            weight_decay,
            nesterov=nesterov,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            use_syrk=use_syrk,
        )

        self.pp_iterations = pp_iterations
        self.pp_beta = pp_beta
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor
        self._muon_scaled_orthogonalize_fn = self.scaled_orthogonalize_fn
        self.scaled_orthogonalize_fn = self._scaled_aurora_orthogonalize

    def _polar(self, matrix: torch.Tensor) -> torch.Tensor:
        scale = get_muon_scale_factor(matrix.size(-2), matrix.size(-1), mode=self.scale_mode) * self.extra_scale_factor
        return self._muon_scaled_orthogonalize_fn(matrix) / scale

    def _orthogonalize_tall(self, matrix: torch.Tensor) -> torch.Tensor:
        num_rows = matrix.size(0)
        eps = torch.finfo(matrix.dtype).eps

        update = matrix / matrix.norm().clamp_min(eps)
        damping = torch.ones((num_rows, 1), dtype=matrix.dtype, device=matrix.device)
        for _ in range(self.pp_iterations):
            row_norms = update.norm(dim=-1, keepdim=True).clamp_min_(eps)
            damping = damping.pow(self.pp_beta) * row_norms.pow(1.0 - self.pp_beta)
            update = self._polar(update / damping)

        return update

    def _scaled_aurora_orthogonalize(self, grad: torch.Tensor) -> torch.Tensor:
        scale = get_muon_scale_factor(grad.size(-2), grad.size(-1), mode=self.scale_mode)
        scale *= self.extra_scale_factor
        if scale == 0.0:
            return torch.zeros_like(grad)

        original_dtype = grad.dtype
        matrix = grad.to(torch.float32)
        num_rows, num_cols = matrix.shape

        if num_rows > num_cols:
            update = self._orthogonalize_tall(matrix)
        else:
            update = self._polar(matrix)

        return (update * scale).to(original_dtype)


Aurora.__doc__ = Aurora.__doc__.format(_args_doc=_args_doc)  # type: ignore[union-attr]
