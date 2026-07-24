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
from typing import override

import torch
import torch.nn as nn
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers import registry
from emerging_optimizers.orthogonalized_optimizers import aurora
from emerging_optimizers.orthogonalized_optimizers.muon import get_muon_scale_factor


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class AuroraTest(parameterized.TestCase):
    @override
    def setUp(self) -> None:
        self.device = FLAGS.device

    @parameterized.parameters(((8, 4),), ((4, 8),), ((8, 8),))  # type: ignore[misc]
    def test_smoke(self, shape: tuple[int, int]) -> None:
        param = nn.Parameter(torch.randn(shape, dtype=torch.float32, device=self.device))
        param.grad = torch.randn_like(param)
        optimizer = aurora.Aurora([param], fp32_matmul_prec="highest")

        optimizer.step()

        self.assertTrue(torch.isfinite(param).all())

    def _make_exact_optimizer(
        self,
        shape: tuple[int, int],
        *,
        extra_scale_factor: float = 1.0,
        pp_iterations: int = 40,
        pp_beta: float = 0.5,
    ) -> aurora.Aurora:
        param = nn.Parameter(torch.randn(shape, dtype=torch.float32, device=self.device))
        optimizer = aurora.Aurora(
            [param],
            fp32_matmul_prec="highest",
            pp_iterations=pp_iterations,
            pp_beta=pp_beta,
            extra_scale_factor=extra_scale_factor,
        )

        def exact_scaled_polar(matrix: torch.Tensor) -> torch.Tensor:
            left, _, right = torch.linalg.svd(matrix, full_matrices=False)
            scale = get_muon_scale_factor(matrix.size(-2), matrix.size(-1), mode="spectral")
            return (left @ right) * scale * extra_scale_factor

        optimizer._muon_scaled_orthogonalize_fn = exact_scaled_polar
        return optimizer

    def test_tall_preconditioning_matches_reference(self) -> None:
        pp_iterations = 3
        pp_beta = 0.5
        optimizer = self._make_exact_optimizer(
            (8, 4),
            pp_iterations=pp_iterations,
            pp_beta=pp_beta,
        )
        matrix = torch.randn((8, 4), dtype=torch.float32, device=self.device)
        eps = torch.finfo(matrix.dtype).eps

        expected = matrix / matrix.norm().clamp_min(eps)
        damping = torch.ones((matrix.size(0), 1), dtype=matrix.dtype, device=matrix.device)
        for _ in range(pp_iterations):
            row_norms = expected.norm(dim=-1, keepdim=True).clamp_min(eps)
            damping = damping.pow(pp_beta) * row_norms.pow(1.0 - pp_beta)
            expected = optimizer._polar(expected / damping)

        actual = optimizer._orthogonalize_tall(matrix)

        torch.testing.assert_close(actual, expected)

    def test_tall_update_is_uniform_and_semi_orthogonal(self) -> None:
        num_rows, num_cols = 8, 4
        optimizer = self._make_exact_optimizer((num_rows, num_cols), pp_iterations=80)
        generator = torch.Generator(device="cpu").manual_seed(1234)
        grad = torch.randn((num_rows, num_cols), dtype=torch.float32, generator=generator).to(self.device)

        update = optimizer.scaled_orthogonalize_fn(grad)
        scale = get_muon_scale_factor(num_rows, num_cols, mode="spectral")
        normalized = update / scale

        gram = normalized.mT @ normalized
        row_norm_sq = normalized.square().sum(dim=-1)
        target = num_cols / num_rows

        torch.testing.assert_close(
            gram,
            torch.eye(num_cols, dtype=gram.dtype, device=gram.device),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            row_norm_sq,
            torch.full_like(row_norm_sq, target),
            atol=2e-3,
            rtol=2e-3,
        )

    @parameterized.parameters(((4, 8),), ((8, 8),))  # type: ignore[misc]
    def test_wide_and_square_updates_match_muon(self, shape: tuple[int, int]) -> None:
        optimizer = self._make_exact_optimizer(shape)
        grad = torch.randn(shape, dtype=torch.float32, device=self.device)

        expected = optimizer._muon_scaled_orthogonalize_fn(grad)
        actual = optimizer.scaled_orthogonalize_fn(grad)

        torch.testing.assert_close(actual, expected)

    def test_zero_extra_scale_returns_zero(self) -> None:
        optimizer = self._make_exact_optimizer((8, 4), extra_scale_factor=0.0)
        grad = torch.randn((8, 4), dtype=torch.float32, device=self.device)

        torch.testing.assert_close(optimizer.scaled_orthogonalize_fn(grad), torch.zeros_like(grad))

    @parameterized.parameters(0, -1)  # type: ignore[misc]
    def test_invalid_pp_iterations_raises_value_error(self, pp_iterations: int) -> None:
        param = nn.Parameter(torch.randn((8, 4), device=self.device))
        with self.assertRaisesRegex(ValueError, "pp_iterations must be at least 1"):
            aurora.Aurora([param], pp_iterations=pp_iterations)

    @parameterized.parameters(-0.1, 1.1)  # type: ignore[misc]
    def test_invalid_pp_beta_raises_value_error(self, pp_beta: float) -> None:
        param = nn.Parameter(torch.randn((8, 4), device=self.device))
        with self.assertRaisesRegex(ValueError, "pp_beta must be in"):
            aurora.Aurora([param], pp_beta=pp_beta)

    @parameterized.parameters(0.0, 1.0)  # type: ignore[misc]
    def test_pp_beta_boundaries_are_valid(self, pp_beta: float) -> None:
        param = nn.Parameter(torch.randn((8, 4), device=self.device))
        aurora.Aurora([param], pp_beta=pp_beta)

    def test_trains_tall_linear_regression(self) -> None:
        num_inputs, num_outputs, num_examples = 4, 16, 128
        generator = torch.Generator(device="cpu").manual_seed(5678)
        inputs = torch.randn((num_examples, num_inputs), generator=generator).to(self.device)
        target_weight = torch.randn((num_outputs, num_inputs), generator=generator).to(self.device)
        targets = inputs @ target_weight.mT
        weight = nn.Parameter(torch.zeros_like(target_weight))
        optimizer = aurora.Aurora(
            [weight],
            lr=0.05,
            momentum=0.9,
            weight_decay=0.0,
            fp32_matmul_prec="highest",
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        initial_loss = torch.nn.functional.mse_loss(inputs @ weight.mT, targets)
        for _ in range(200):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(inputs @ weight.mT, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
        final_loss = torch.nn.functional.mse_loss(inputs @ weight.mT, targets)

        logging.info("Aurora regression loss: %.6f -> %.6f", initial_loss.item(), final_loss.item())
        self.assertLess(final_loss.item(), initial_loss.item() * 1e-3)

    def test_registered(self) -> None:
        self.assertIs(registry.get_optimizer_cls("aurora"), aurora.Aurora)


if __name__ == "__main__":
    absltest.main()
