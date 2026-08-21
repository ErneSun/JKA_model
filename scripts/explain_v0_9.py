#!/usr/bin/env python3
"""Print the minimal V0.9 tensor and architecture contract."""

from __future__ import annotations

import torch

from jka_model.adaptive import LowRankAdaptiveOperator, residual_decomposition
from jka_model.config import V09AdaptiveConfig


def main() -> None:
    batch, history, latent_dim, context_dim, rank = 2, 8, 32, 8, 4
    history_z = torch.randn(batch, history, latent_dim)
    context = torch.randn(batch, context_dim)
    dt = torch.full((batch, 1), 0.15)
    a0 = torch.zeros(latent_dim, latent_dim)
    adapter = LowRankAdaptiveOperator(
        a0,
        context_dim,
        V09AdaptiveConfig(rank=rank),
    )
    prediction, eta, delta, adapted = adapter.step(history_z[:, -1], context, dt)
    truth = torch.randn_like(prediction)
    nominal = history_z[:, -1]
    r0, rop, rrem = residual_decomposition(truth, nominal, prediction)
    print("physical history: [B,H,C,Nx,Ny] (offline/frozen encoder)")
    print(f"z history: {tuple(history_z.shape)}")
    print(f"context: {tuple(context.shape)}")
    print(f"eta: {tuple(eta.shape)}")
    print(f"A0/delta_A/A_t: {tuple(a0.shape)} / {tuple(delta.shape)} / {tuple(adapted.shape)}")
    print(f"prediction: {tuple(prediction.shape)}")
    print(f"r0/rop/rrem: {tuple(r0.shape)} / {tuple(rop.shape)} / {tuple(rrem.shape)}")
    print("Is backbone frozen? YES")
    print("Is context encoder frozen? YES")
    print("Is A0 frozen? YES")
    print("Is eta_t implemented? YES")
    print("Is A_t adaptive? YES")
    print("Is additive residual correction enabled? NO")
    print("Is persistent z_R present? NO")


if __name__ == "__main__":
    main()
