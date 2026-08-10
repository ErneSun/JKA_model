"""Direct-state continuous-time Koopman generator and exact propagation.

Mathematics uses column states: ``z_next = K @ z`` with
``K(dt) = exp(A * dt)``. A tensor row ``z[b]`` stores one such mathematical
column state, so shared-matrix batch code uses the equivalent ``z @ K.T``.
No Euler/RK approximation is used in this model.
"""

from __future__ import annotations

from numbers import Real

import torch
from torch import Tensor, nn

from jka_model.metrics import SpectrumDiagnostics, continuous_spectrum


class ContinuousKoopmanCore(nn.Module):
    """Autonomous ``dz/dt=A z`` model with fixed or trainable generator ``A``."""

    def __init__(
        self,
        state_dim: int,
        *,
        generator: Tensor | None = None,
        trainable: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError("ContinuousKoopmanCore supports float32 or float64")
        if generator is None:
            matrix = torch.zeros((state_dim, state_dim), dtype=dtype, device=device)
        else:
            if generator.shape != (state_dim, state_dim):
                raise ValueError("generator must have shape [state_dim,state_dim]")
            if not torch.is_floating_point(generator) or not torch.isfinite(generator).all():
                raise ValueError("generator must be a finite floating-point tensor")
            matrix = generator.detach().to(device=device, dtype=dtype).clone()
        self.state_dim = state_dim
        if trainable:
            self.A = nn.Parameter(matrix)
        else:
            self.register_buffer("A", matrix)

    @property
    def trainable(self) -> bool:
        return isinstance(self.A, nn.Parameter) and self.A.requires_grad

    def generator_matrix(self) -> Tensor:
        """Return the live continuous generator tensor ``A``."""
        return self.A

    def _dt_tensor(self, dt: Tensor | Real) -> Tensor:
        if isinstance(dt, Real):
            value = torch.tensor(dt, device=self.A.device, dtype=self.A.dtype)
        elif isinstance(dt, Tensor):
            if dt.ndim not in {0, 1}:
                raise ValueError("dt must be scalar or have shape [B]")
            value = dt.to(device=self.A.device, dtype=self.A.dtype)
        else:
            raise TypeError("dt must be a real scalar or torch.Tensor")
        if not torch.isfinite(value).all():
            raise ValueError("dt must contain only finite values")
        if torch.any(value < 0):
            raise ValueError("negative dt is not supported")
        return value

    def transition_matrix(self, dt: Tensor | Real) -> Tensor:
        """Return ``exp(A*dt)`` for scalar dt or batched ``dt[B]``.

        Autocast is disabled around ``matrix_exp``. Float32 generators compute in
        float32; float64 reference models compute in float64.
        """
        dt_tensor = self._dt_tensor(dt)
        with torch.autocast(device_type=self.A.device.type, enabled=False):
            if dt_tensor.ndim == 0:
                scaled = self.A * dt_tensor
            else:
                scaled = self.A.unsqueeze(0) * dt_tensor[:, None, None]
            return torch.matrix_exp(scaled)

    def _validate_state(self, z: Tensor) -> None:
        if z.ndim not in {1, 2} or z.shape[-1] != self.state_dim:
            raise ValueError("z must have shape [d] or [B,d]")
        if z.device != self.A.device or z.dtype != self.A.dtype:
            raise ValueError("z device/dtype must match the Koopman generator")
        if not torch.isfinite(z).all():
            raise ValueError("z must contain only finite values")

    def step(self, z: Tensor, dt: Tensor | Real) -> Tensor:
        """Propagate one exact matrix-exponential step using column-state semantics."""
        self._validate_state(z)
        # Keep both matrix_exp and its application in the generator dtype. Without this
        # outer island, AMP can downcast matmul/einsum and the next rollout step no longer
        # satisfies the core's explicit dtype contract.
        with torch.autocast(device_type=self.A.device.type, enabled=False):
            transition = self.transition_matrix(dt)
            if z.ndim == 1:
                if transition.ndim != 2:
                    raise ValueError("single state requires scalar dt")
                return transition @ z
            if transition.ndim == 2:
                return z @ transition.transpose(-1, -2)
            if transition.shape[0] != z.shape[0]:
                raise ValueError("batch dt length must equal state batch size")
            return torch.einsum("bij,bj->bi", transition, z)

    def rollout(
        self,
        z0: Tensor,
        dts: Tensor | Real,
        horizon: int | None = None,
    ) -> Tensor:
        """Closed-loop rollout including ``z0`` as the first returned state."""
        self._validate_state(z0)
        if isinstance(dts, Real) or (isinstance(dts, Tensor) and dts.ndim == 0):
            if horizon is None or horizon < 1:
                raise ValueError("scalar dts requires a positive horizon")
            scalar = self._dt_tensor(dts)
            schedule = scalar.repeat(horizon)
        elif isinstance(dts, Tensor):
            if dts.ndim not in {1, 2}:
                raise ValueError("rollout dts must be scalar, [H], or [B,H]")
            schedule = self._dt_tensor(dts.reshape(-1)).reshape(dts.shape)
            inferred_horizon = schedule.shape[-1]
            if inferred_horizon < 1:
                raise ValueError("rollout schedule must contain at least one interval")
            if horizon is not None and horizon != inferred_horizon:
                raise ValueError("horizon must match the supplied dt schedule")
            horizon = inferred_horizon
        else:
            raise TypeError("dts must be a real scalar or torch.Tensor")

        assert horizon is not None
        if z0.ndim == 1 and schedule.ndim == 2:
            raise ValueError("single-state rollout does not accept a batched dt schedule")
        if z0.ndim == 2 and schedule.ndim == 2 and schedule.shape[0] != z0.shape[0]:
            raise ValueError("batched schedule first dimension must equal batch size")
        states = [z0]
        current = z0
        for index in range(horizon):
            current_dt = schedule[index] if schedule.ndim == 1 else schedule[:, index]
            current = self.step(current, current_dt)
            states.append(current)
        time_axis = 0 if z0.ndim == 1 else 1
        return torch.stack(states, dim=time_axis)

    def spectrum(self) -> SpectrumDiagnostics:
        """Return detached continuous-time eigenvalue diagnostics."""
        return continuous_spectrum(self.A)
