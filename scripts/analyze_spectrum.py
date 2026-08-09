#!/usr/bin/env python3
"""Print detached continuous-time spectrum from a V0.3 checkpoint or toy config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from jka_model.config import load_config
from jka_model.data import damped_oscillator_generator_matrix
from jka_model.models import ContinuousKoopmanCore
from jka_model.utils import load_checkpoint


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs" / "v0_3_smoke.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    arguments = parser.parse_args()
    if arguments.checkpoint is None:
        config = load_config(arguments.config)
        if config.koopman is None or config.oscillator is None:
            raise RuntimeError("spectrum analysis requires V0.3 config sections")
        dtype = torch.float64 if config.koopman.dtype == "float64" else torch.float32
        generator = damped_oscillator_generator_matrix(
            config.oscillator.omega0, config.oscillator.gamma, dtype=dtype
        )
        core = ContinuousKoopmanCore(
            config.koopman.state_dim,
            generator=generator,
            trainable=False,
            dtype=dtype,
        )
        source = "config:true_damped_generator"
    else:
        checkpoint = load_checkpoint(arguments.checkpoint)
        if checkpoint.online_model_state is None or checkpoint.config is None:
            raise RuntimeError("checkpoint must contain model state and resolved config")
        config = checkpoint.config
        if config.koopman is None:
            raise RuntimeError("checkpoint config has no V0.3 Koopman section")
        dtype = torch.float64 if config.koopman.dtype == "float64" else torch.float32
        core = ContinuousKoopmanCore(
            config.koopman.state_dim, trainable=False, dtype=dtype
        )
        core.load_state_dict(checkpoint.online_model_state)
        source = str(arguments.checkpoint)
    spectrum = core.spectrum()
    output = {
        "source": source,
        "continuous_eigenvalues": [str(value) for value in spectrum.eigenvalues.tolist()],
        "growth_rates": spectrum.growth_rates.tolist(),
        "angular_frequencies": spectrum.angular_frequencies.tolist(),
        "frequencies_hz": spectrum.frequencies_hz.tolist(),
        "detached": all(not tensor.requires_grad for tensor in spectrum.as_tuple()),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
