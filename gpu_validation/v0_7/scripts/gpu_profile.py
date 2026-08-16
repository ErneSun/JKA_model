#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for item in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(item))

from jka_model.config import load_config  # noqa: E402
from jka_model.residual import build_closure  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    assert config.koopman and config.residual_closure and config.memory_sweep
    parameter_dim = config.data.parameter_dim if config.residual_closure.include_parameters else 0
    counts = {}
    for history in config.memory_sweep.history_lengths:
        counts[str(history)] = {}
        for variant in ("history", "instantaneous", "shuffled_history"):
            model = build_closure(
                variant,
                latent_dim=config.koopman.state_dim,
                history=history,
                parameter_dim=parameter_dim,
                hidden_dim=config.residual_closure.hidden_dim,
                depth=config.residual_closure.depth,
            )
            counts[str(history)][variant] = sum(
                parameter.numel() for parameter in model.parameters()
            )
    print(
        json.dumps({"closure_parameter_counts_by_history": counts}, indent=2, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
