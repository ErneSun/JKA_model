#!/usr/bin/env python3
"""Explain V0.7 residual and finite-history semantics from a completed sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args()
    print("V0.7 freezes the V0.6 online encoder, Koopman operator, and decoder.")
    print("Target: r_t = stopgrad(E_online(U_{t+1}) - exp(A*dt_t)E_online(U_t)).")
    print("H=1 sees only current latent z_t, next dt, and static parameters: Markovian baseline.")
    print("H>1 additionally sees ordered past latent states and their intervals.")
    print(
        "Memory is accepted only if ordered history beats matched instantaneous "
        "and shuffled controls."
    )
    if args.results_dir is None:
        return
    path = args.results_dir / "evaluation" / "history_sweep.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    print("\nH=1/H=2/H=4 concrete comparison (mean over available seeds):")
    for history in (1, 2, 4):
        selected = [
            row
            for row in rows
            if int(row["history_steps"]) == history and row["variant"] == "history"
        ]
        if not selected:
            continue
        residual = sum(float(row["residual_nrmse"]) for row in selected) / len(selected)
        rollout = sum(float(row["closed_loop_field_rmse"]) for row in selected) / len(selected)
        print(
            f"H={history}: input=current plus {history - 1} past state(s); "
            f"residual NRMSE={residual:.6g}; closed-loop field RMSE={rollout:.6g}"
        )


if __name__ == "__main__":
    main()
