#!/usr/bin/env python3
"""One-command, non-silent V0.9 validation on one CUDA GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import torch

ROOT = Path(__file__).resolve().parents[3]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from eval.evaluate_v0_9 import evaluate_v0_9  # noqa: E402
from jka_model.adaptive import aggregate_v0_9_results, audit_v0_8_handoff  # noqa: E402
from jka_model.config import ProjectConfig, load_config, save_config  # noqa: E402
from jka_model.data import (  # noqa: E402
    generate_v0_9_cylinder_wake_trajectories,
    save_cylinder_wake_dataset,
    validate_v0_9_cylinder_wake_dataset,
)
from jka_model.utils import create_versioned_session, get_git_commit  # noqa: E402
from train.prepare_v0_9 import prepare_v0_9_cache  # noqa: E402
from train.train_v0_9 import train_v0_9  # noqa: E402


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _stage(label: str, action):
    print(f"[V0.9][validation] {label}: START", flush=True)
    result = action()
    print(f"[V0.9][validation] {label}: PASS", flush=True)
    return result


def _run_tests(python: str, log_path: Path) -> None:
    command = [
        python,
        "-m",
        "pytest",
        "-q",
        "tests/test_v0_9_adaptive.py",
        "tests/test_v0_9_physical_problem.py",
        "tests/test_v0_9_workflow.py",
    ]
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"targeted V0.9 tests failed with exit code {code}")


def _latest_ready_v0_8() -> str:
    candidates: list[tuple[float, str]] = []
    root = ROOT / "gpu_validation" / "v0_8" / "results"
    for result in root.iterdir() if root.is_dir() else ():
        decision_path = result / "evaluation" / "v0_8_scientific_decision.json"
        completion_path = result / "completion.json"
        raw = ROOT / "runs" / "v0_8" / result.name
        if not decision_path.is_file() or not completion_path.is_file() or not raw.is_dir():
            continue
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if bool(decision.get("v0_9_ready")) and completion.get("status") == "PASS":
            candidates.append((completion_path.stat().st_mtime, result.name))
    if not candidates:
        raise ValueError("no strict-ready V0.8 raw+compact handoff found; pass --v0-8-id")
    return max(candidates)[1]


def _resolved(
    template: ProjectConfig,
    *,
    seed: int,
    dataset_path: Path,
    run_root: Path,
    condition_mode: str,
    rank: int,
    operator_seed: int,
) -> ProjectConfig:
    payload = template.to_dict()
    payload["training"].update(
        {"seed": seed, "stage": "adaptive", "run_root": str(run_root.resolve())}
    )
    payload["data"]["split"]["seed"] = seed
    payload["cylinder_wake_2d"]["dataset_path"] = str(dataset_path.resolve())
    payload["v0_9_adaptive"].update({"condition_mode": condition_mode, "rank": rank})
    payload["v0_9_training"]["operator_initialization_seed"] = operator_seed
    payload["tags"] = [
        "v0.9",
        "gpu-formal",
        "cylinder-wake-controlled-inlet",
        f"backbone-seed-{seed}",
        f"operator-seed-{operator_seed}",
        f"condition-{condition_mode}",
        f"rank-{rank}",
    ]
    return ProjectConfig.from_dict(payload)


def validate_completion_payload(payload: dict[str, Any]) -> None:
    required = {
        "requested_validation_id",
        "resolved_validation_id",
        "v0_8_validation_id",
        "status",
        "git_commit",
        "all_required_stages_completed",
        "selected_rank",
        "formal_training_run_count",
        "formal_evaluation_run_count",
        "scientific_status",
        "v1_0_readiness",
        "output_paths",
    }
    if required - set(payload) or payload["status"] != "PASS":
        raise ValueError("V0.9 completion payload is incomplete")
    if not payload["all_required_stages_completed"]:
        raise ValueError("V0.9 completion cannot claim PASS with incomplete stages")
    if payload["formal_training_run_count"] != 18 or payload["formal_evaluation_run_count"] != 18:
        raise ValueError("V0.9 formal nested matrix must contain 18 train/evaluation runs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-id",
        default=datetime.now(timezone.utc).strftime("v09-%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--v0-8-id", default="")
    parser.add_argument("--seeds", nargs="+", type=int, default=[47, 53, 59])
    parser.add_argument("--skip-software-tests", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        raise SystemExit("V0.9 formal validation requires exactly three unique seeds")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    if dirty and not args.allow_dirty:
        raise SystemExit("formal V0.9 validation requires a clean git working tree")
    results_root = ROOT / "gpu_validation" / "v0_9" / "results"
    session = create_versioned_session(
        ROOT / "runs" / "v0_9", args.validation_id, reserved_roots=(results_root,)
    )
    raw = session.path
    compact = results_root / session.resolved_id
    compact.mkdir(parents=True, exist_ok=False)
    for name in ("software", "configs", "data", "seeds", "rank_sweep", "logs"):
        (raw / name).mkdir()
    workflow_stream = (raw / "logs" / "workflow.log").open("w", encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_stdout, workflow_stream)  # type: ignore[assignment]
    sys.stderr = _Tee(original_stderr, workflow_stream)  # type: ignore[assignment]
    current_stage = "session_initialization"
    current_run: dict[str, Any] | None = None
    completed = {"rank_sweep": 0, "formal_training": 0, "formal_evaluation": 0}
    last_checkpoint: str | None = None
    try:
        print(
            f"[V0.9][validation] SESSION requested={session.requested_id} "
            f"resolved={session.resolved_id} gpu={torch.cuda.get_device_name(0)}",
            flush=True,
        )
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("formal RTX-5080 workflow requires CUDA BF16 support")
        python = str(ROOT / ".venv" / "bin" / "python")
        if not Path(python).is_file():
            python = sys.executable
        if not args.skip_software_tests:
            current_stage = "targeted_software_tests"
            _stage(
                "G0 targeted software tests",
                lambda: _run_tests(python, raw / "software" / "pytest.log"),
            )

        current_stage = "v0_8_strict_handoff"
        v08_id = args.v0_8_id or _latest_ready_v0_8()
        handoff = _stage(
            f"G1 strict V0.8 handoff id={v08_id}",
            lambda: audit_v0_8_handoff(
                v08_id,
                runs_root=ROOT / "runs" / "v0_8",
                results_root=ROOT / "gpu_validation" / "v0_8" / "results",
            ),
        )
        if {item.backbone_seed for item in handoff.seeds} != set(args.seeds):
            raise ValueError("requested V0.9 seeds do not match the strict V0.8 handoff")
        handoff_payload = {
            "validation_id": handoff.validation_id,
            "raw_run": str(handoff.raw_run),
            "compact_result": str(handoff.compact_result),
            "route": handoff.route,
            "context_family": handoff.context_family,
            "seeds": [
                {
                    **asdict(item),
                    "backbone_checkpoint": str(item.backbone_checkpoint),
                    "context_checkpoint": str(item.context_checkpoint),
                }
                for item in handoff.seeds
            ],
        }
        (raw / "v0_8_handoff_audit.json").write_text(
            json.dumps(handoff_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        template = load_config(
            ROOT / "gpu_validation" / "v0_9" / "configs" / "gpu_adaptive_koopman.yaml"
        )
        assert template.cylinder_wake_2d and template.v0_9_condition
        artifacts: dict[int, dict[str, Path]] = {}
        current_stage = "controlled_physical_data"
        for item in handoff.seeds:
            seed = item.backbone_seed
            current_run = {"seed": seed, "kind": "controlled_physical_data"}
            data = _stage(
                f"G2 controlled cylinder data seed={seed}",
                lambda seed=seed: generate_v0_9_cylinder_wake_trajectories(
                    template.cylinder_wake_2d,
                    template.v0_9_condition,
                    seed=seed,
                    device="cuda",
                ),
            )
            acceptance = validate_v0_9_cylinder_wake_dataset(
                data, template.cylinder_wake_2d, template.v0_9_condition
            )
            if acceptance["status"] != "PASS":
                raise RuntimeError(f"V0.9 physical acceptance failed for seed {seed}")
            dataset_path = raw / "data" / f"controlled_cylinder_seed_{seed}.pt"
            save_cylinder_wake_dataset(data, template.cylinder_wake_2d, dataset_path)
            (raw / "data" / f"physical_acceptance_seed_{seed}.json").write_text(
                json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            base_config = _resolved(
                template,
                seed=seed,
                dataset_path=dataset_path,
                run_root=raw,
                condition_mode="latent_inferred",
                rank=template.v0_9_adaptive.rank,  # type: ignore[union-attr]
                operator_seed=template.v0_9_training.operator_initialization_seed,  # type: ignore[union-attr]
            )
            cache_path = raw / "seeds" / f"seed_{seed}" / "cache" / "adaptive_cache.pt"
            _stage(
                f"G3 adaptive latent cache seed={seed}",
                lambda base_config=base_config, item=item, dataset_path=dataset_path,
                cache_path=cache_path: prepare_v0_9_cache(
                    base_config,
                    backbone_checkpoint=item.backbone_checkpoint,
                    context_checkpoint=item.context_checkpoint,
                    physical_dataset=dataset_path,
                    destination=cache_path,
                    device="cuda",
                ),
            )
            artifacts[seed] = {"dataset": dataset_path, "cache": cache_path}

        current_stage = "validation_rank_sweep"
        first = handoff.seeds[0]
        assert template.v0_9_adaptive and template.v0_9_evaluation
        sweep_scores: dict[int, list[float]] = {
            rank: [] for rank in template.v0_9_adaptive.rank_candidates
        }
        first_operator_seed = template.v0_9_evaluation.operator_initialization_seeds[0]
        for rank in template.v0_9_adaptive.rank_candidates:
            for mode in ("known", "latent_inferred"):
                current_run = {"kind": "rank_sweep", "rank": rank, "condition_mode": mode}
                config = _resolved(
                    template,
                    seed=first.backbone_seed,
                    dataset_path=artifacts[first.backbone_seed]["dataset"],
                    run_root=raw,
                    condition_mode=mode,
                    rank=rank,
                    operator_seed=first_operator_seed,
                )
                run_dir = raw / "rank_sweep" / f"rank_{rank}" / mode
                trained = _stage(
                    f"G4 rank sweep r={rank} mode={mode}",
                    lambda config=config, run_dir=run_dir: train_v0_9(
                        config,
                        context_checkpoint=first.context_checkpoint,
                        adaptive_cache=artifacts[first.backbone_seed]["cache"],
                        run_dir=run_dir,
                        device="cuda",
                    ),
                )
                save_config(config, raw / "configs" / f"rank_{rank}_{mode}.yaml")
                sweep_scores[rank].append(trained.validation_metrics["total"])
                last_checkpoint = str(trained.best_checkpoint)
                completed["rank_sweep"] += 1
        mean_scores = {rank: sum(values) / len(values) for rank, values in sweep_scores.items()}
        minimum = min(mean_scores.values())
        eligible = [rank for rank, score in mean_scores.items() if score <= minimum * 1.02]
        selected_rank = min(eligible)
        rank_selection = {
            "selection_split": "validation",
            "test_opened": False,
            "scores": mean_scores,
            "near_optimal_tolerance": 0.02,
            "selected_rank": selected_rank,
            "selection_rule": (
                "smallest rank within 2% of minimum mean known/latent validation total"
            ),
        }
        (raw / "rank_selection.json").write_text(
            json.dumps(rank_selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        current_stage = "formal_nested_training_and_locked_test"
        for item in handoff.seeds:
            seed = item.backbone_seed
            for mode in ("known", "latent_inferred"):
                for operator_seed in template.v0_9_evaluation.operator_initialization_seeds:
                    current_run = {
                        "seed": seed,
                        "condition_mode": mode,
                        "operator_seed": operator_seed,
                        "rank": selected_rank,
                    }
                    config = _resolved(
                        template,
                        seed=seed,
                        dataset_path=artifacts[seed]["dataset"],
                        run_root=raw,
                        condition_mode=mode,
                        rank=selected_rank,
                        operator_seed=operator_seed,
                    )
                    run_dir = (
                        raw
                        / "seeds"
                        / f"seed_{seed}"
                        / "formal"
                        / mode
                        / f"init_{operator_seed}"
                    )
                    trained = _stage(
                        f"G5 train seed={seed} mode={mode} init={operator_seed}",
                        lambda config=config, run_dir=run_dir, item=item, seed=seed: train_v0_9(
                            config,
                            context_checkpoint=item.context_checkpoint,
                            adaptive_cache=artifacts[seed]["cache"],
                            run_dir=run_dir,
                            device="cuda",
                        ),
                    )
                    completed["formal_training"] += 1
                    last_checkpoint = str(trained.best_checkpoint)
                    _stage(
                        f"G6 locked test seed={seed} mode={mode} init={operator_seed}",
                        lambda config=config, run_dir=run_dir, item=item, seed=seed,
                        trained=trained: evaluate_v0_9(
                            config,
                            checkpoint=trained.best_checkpoint,
                            context_checkpoint=item.context_checkpoint,
                            adaptive_cache=artifacts[seed]["cache"],
                            backbone_checkpoint=item.backbone_checkpoint,
                            physical_dataset=artifacts[seed]["dataset"],
                            output_dir=run_dir,
                            device="cuda",
                        ),
                    )
                    completed["formal_evaluation"] += 1
                    save_config(
                        config,
                        raw
                        / "configs"
                        / f"seed_{seed}_{mode}_init_{operator_seed}.yaml",
                    )

        current_stage = "nested_aggregation"
        decision = _stage(
            "G7 nested-seed aggregation and compact report",
            lambda: aggregate_v0_9_results(raw, compact),
        )
        if not bool(decision.get("compact_audit", {}).get("complete")):
            raise RuntimeError("V0.9 compact audit is incomplete")
        completion = {
            "requested_validation_id": session.requested_id,
            "resolved_validation_id": session.resolved_id,
            "v0_8_validation_id": v08_id,
            "status": "PASS",
            "git_commit": get_git_commit(ROOT),
            "all_required_stages_completed": True,
            "selected_rank": selected_rank,
            "rank_sweep_run_count": completed["rank_sweep"],
            "formal_training_run_count": completed["formal_training"],
            "formal_evaluation_run_count": completed["formal_evaluation"],
            "scientific_status": decision["low_rank_operator_adaptation"],
            "v1_0_readiness": decision["v1_0_readiness"],
            "output_paths": {
                "raw_run": str(raw),
                "compact_result": str(compact),
                "report": str(compact / "report.md"),
            },
        }
        validate_completion_payload(completion)
        for destination in (raw / "completion.json", compact / "completion.json"):
            destination.write_text(
                json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(
            f"V0.9 VALIDATION COMPLETE status=PASS id={session.resolved_id} "
            f"science={decision['low_rank_operator_adaptation']} "
            f"report={compact / 'report.md'}",
            flush=True,
        )
    except BaseException as error:
        failure = {
            "validation_id": session.resolved_id,
            "status": "FAILED_INCOMPLETE",
            "failed_stage": current_stage,
            "failed_run": current_run,
            "exception_summary": f"{type(error).__name__}: {error}",
            "completed_work": completed,
            "last_valid_checkpoint": last_checkpoint,
            "git_commit": get_git_commit(ROOT),
            "traceback": traceback.format_exc(),
        }
        for destination in (raw / "failure.json", compact / "failure.json"):
            destination.write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(
            f"V0.9 VALIDATION FAILED id={session.resolved_id} "
            f"report={compact / 'failure.json'}",
            flush=True,
        )
        raise
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        workflow_stream.close()


if __name__ == "__main__":
    main()
