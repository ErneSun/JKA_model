"""Checkpoint-only Stage-1 bottleneck audit; no optimizer, fitting of networks or gates."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from jka_model.adaptive.cache import load_adaptive_cache
from jka_model.adaptive.objectives import Phase2TrainingState, differentiable_adaptive_rollout
from jka_model.config import ProjectConfig, stable_config_hash
from jka_model.data import data_fingerprint, load_cylinder_wake_dataset
from jka_model.manifold.bottleneck import (
    diagnostic_priority,
    error_decomposition,
    evaluate_gauge,
    field_views,
    fit_gauge,
    latent_spectrum,
    summarize_errors,
    transition_contract,
)
from jka_model.manifold.joint import RawFieldAdaptiveRolloutDataset
from jka_model.residual.cache import file_sha256
from jka_model.utils import set_global_seed
from train.train_v0_9_phase3 import _build_phase3_models


def relocate_run_path(path: str | Path, root: Path) -> Path:
    """Relocate only the recorded runs/ suffix, never search for similarly named files."""
    parts = Path(path).parts
    if "runs" not in parts:
        raise ValueError(f"source is outside the versioned runs tree: {path}")
    suffix = parts[parts.index("runs") :]
    if ".." in suffix:
        raise ValueError("source run path must not traverse parents")
    result = root.joinpath(*suffix).resolve()
    if not result.is_relative_to((root / "runs").resolve()):
        raise ValueError("source path escapes runs tree")
    if not result.is_file():
        raise FileNotFoundError(f"restore this server artifact (no retraining needed): {result}")
    return result


def validate_route_payload(payload: dict[str, Any], row: dict[str, Any]) -> ProjectConfig:
    if payload.get("schema_version") not in {"v0.9-phase3-route-1", "v0.9-phase3-route-2"}:
        raise ValueError("unsupported Phase-3 checkpoint schema")
    if payload.get("route") != "joint" or row.get("route") != "joint":
        raise ValueError("Stage-1 audit currently requires the matched joint source matrix")
    # Validate the stored representation BEFORE runtime defaults are added. This is
    # not a hash bypass and does not mutate historical checkpoint dictionaries.
    if stable_config_hash(payload["config"]) != payload.get("config_hash"):
        raise ValueError("source checkpoint raw config hash mismatch")
    config = ProjectConfig.from_dict(payload["config"])
    if (
        config.training.seed != row["seed"]
        or config.v0_9_training.operator_initialization_seed != row["operator_seed"]
        or config.v0_9_adaptive.condition_mode != row["condition_mode"]
    ):
        raise ValueError("source summary/checkpoint cell mismatch")
    return config


def replay_state(config: ProjectConfig, payload: dict[str, Any]) -> Phase2TrainingState:
    """Replay INITIAL admission used during inference, never the later locked-test verdict."""
    phase3, phase2 = config.v0_9_phase3, config.v0_9_phase2
    if phase3 is None or phase2 is None:
        raise ValueError("missing Phase-3 replay configuration")
    latent = config.v0_9_adaptive.condition_mode == "latent_inferred"
    admission = payload.get("observer_admission")
    if phase3.observer_admission_enabled and latent:
        if not isinstance(admission, dict) or not isinstance(admission.get("admitted"), bool):
            raise ValueError("missing initial observer admission; cannot safely replay")
    fallback = phase3.observer_admission_enabled and latent and not admission["admitted"]
    return Phase2TrainingState(
        name="phase3_joint_refinement",
        active_components="dynamic" if fallback else "full",
        train_component="dynamic" if fallback else "full",
        use_oracle_condition=False,
        detach_static=False,
        observer_only=False,
        observer_weight=0.0 if phase3.observer_admission_enabled else 1.0,
        delta_budget=phase2.symmetric_delta_budget,
        condition_admitted=not fallback,
    )


def _same_state(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[k].cpu(), right[k].cpu()) for k in left
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


@torch.no_grad()
def audit_checkpoint(
    *,
    root: Path,
    row: dict[str, Any],
    phase2_id: str,
    handoff: dict[str, Any],
    output: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """All inherited-stride windows, identical origins at every reported horizon.

    TRAIN encodings fit only a diagnostic alignment. VALIDATION picks a repair
    priority. TEST remains descriptive; no checkpoint or scientific gate changes.
    """
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = relocate_run_path(row["checkpoint"], root)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = validate_route_payload(payload, row)
    set_global_seed(
        config.v0_9_training.operator_initialization_seed,
        deterministic=config.training.deterministic,
    )
    _write_json(output / "checkpoint_config.json", payload["config"])
    _write_json(output / "runtime_config.json", config.to_dict())
    selected = torch.device(device)
    phase3, cylinder, training = config.v0_9_phase3, config.cylinder_wake_2d, config.v0_9_training
    if any(x is None for x in (phase3, cylinder, training)):
        raise ValueError("incomplete audit configuration")
    for section in ("validation", "locked_test"):
        for horizon in training.active_observable_horizons:
            key = f"decoded_field_relative_l2_h{horizon}"
            value = row[section].get(key)
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"missing/nonfinite source replay metric: {section}/{key}")
    seed = row["seed"]
    phase2 = root / "runs" / "v0_9" / phase2_id
    dataset_path = phase2 / "data" / f"controlled_cylinder_seed_{seed}.pt"
    cache_path = phase2 / "seeds" / f"seed_{seed}" / "cache" / "adaptive_cache.pt"
    context_path = relocate_run_path(handoff["context_checkpoint"], root)
    backbone_path = relocate_run_path(handoff["backbone_checkpoint"], root)
    cache = load_adaptive_cache(cache_path)
    for key, expected in (
        ("source_context_sha256", file_sha256(context_path)),
        ("source_backbone_sha256", file_sha256(backbone_path)),
        ("adaptive_cache_fingerprint", cache.fingerprint),
    ):
        if payload.get(key) != expected:
            raise ValueError(f"audit provenance mismatch: {key}")
    dataset = load_cylinder_wake_dataset(dataset_path, cylinder)
    if data_fingerprint(dataset.records, dataset.problem_spec) != cache.data_fingerprint:
        raise ValueError("raw dataset/cache fingerprint mismatch")
    backbone, reference_encoder, reference_decoder, adaptive, normalizer, context = (
        _build_phase3_models(
            config,
            cache=cache,
            context_checkpoint=context_path,
            backbone_checkpoint=backbone_path,
            device=selected,
            route="joint",
        )
    )
    if not _same_state(reference_encoder.state_dict(), payload["reference_encoder_state"]):
        raise ValueError("checkpoint reference encoder differs from inherited backbone")
    if not _same_state(reference_decoder.state_dict(), payload["reference_decoder_state"]):
        raise ValueError("checkpoint reference decoder differs from inherited backbone")
    if not normalizer.matches_state_dict(payload["normalizer_state"]):
        raise ValueError("checkpoint normalizer differs from inherited backbone")
    inherited_teacher = {
        k: v.cpu().clone() for k, v in backbone.target_encoder.state_dict().items()
    }
    backbone.load_state_dict(payload["backbone_state"], strict=True)
    adaptive.load_state_dict(payload["adaptive_state"], strict=True)
    if not _same_state(inherited_teacher, backbone.target_encoder.state_dict()):
        raise ValueError(
            "joint teacher is no longer frozen; a different audit contract is required"
        )
    if not torch.equal(adaptive.operator_adapter.nominal_generator.cpu(), cache.nominal_generator):
        raise ValueError("joint nominal generator differs from frozen source")
    for model in (backbone, adaptive, reference_encoder, reference_decoder):
        model.eval().requires_grad_(False)
    state = replay_state(config, payload)
    mean, std = (payload[k].to(selected) for k in ("condition_mean", "condition_std"))
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
        raise ValueError("invalid stored condition normalization")
    horizons = sorted({1, *training.rollout_horizons, *training.active_observable_horizons})
    maximum = max(horizons)
    history = int(context["history_length_steps"])
    record_by_id = {record.trajectory_id: record for record in dataset.records}
    split_ids = {
        split: {t.trajectory_id for t in cache.select(split)}
        for split in ("train", "validation", "test")
    }
    if (
        split_ids["train"] & split_ids["validation"]
        or split_ids["train"] & split_ids["test"]
        or split_ids["validation"] & split_ids["test"]
    ):
        raise ValueError("audit train/validation/test trajectory leakage")
    encoded: dict[str, dict[str, torch.Tensor]] = {}
    # Chunked inference, not a new learned cache. Future encodings are diagnostics only.
    for trajectory in cache.trajectories:
        record = record_by_id[trajectory.trajectory_id]
        chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
        for raw in record.states_raw.split(8):
            x = normalizer.transform(raw.to(selected))
            for name, encoder in (
                ("online", backbone.encode),
                ("teacher", backbone.encode_target),
                ("inherited_online", reference_encoder),
            ):
                z = encoder(x).cpu()
                if not bool(torch.isfinite(z).all()):
                    raise ValueError("nonfinite audit encoding")
                chunks[name].append(z)
        encoded[trajectory.trajectory_id] = {name: torch.cat(z) for name, z in chunks.items()}

    def latent_split(split: str, name: str) -> torch.Tensor:
        return torch.cat([encoded[t.trajectory_id][name] for t in cache.select(split)])

    try:
        gauge = fit_gauge(
            latent_split("train", "online"), latent_split("train", "inherited_online")
        )
    except ValueError as error:
        # Constant representations are a diagnostic result, not a reason to lose field audit.
        gauge = None
        gauge_unavailable = str(error)
    gauge_report = {
        split: (
            evaluate_gauge(
                gauge,
                latent_split(split, "online"),
                latent_split(split, "inherited_online"),
                cache.nominal_generator,
            )
            if gauge is not None
            else {"unavailable": gauge_unavailable}
        )
        for split in ("validation", "test")
    }
    spectra = {
        split: {
            name: latent_spectrum(latent_split(split, name))
            for name in ("online", "teacher", "inherited_online")
        }
        for split in ("train", "validation", "test")
    }
    all_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    source_metrics: dict[str, list[float]] = defaultdict(list)
    with (output / "window_metrics.jsonl").open("w") as stream:
        for split in ("validation", "test"):
            windows = RawFieldAdaptiveRolloutDataset(
                cache,
                dataset.records,
                split,
                history,
                maximum,
                stride=phase3.raw_field_rollout_stride,
            )
            for index in range(len(windows)):
                batch = windows[index]
                trajectory, _, origin = windows.items[index]
                identifier = trajectory.trajectory_id
                _write_json(
                    output / "active_window.json",
                    {
                        "split": split,
                        "trajectory_id": identifier,
                        "origin": origin,
                        "window_index": index,
                        "split_window_count": len(windows),
                    },
                )
                z = encoded[identifier]
                conditions = (
                    (batch["future_condition_targets"].to(selected) - mean) / std
                    if row["condition_mode"] == "known"
                    else None
                )
                rollout = differentiable_adaptive_rollout(
                    adaptive,
                    z["online"][origin - history + 1 : origin + 1].to(selected)[None],
                    batch["history_dts"].to(selected)[None],
                    batch["future_dts"].to(selected)[None],
                    batch["context_parameters"].to(selected)[None],
                    None if conditions is None else conditions[None],
                    phase2_state=state,
                )
                manifest = {
                    "split": split,
                    "trajectory_id": identifier,
                    "origin": origin,
                    "horizons": horizons,
                    "schedule_type": trajectory.schedule_type,
                    "input": transition_contract(trajectory.conditions, origin, maximum),
                }
                manifests.append(manifest)
                for horizon in horizons:
                    mask = batch["valid_mask"].to(selected)
                    truth = batch["target_raw"][horizon - 1].to(selected)
                    truth_views = field_views(truth, mask, dx=cylinder.dx, dy=cylinder.dy)
                    refs = {
                        name: field_views(
                            normalizer.inverse_transform(
                                backbone.decode(z[name][origin + horizon].to(selected)[None])
                            )[0],
                            mask,
                            dx=cylinder.dx,
                            dy=cylinder.dy,
                        )
                        for name in ("online", "teacher")
                    }
                    inherited = field_views(
                        normalizer.inverse_transform(
                            reference_decoder(
                                z["inherited_online"][origin + horizon].to(selected)[None]
                            )
                        )[0],
                        mask,
                        dx=cylinder.dx,
                        dy=cylinder.dy,
                    )
                    input_info = transition_contract(trajectory.conditions, origin, horizon)
                    input_group = (
                        "changing_input"
                        if input_info["changed_at_origin"] or input_info["changes_after_origin"]
                        else "unchanged_input"
                    )
                    for predictor, key in (
                        ("adaptive", "adapted"),
                        ("same_start_nominal", "nominal"),
                    ):
                        predicted = normalizer.inverse_transform(
                            backbone.decode(rollout[key][:, horizon - 1])
                        )[0]
                        views = field_views(predicted, mask, dx=cylinder.dx, dy=cylinder.dy)
                        # Match the inherited field metric exactly (float32 norm floor).
                        if predictor == "adaptive":
                            source_metrics[f"{split}:decoded_field_relative_l2_h{horizon}"].append(
                                float(
                                    (predicted - truth).flatten().norm()
                                    / truth.flatten().norm().clamp_min(1e-12)
                                )
                            )
                        for reference, rviews in refs.items():
                            for channel, y in truth_views.items():
                                terms = error_decomposition(views[channel], rviews[channel], y)
                                base = {
                                    "split": split,
                                    "trajectory_id": identifier,
                                    "origin": origin,
                                    "horizon": horizon,
                                    "channel": channel,
                                    "predictor": predictor,
                                    "reference": reference,
                                    "input_group": "all",
                                    **terms,
                                }
                                all_rows.extend((base, {**base, "input_group": input_group}))
                                stream.write(
                                    json.dumps(
                                        {**base, "actual_input_group": input_group}, allow_nan=False
                                    )
                                    + "\n"
                                )
                    # Frozen inherited autoencoder is a reconstruction control, NOT the
                    # earlier trained frozen-adaptive predictor used in scientific gates.
                    for channel, y in truth_views.items():
                        base = {
                            "split": split,
                            "trajectory_id": identifier,
                            "origin": origin,
                            "horizon": horizon,
                            "channel": channel,
                            "predictor": "inherited_reconstruction_only",
                            "reference": "inherited_online",
                            "input_group": "all",
                            **error_decomposition(inherited[channel], inherited[channel], y),
                        }
                        all_rows.append(base)
                        stream.write(json.dumps(base, allow_nan=False) + "\n")
            print(f"[V0.9][bottleneck] {split} replay: PASS windows={len(windows)}", flush=True)
    replay = {}
    for key, values in source_metrics.items():
        split, metric = key.split(":")
        source = row["validation" if split == "validation" else "locked_test"].get(metric)
        actual = sum(values) / len(values)
        replay[key] = {
            "source": source,
            "replayed": actual,
            "absolute_difference": None if source is None else abs(actual - source),
        }
        # H1 is new. Others verify actual saved predictor replay, not just algebra.
        if source is not None and abs(actual - source) > 2e-5 + 2e-4 * abs(source):
            _write_json(output / "replay_check.json", replay)
            raise ValueError(f"checkpoint replay disagrees with source metric: {key}")
    summaries = summarize_errors(all_rows)
    priorities = [
        dict(
            horizon=r["horizon"],
            channel=r["channel"],
            reference=r["reference"],
            priority=diagnostic_priority(r),
        )
        for r in summaries
        if r["split"] == "validation"
        and r["predictor"] == "adaptive"
        and r["input_group"] == "all"
        and r["channel"] in {"field_all", "velocity_fluid", "pressure_fluid"}
    ]
    result = {
        "cell": {k: row[k] for k in ("seed", "condition_mode", "operator_seed", "route")},
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_config_hash": payload["config_hash"],
        "runtime_config_hash": config.stable_hash,
        "source_git_commit": payload.get("git_commit"),
        "source_backbone_sha256": payload["source_backbone_sha256"],
        "source_context_sha256": payload["source_context_sha256"],
        "adaptive_cache_fingerprint": cache.fingerprint,
        "data_fingerprint": cache.data_fingerprint,
        "device": str(selected),
        "dtype": "float32 inference / float64 diagnostics",
        "deterministic": config.training.deterministic,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "condition_route": state.active_components,
        "condition_admitted": state.condition_admitted,
        "input_permission": "future_transition_schedule"
        if row["condition_mode"] == "known"
        else "observed_history_only; no future input or future true states",
        "horizons": horizons,
        "window_count": len(manifests),
        "summary": summaries,
        "spectra": spectra,
        "gauge": gauge_report,
        "source_replay": replay,
        "validation_priorities": priorities,
        "scientific_acceptance": "NOT_EVALUATED",
    }
    _write_json(output / "window_manifest.json", manifests)
    _write_json(
        output / "gauge_fit_manifest.json",
        {
            "fit_split": "train",
            "state_sampling": "every original state exactly once per trajectory",
            "trajectory_ids": {split: sorted(ids) for split, ids in split_ids.items()},
        },
    )
    if gauge is not None:
        _write_json(output / "gauge_fit.json", {k: v.tolist() for k, v in gauge.items()})
    _write_json(output / "audit.json", result)
    return result


def write_audit_report(destination: Path, results: list[dict[str, Any]], *, source_id: str) -> None:
    """Always separate stage completion from scientific support; do not pool 18 cells."""
    lines = [
        "# V0.9 Stage-1 same-sample bottleneck audit",
        "",
        f"Source: `{source_id}`. Read-only inference; no retraining or changed gates.",
        "",
        "## Interpretation contract",
        "",
        "`total MSE = propagation MSE + reference-decode MSE + signed cross term`.",
        "Online reference uses D(E(x)); teacher reference uses D(E_EMA(x)). The latter is "
        "cross-decoding, not an autoencoder error floor. Neither is a universal lower bound.",
        "Raw fields use the original nondimensional channel units. Pressure-demeaned and "
        "fluid/interior metrics are additional diagnostics, not replacement acceptance metrics.",
        "Means: windows within trajectory, then equal trajectories; seeds/initializations stay "
        "separate. Source replay checks use the original equal-window metric.",
        "",
        "## Validation-only repair priorities",
        "",
        "Heuristic: one component >2x the other, unless |cross| >0.5*(propagation+decode). "
        "This labels investigation priorities, not causes proven by intervention. Disagreement "
        "between references, channels or seeds means MIXED/UNRESOLVED; do not majority-vote "
        "it into scientific support.",
        "",
        "| Seed | Mode | Init | H | Channel | Online priority | Teacher priority |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in results:
        cell = run["cell"]
        indexed = {
            (p["horizon"], p["channel"], p["reference"]): p["priority"]
            for p in run["validation_priorities"]
        }
        for horizon in run["horizons"]:
            for channel in ("field_all", "velocity_fluid", "pressure_fluid"):
                lines.append(
                    f"| {cell['seed']} | {cell['condition_mode']} | {cell['operator_seed']} "
                    f"| {horizon} | {channel} | {indexed[horizon, channel, 'online']} "
                    f"| {indexed[horizon, channel, 'teacher']} |"
                )
    lines += [
        "",
        "## Numerical decomposition at the longest horizon (validation)",
        "",
        "All entries are MSE in the declared raw nondimensional channel metric. "
        "The signed cross term must be retained when comparing component sizes.",
        "",
        "| Seed | Mode | Init | Channel | Reference | Total | Propagation | Decode | Cross |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run in results:
        cell = run["cell"]
        for r in run["summary"]:
            if (
                r["split"] == "validation"
                and r["horizon"] == max(run["horizons"])
                and r["predictor"] == "adaptive"
                and r["input_group"] == "all"
                and r["channel"] in {"field_all", "velocity_fluid", "pressure_fluid"}
            ):
                lines.append(
                    f"| {cell['seed']} | {cell['condition_mode']} | {cell['operator_seed']} "
                    f"| {r['channel']} | {r['reference']} | {r['total_mse']:.6g} "
                    f"| {r['propagation_mse']:.6g} | {r['reference_decode_mse']:.6g} "
                    f"| {r['cross_term']:.6g} |"
                )
    lines += [
        "",
        "## Next-stage decision boundaries",
        "",
        "- Decode-dominated: first isolate representation/decoder repair; do not enlarge "
        "the adaptive operator to hide reconstruction error.",
        "- Propagation-dominated: compare fixed refit versus input-conditioned propagation "
        "on the same frozen representation and physical targets.",
        "- Strong cross terms or inconsistent references: joint error geometry is unresolved; "
        "do not assert a decoder floor or a unique bottleneck.",
        "- Changing-input subsets diagnose timing sensitivity only. They do not establish "
        "unidentifiability; known future forcing and history-only forecasting have different "
        "information permissions. A future paired same-history/different-input experiment "
        "would be needed for a constructive identifiability claim.",
        "- Gauge is fitted on TRAIN, evaluated on validation/test, with a train-selected 99% "
        "energy subspace. Data-action commutators are diagnostics, not relaxed old gates.",
        "",
        "TEST is descriptive and previously inspected, not fresh independent validation. "
        "Any next-stage tuning uses TRAIN/VALIDATION only; later confirmatory claims need "
        "a fresh locked test set. This audit cannot declare V0.9 supported or V1.0 ready.",
        "",
        "Full numerical decompositions, spectra, gauge and source-replay checks: `audit.json`. "
        "Per-window rows/manifests are under the corresponding `runs/v0_9/<id>/cells/`.",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")
