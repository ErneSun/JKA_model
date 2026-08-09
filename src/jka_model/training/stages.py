"""Single source of truth for future freeze/unfreeze behavior."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

from torch import nn
from torch.optim import Optimizer


class TrainStage(str, Enum):
    """Architecturally permitted training phases."""

    KOOPMAN = "koopman"
    JEPA = "jepa"
    RESIDUAL = "residual"
    JOINT = "joint"


_STAGE_TRAINABLE_GROUPS: dict[TrainStage, frozenset[str]] = {
    TrainStage.KOOPMAN: frozenset(
        {"koopman_encoder", "online_encoder", "koopman_core", "training_decoder"}
    ),
    TrainStage.JEPA: frozenset(
        {"koopman_encoder", "online_encoder", "koopman_core", "training_decoder"}
    ),
    TrainStage.RESIDUAL: frozenset({"residual_memory", "residual_head", "gate"}),
    TrainStage.JOINT: frozenset(
        {
            "koopman_encoder",
            "online_encoder",
            "koopman_core",
            "residual_memory",
            "residual_head",
            "gate",
            "training_decoder",
        }
    ),
}

_KNOWN_GROUPS = frozenset().union(*_STAGE_TRAINABLE_GROUPS.values()) | {
    "target_encoder",
}


def _discover_module_groups(model: nn.Module) -> Mapping[str, nn.Module]:
    provider = getattr(model, "train_stage_modules", None)
    if provider is not None:
        groups = provider()
        if not isinstance(groups, Mapping):
            raise TypeError("train_stage_modules() must return a mapping of group names to modules")
        return groups
    return {name: child for name, child in model.named_children() if name in _KNOWN_GROUPS}


def configure_train_stage(
    model: nn.Module,
    stage: TrainStage,
    module_groups: Mapping[str, nn.Module] | None = None,
) -> dict[str, bool]:
    """Configure all parameter trainability from a single stage policy.

    A future composite model either exposes canonical child names (for example
    ``koopman_core`` and ``residual_memory``) or implements
    ``train_stage_modules() -> Mapping[str, nn.Module]``. Every model parameter must
    belong to exactly one registered group; unowned parameters fail loudly.

    Returns a group-to-trainability mapping suitable for structured logging.
    This function does not construct or mutate an optimizer. Stage transitions must
    create a new optimizer from the resulting trainable parameters.
    """
    if not isinstance(stage, TrainStage):
        raise TypeError("stage must be a TrainStage")
    groups = dict(module_groups if module_groups is not None else _discover_module_groups(model))
    if not groups and any(True for _ in model.parameters()):
        raise ValueError("model has parameters but exposes no train-stage module groups")
    unknown_groups = set(groups) - _KNOWN_GROUPS
    if unknown_groups:
        names = ", ".join(sorted(unknown_groups))
        raise ValueError(f"unknown train-stage module group(s): {names}")
    if any(not isinstance(module, nn.Module) for module in groups.values()):
        raise TypeError("every train-stage group value must be torch.nn.Module")

    owned: dict[int, str] = {}
    for group_name, module in groups.items():
        for parameter in module.parameters():
            parameter_id = id(parameter)
            if parameter_id in owned:
                raise ValueError(
                    f"parameter is owned by multiple train-stage groups: "
                    f"{owned[parameter_id]!r} and {group_name!r}"
                )
            owned[parameter_id] = group_name
    all_parameters = {id(parameter) for parameter in model.parameters()}
    unowned = all_parameters - set(owned)
    if unowned:
        raise ValueError(f"{len(unowned)} model parameter(s) are not owned by a train-stage group")

    trainable_groups = _STAGE_TRAINABLE_GROUPS[stage]
    result: dict[str, bool] = {}
    for group_name, module in groups.items():
        trainable = group_name in trainable_groups
        module.requires_grad_(trainable)
        result[group_name] = trainable
    return result


def configure_trainable(
    model: nn.Module,
    stage: TrainStage,
    module_groups: Mapping[str, nn.Module] | None = None,
) -> dict[str, bool]:
    """Architecture-document spelling of :func:`configure_train_stage`."""
    return configure_train_stage(model, stage, module_groups)


def assert_optimizer_matches_trainable_params(model: nn.Module, optimizer: Optimizer) -> None:
    """Verify exact, unique optimizer ownership of every trainable parameter."""
    optimizer_ids: list[int] = []
    for group in optimizer.param_groups:
        optimizer_ids.extend(id(parameter) for parameter in group["params"])
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise ValueError("an optimizer parameter appears in more than one parameter group")

    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimizer_id_set = set(optimizer_ids)
    missing = trainable_ids - optimizer_id_set
    extra = optimizer_id_set - trainable_ids
    if missing or extra:
        raise ValueError(
            "optimizer parameters do not exactly match trainable model parameters "
            f"(missing={len(missing)}, frozen_or_external={len(extra)})"
        )
