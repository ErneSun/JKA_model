from __future__ import annotations

import pytest

from jka_model.config import ArchitectureConfig, DataConfig, ProjectConfig, TrainingConfig
from jka_model.contracts import (
    BoundarySpec,
    ChannelSpec,
    DtMode,
    GeometrySpec,
    GridSpec,
    NormalizationSpec,
    ProblemSpec,
)


@pytest.fixture
def toy_problem_spec() -> ProblemSpec:
    return ProblemSpec(
        name="toy_flow",
        channels=(ChannelSpec("rho", "kg/m^3"), ChannelSpec("u", "m/s")),
        spatial_dim=2,
        grid=GridSpec(
            layout="channels_first",
            shape=(4, 5),
            spacing=(0.25, 0.2),
            coordinates_required=True,
            cell_weights_required=True,
            metadata={"axes": ["x", "y"]},
        ),
        boundary=BoundarySpec("periodic", {"axes": [0, 1]}),
        action_dim=1,
        parameter_dim=1,
        dt_mode=DtMode.CONSTANT,
        constant_dt=0.1,
        normalization=NormalizationSpec("standardize", {"fit_split": "train"}),
        geometry=GeometrySpec(mask_required=True, metadata={"meaning": "fluid"}),
        observable_requirements=("mass", "kinetic_energy"),
        metadata={"source": "test"},
    )


@pytest.fixture
def toy_config() -> ProjectConfig:
    return ProjectConfig(
        architecture=ArchitectureConfig(),
        training=TrainingConfig(seed=17),
        data=DataConfig(
            problem_name="toy_flow",
            action_dim=1,
            parameter_dim=1,
            dt_mode=DtMode.CONSTANT,
            constant_dt=0.1,
            normalization="standardize",
        ),
        tags=("test", "v0.1"),
    )

