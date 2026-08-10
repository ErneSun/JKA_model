"""Non-latent raw-state physical-law contracts and diagnostics."""

from jka_model.physics.constraints import (
    AdvectionDiffusionOperatorConstraint2D,
    ConstraintResult,
    DiscretePDEResidualConstraint,
    FiniteValueConstraint,
    MassConservation2DConstraint,
    MassConservationConstraint,
    PeriodicBoundaryConstraint,
    PhysicsConstraint,
    StateAdmissibilityConstraint,
    evaluate_constraints,
)
from jka_model.physics.operators import (
    periodic_first_derivative,
    periodic_first_derivative_2d,
    periodic_laplacian_2d,
    periodic_second_derivative,
    periodic_second_derivative_2d,
    weighted_integral,
    weighted_integral_2d,
)
from jka_model.physics.probes import (
    ChannelMeanProbe,
    ChannelRMSProbe,
    PhysicalProbe,
    evaluate_batch_probes,
    evaluate_probes,
)
from jka_model.physics.registry import (
    create_constraint,
    get_constraint_factory,
    register_constraint,
)

__all__ = [
    "AdvectionDiffusionOperatorConstraint2D",
    "ChannelMeanProbe",
    "ChannelRMSProbe",
    "ConstraintResult",
    "DiscretePDEResidualConstraint",
    "FiniteValueConstraint",
    "MassConservationConstraint",
    "MassConservation2DConstraint",
    "PeriodicBoundaryConstraint",
    "PhysicalProbe",
    "PhysicsConstraint",
    "StateAdmissibilityConstraint",
    "create_constraint",
    "evaluate_batch_probes",
    "evaluate_constraints",
    "evaluate_probes",
    "get_constraint_factory",
    "periodic_first_derivative",
    "periodic_first_derivative_2d",
    "periodic_laplacian_2d",
    "periodic_second_derivative",
    "periodic_second_derivative_2d",
    "register_constraint",
    "weighted_integral",
    "weighted_integral_2d",
]
