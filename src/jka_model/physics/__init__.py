"""Non-latent raw-state physical-law contracts and diagnostics."""

from jka_model.physics.constraints import (
    ConstraintResult,
    DiscretePDEResidualConstraint,
    FiniteValueConstraint,
    MassConservationConstraint,
    PeriodicBoundaryConstraint,
    PhysicsConstraint,
    StateAdmissibilityConstraint,
    evaluate_constraints,
)
from jka_model.physics.operators import (
    periodic_first_derivative,
    periodic_second_derivative,
    weighted_integral,
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
    "ChannelMeanProbe",
    "ChannelRMSProbe",
    "ConstraintResult",
    "DiscretePDEResidualConstraint",
    "FiniteValueConstraint",
    "MassConservationConstraint",
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
    "periodic_second_derivative",
    "register_constraint",
    "weighted_integral",
]
