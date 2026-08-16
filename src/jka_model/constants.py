"""Project-wide compatibility constants.

These values live in one module so checkpoints and logs never depend on scattered
string literals.
"""

PROJECT_VERSION = "0.6.0"
ARCHITECTURE_REVISION = "2.2"
CHECKPOINT_SCHEMA_VERSION = 6

# Schema-5 V0.5 checkpoints remain readable and may initialize V0.6. A V0.6 resume
# still requires a JEPA-stage schema-6 checkpoint with explicit target/EMA state.
V0_5_PROJECT_VERSION = "0.5.0"
V0_5_CHECKPOINT_SCHEMA_VERSION = 5
SUPPORTED_CONFIG_PROJECT_VERSIONS = frozenset({V0_5_PROJECT_VERSION, PROJECT_VERSION})
