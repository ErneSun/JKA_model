"""Project-wide compatibility constants.

These values live in one module so checkpoints and logs never depend on scattered
string literals.
"""

PROJECT_VERSION = "0.9.0"
ARCHITECTURE_REVISION = "2.2"
CHECKPOINT_SCHEMA_VERSION = 9

# Schema-5 V0.5 and schema-6 V0.6 checkpoints remain readable. V0.7 closure
# checkpoints use schema 7 and store a standalone frozen backbone plus closure.
V0_5_PROJECT_VERSION = "0.5.0"
V0_5_CHECKPOINT_SCHEMA_VERSION = 5
V0_6_PROJECT_VERSION = "0.6.0"
V0_6_CHECKPOINT_SCHEMA_VERSION = 6
V0_7_PROJECT_VERSION = "0.7.0"
V0_7_CHECKPOINT_SCHEMA_VERSION = 7
V0_8_PROJECT_VERSION = "0.8.0"
V0_8_CHECKPOINT_SCHEMA_VERSION = 8
SUPPORTED_CONFIG_PROJECT_VERSIONS = frozenset(
    {
        V0_5_PROJECT_VERSION,
        V0_6_PROJECT_VERSION,
        V0_7_PROJECT_VERSION,
        V0_8_PROJECT_VERSION,
        PROJECT_VERSION,
    }
)
