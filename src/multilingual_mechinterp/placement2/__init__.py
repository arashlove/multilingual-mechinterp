"""Placement 2: multilingual Moral Machine STAY/SWERVE study.

Separate from the Placement 1 modules, which stay untouched.
"""

from multilingual_mechinterp.placement2.experiment import (
    APPROVAL_REQUIRED,
    STAGE_STATES,
    STAGES,
    RunContext,
    load_placement2_config,
    pending_approvals,
)

__all__ = [
    "APPROVAL_REQUIRED",
    "STAGES",
    "STAGE_STATES",
    "RunContext",
    "load_placement2_config",
    "pending_approvals",
]
