"""
LegionADK Governance Subsystem.
Provides Chain-of-Verification (CoV) tools verification and Tool-Signature Circuit Breaker.
"""

from LegionADK.governance.cov_verifier import (
    DeterministicVerifier,
    ToolVerifier,
    VerificationVerdict,
)
from LegionADK.governance.circuit_breaker import (
    CircuitState,
    ToolCircuitBreaker,
    clear_all_sessions,
    compute_tool_signature,
    get_circuit_state,
    is_open,
    prune_idle_sessions,
    record_tool_outcome,
    reset_session,
)

__all__ = [
    # Chain-of-Verification (CoV)
    "VerificationVerdict",
    "ToolVerifier",
    "DeterministicVerifier",
    # Tool-Signature Circuit Breaker
    "CircuitState",
    "compute_tool_signature",
    "ToolCircuitBreaker",
    "record_tool_outcome",
    "get_circuit_state",
    "is_open",
    "reset_session",
    "prune_idle_sessions",
    "clear_all_sessions",
]
