"""
LegionADK Governance Subsystem.
Provides Chain-of-Verification (CoV) tools verification, Tool-Signature Circuit Breaker,
and SmartGuardian policy aggregator.
"""

from LegionADK.governance.cov_verifier import (
    DeterministicVerifier,
    ToolVerifier,
    VerificationVerdict,
)
from LegionADK.governance.circuit_breaker import (
    CircuitState,
    ToolCircuitBreaker,
    clear_all_sessions as clear_all_circuit_breaker_sessions,
    compute_tool_signature,
    get_circuit_state,
    is_open,
    prune_idle_sessions as prune_idle_circuit_breaker_sessions,
    record_tool_outcome,
    reset_session as reset_circuit_breaker_session,
)
from LegionADK.governance.smart_guardian import (
    GovernanceEvent,
    GovernancePolicy,
    GuardianDecision,
    GuardianReport,
    SmartGuardian,
    clear_all_sessions as clear_all_guardian_sessions,
    get_session_action,
    observe_event,
    prune_idle_sessions as prune_idle_guardian_sessions,
    reset_session as reset_guardian_session,
    session_journal,
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
    "reset_circuit_breaker_session",
    "prune_idle_circuit_breaker_sessions",
    "clear_all_circuit_breaker_sessions",
    # Smart Guardian
    "GovernanceEvent",
    "GovernancePolicy",
    "GuardianReport",
    "GuardianDecision",
    "SmartGuardian",
    "observe_event",
    "session_journal",
    "get_session_action",
    "reset_guardian_session",
    "prune_idle_guardian_sessions",
    "clear_all_guardian_sessions",
]
