"""
Runtime module for LegionADK.
Provides turn compaction, circuit breaker, and agent execution controls.
"""

from LegionADK.runtime.turn_compaction import (
    COMPACTED_SUMMARY_MARKER,
    DEFAULT_MAX_TOKEN_BUDGET,
    DEFAULT_MAX_OUTPUT_CHARS,
    DEFAULT_MIN_HISTORY_FLOOR,
    MAX_CONSECUTIVE_EMPTY_TURNS,
    SystemOverheadExceededError,
    _is_immutable_system_anchor,
    estimate_tokens,
    strip_internal_metadata,
    prune_old_tool_outputs,
    summarize_oldest_history,
    compact_context_two_tier,
    check_turn_loop_circuit_breaker,
    compact_turn_history,
)

__all__ = [
    "COMPACTED_SUMMARY_MARKER",
    "DEFAULT_MAX_TOKEN_BUDGET",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "DEFAULT_MIN_HISTORY_FLOOR",
    "MAX_CONSECUTIVE_EMPTY_TURNS",
    "SystemOverheadExceededError",
    "_is_immutable_system_anchor",
    "estimate_tokens",
    "strip_internal_metadata",
    "prune_old_tool_outputs",
    "summarize_oldest_history",
    "compact_context_two_tier",
    "check_turn_loop_circuit_breaker",
    "compact_turn_history",
]
