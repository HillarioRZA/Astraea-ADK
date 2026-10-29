"""
LegionADK Tools Subsystem.
Includes Tool Registry, RBAC Authorization Guard, and Universal 50KB Output Spill Guard.
"""

from LegionADK.tools.permission_registry import (
    ToolSpec,
    ToolRegistryError,
    ToolPermissionError,
    ToolNotFoundError,
    ToolArgumentError,
    ToolExecutionError,
    register_tool,
    get_registered_tool,
    list_registered_tools,
    unregister_tool,
    clear_tool_registry,
    authorize_tool_call,
    export_permitted_tools,
    execute_registered_tool,
)

from LegionADK.tools.output_spill_guard import (
    DEFAULT_MAX_OUTPUT_BYTES,
    TRUNCATION_MARKER,
    DEFAULT_SPILL_DIR_NAME,
    SpillResult,
    truncate_utf8_bytes,
    process_output,
    process_output_sync,
    guard_tool_output,
    guard_tool_output_sync,
    read_spilled_output,
    read_spilled_output_sync,
    prune_spill_cache,
    prune_spill_cache_sync,
)

__all__ = [
    # Permission Registry & RBAC Guard
    "ToolSpec",
    "ToolRegistryError",
    "ToolPermissionError",
    "ToolNotFoundError",
    "ToolArgumentError",
    "ToolExecutionError",
    "register_tool",
    "get_registered_tool",
    "list_registered_tools",
    "unregister_tool",
    "clear_tool_registry",
    "authorize_tool_call",
    "export_permitted_tools",
    "execute_registered_tool",
    # Output Spill Guard
    "DEFAULT_MAX_OUTPUT_BYTES",
    "TRUNCATION_MARKER",
    "DEFAULT_SPILL_DIR_NAME",
    "SpillResult",
    "truncate_utf8_bytes",
    "process_output",
    "process_output_sync",
    "guard_tool_output",
    "guard_tool_output_sync",
    "read_spilled_output",
    "read_spilled_output_sync",
    "prune_spill_cache",
    "prune_spill_cache_sync",
]
