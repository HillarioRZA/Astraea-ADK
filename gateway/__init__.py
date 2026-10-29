"""
Gateway module for LegionADK.
"""

from LegionADK.gateway.model_router import ModelRouter, UpstreamChannel
from LegionADK.gateway.cache_boundary import (
    LEGION_CACHE_BOUNDARY,
    compute_sha256_digest,
    sort_tools_alphabetically,
    split_system_prompt_cache_boundary,
    ensure_system_prompt_cache_boundary,
    assemble_cached_system_prompt,
    assemble_cached_system_prompt_from_spec,
)

from LegionADK.gateway.tool_call_repair import (
    repair_malformed_tool_call,
    parse_and_repair_json,
    extract_tool_calls,
    extract_reasoning_and_content,
)

__all__ = [
    "ModelRouter",
    "UpstreamChannel",
    "LEGION_CACHE_BOUNDARY",
    "compute_sha256_digest",
    "sort_tools_alphabetically",
    "split_system_prompt_cache_boundary",
    "ensure_system_prompt_cache_boundary",
    "assemble_cached_system_prompt",
    "assemble_cached_system_prompt_from_spec",
    "repair_malformed_tool_call",
    "parse_and_repair_json",
    "extract_tool_calls",
    "extract_reasoning_and_content",
]
