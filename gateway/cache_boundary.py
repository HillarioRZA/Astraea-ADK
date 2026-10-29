"""
Byte-Stable System Prompt Cache Boundary & Section Digest Engine untuk LegionADK.
Menegakkan pemisahan tegas antara Upper Static Prefix (di-cache) dan Lower Volatile Context (dinamis).
Bekerja 100% di memori (RAM) tanpa kueri database.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from LegionADK.core.llm_spec import LLMSpec

LEGION_CACHE_BOUNDARY = "\n<!-- LEGION_CACHE_BOUNDARY -->\n"


def compute_sha256_digest(content: str) -> str:
    """
    Menghitung hash SHA-256 heksadesimal murni di RAM untuk string atau payload teks
    secara presisi byte-level tanpa memodifikasi whitespace.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sort_tools_alphabetically(
    tools: Optional[Union[List[Union[Dict[str, Any], str]], Tuple[str, ...]]] = None
) -> Tuple[List[Any], str]:
    """
    Mengurutkan definisi skema atau daftar nama tool secara alfabetis murni di memori (RAM).
    Menjamin stabilitas urutan byte pada skema perkakas yang dikirim ke LLM agar cache tidak goyah.

    Mendukung:
    - List/Tuple of string nama tool (misal dari LLMSpec.tool_permissions: ('web_search', 'bash'))
    - List of dictionary skema tool (OpenAI / Anthropic function declarations)

    Mengembalikan:
        Tuple (sorted_tools, tool_schema_digest_sha256).
    """
    if not tools:
        empty_digest = hashlib.sha256(b"empty_tools").hexdigest()
        return [], empty_digest

    def _extract_tool_name(item: Union[Dict[str, Any], str]) -> str:
        if isinstance(item, str):
            return item.strip().lower()
        if isinstance(item, dict):
            if "function" in item and isinstance(item["function"], dict):
                return str(item["function"].get("name", "")).strip().lower()
            return str(item.get("name", item.get("tool_id", ""))).strip().lower()
        return str(item).strip().lower()

    # Urutkan list murni di RAM tanpa query database
    sorted_tools = sorted(tools, key=_extract_tool_name)

    # Serialisasi deterministik untuk menghitung SHA-256 digest (termasuk name, description, & params)
    normalized_schemas: List[str] = []
    for t in sorted_tools:
        if isinstance(t, str):
            normalized_schemas.append(f"tool_name:{t.strip()}")
        elif isinstance(t, dict):
            if "function" in t and isinstance(t["function"], dict):
                fn = t["function"]
                fn_name = fn.get("name", "")
                fn_desc = fn.get("description", "")
                fn_params = json.dumps(fn.get("parameters", {}), sort_keys=True)
                normalized_schemas.append(f"{fn_name}:{fn_desc}:{fn_params}")
            else:
                t_name = t.get("name", t.get("tool_id", ""))
                t_desc = t.get("description", "")
                t_params = json.dumps(
                    t.get("parameters", t.get("input_schema", {})),
                    sort_keys=True,
                )
                normalized_schemas.append(f"{t_name}:{t_desc}:{t_params}")

    raw_payload = ";".join(normalized_schemas).encode("utf-8")
    digest = hashlib.sha256(raw_payload).hexdigest()
    return sorted_tools, digest


def split_system_prompt_cache_boundary(
    text: str,
) -> Optional[Tuple[str, str]]:
    """
    Memisahkan system prompt pada batas tanda cache boundary di memori.

    Mengembalikan:
        Tuple (stable_prefix, dynamic_suffix) jika boundary ditemukan,
        atau None jika boundary tidak ada.
    """
    idx = text.find(LEGION_CACHE_BOUNDARY)
    if idx == -1:
        return None

    stable_prefix = text[:idx].rstrip()
    dynamic_suffix = text[idx + len(LEGION_CACHE_BOUNDARY):].lstrip()
    return stable_prefix, dynamic_suffix


def ensure_system_prompt_cache_boundary(system_prompt: str) -> str:
    """
    Memastikan system prompt memiliki tanda LEGION_CACHE_BOUNDARY di akhir jika belum ada.
    """
    clean_prompt = system_prompt.strip()
    if not clean_prompt:
        return clean_prompt
    if LEGION_CACHE_BOUNDARY in clean_prompt:
        return clean_prompt
    return f"{clean_prompt}{LEGION_CACHE_BOUNDARY}"


def assemble_cached_system_prompt(
    persona_prompt: str,
    sop_template: str,
    tools: Optional[List[Union[Dict[str, Any], str]]] = None,
    dynamic_context: str = "",
    agent_name: str = "",
    handle: str = "",
    max_token_budget: Optional[int] = None,
    max_static_ratio: float = 0.6,
) -> Tuple[str, str, str]:
    """
    Merakit system prompt berlapis dengan batas byte-stable prompt cache.
    Seluruh penghitungan hash SHA-256 dan pengurutan tools dilakukan murni di RAM.

    UPPER SECTION (100% Byte-Stable & Di-cache oleh LLM Provider):
      - Role Identifier (@handle & nama agen)
      - Layer 1: Immutable Soul (persona_prompt)
      - Layer 2: Static Operational SOP (sop_template)
      - Authorized Tools (skema/nama tools diurutkan alfabetis di RAM)

    BOUNDARY:
      <!-- LEGION_CACHE_BOUNDARY -->

    LOWER SECTION (Volatile Dynamic Turn Context):
      - Layer 3: Konteks dinamis per-turn, instruksi task, riwayat percakapan.

    Mengembalikan:
      Tuple (full_assembled_prompt, system_prompt_digest, tool_schema_digest).
    """
    # 1. Hitung tool digest murni di memori
    sorted_tools, tool_digest = sort_tools_alphabetically(tools)

    # 2. Bangun Upper Static Prefix (bagian yang terkunci di cache)
    upper_parts: List[str] = []

    if agent_name or handle:
        upper_parts.append(f"=== ROLE IDENTIFIER: {agent_name} ({handle}) ===".strip())
        upper_parts.append("")

    upper_parts.extend([
        "=== LAYER 1: IMMUTABLE SOUL & PERSONA ===",
        persona_prompt.strip(),
        "",
        "=== LAYER 2: STATIC OPERATIONAL SOP & RULES ===",
        sop_template.strip(),
    ])

    if sorted_tools:
        tool_descriptions: List[str] = []
        for t in sorted_tools:
            if isinstance(t, str):
                tool_descriptions.append(f"- `{t}`")
            elif isinstance(t, dict):
                fn_obj = t.get("function") if isinstance(t.get("function"), dict) else t
                t_name = fn_obj.get("name", fn_obj.get("tool_id", ""))
                t_desc = fn_obj.get("description", "")
                if t_desc:
                    tool_descriptions.append(f"- `{t_name}`: {t_desc}")
                else:
                    tool_descriptions.append(f"- `{t_name}`")
        upper_parts.append("")
        upper_parts.append("=== AUTHORIZED TOOLS (DETERMINISTIC IN-MEMORY ORDER) ===")
        upper_parts.append("\n".join(tool_descriptions))

    stable_prefix = "\n".join(upper_parts).strip()

    # Guard Lapisan 1: validasi alokasi overhead static system prompt
    if max_token_budget is not None:
        from LegionADK.runtime.turn_compaction import SystemOverheadExceededError, estimate_tokens
        prefix_tokens = estimate_tokens(stable_prefix)
        allowed_tokens = int(max_token_budget * max_static_ratio)
        if prefix_tokens > allowed_tokens:
            raise SystemOverheadExceededError(
                f"Static system prompt overhead ({prefix_tokens} tokens) melampaui batas alokasi "
                f"statis {int(max_static_ratio * 100)}% dari budget ({allowed_tokens}/{max_token_budget} tokens). "
                f"Kurangi panjang persona_prompt, SOP, atau skema tools."
            )

    # Penghitung hash SHA-256 system_prompt_digest murni di RAM
    system_prompt_digest = compute_sha256_digest(stable_prefix)

    # 3. Rakit dengan Lower Dynamic Section
    if dynamic_context.strip():
        lower_section = f"=== DYNAMIC TASK CONTEXT & INSTRUCTIONS ===\n{dynamic_context.strip()}"
        full_prompt = f"{stable_prefix}{LEGION_CACHE_BOUNDARY}{lower_section}"
    else:
        full_prompt = ensure_system_prompt_cache_boundary(stable_prefix)

    return full_prompt, system_prompt_digest, tool_digest


def assemble_cached_system_prompt_from_spec(
    spec: LLMSpec,
    dynamic_context: str = "",
    max_token_budget: Optional[int] = None,
    max_static_ratio: float = 0.6,
) -> Tuple[str, str, str]:
    """
    Helper untuk merakit cached system prompt langsung dari objek LLMSpec.
    Menjadikan spec.compile_identity() sebagai sumber kanonis tunggal,
    sehingga digest SHA-256 identik 100% dengan compile_identity.
    """
    stable_prefix = spec.compile_identity()

    # Guard Lapisan 1: validasi alokasi overhead static system prompt
    if max_token_budget is not None:
        from LegionADK.runtime.turn_compaction import SystemOverheadExceededError, estimate_tokens
        prefix_tokens = estimate_tokens(stable_prefix)
        allowed_tokens = int(max_token_budget * max_static_ratio)
        if prefix_tokens > allowed_tokens:
            raise SystemOverheadExceededError(
                f"Static system prompt overhead ({prefix_tokens} tokens) melampaui batas alokasi "
                f"statis {int(max_static_ratio * 100)}% dari budget ({allowed_tokens}/{max_token_budget} tokens). "
                f"Kurangi panjang persona_prompt, SOP, atau skema tools."
            )

    system_prompt_digest = compute_sha256_digest(stable_prefix)
    _, tool_digest = sort_tools_alphabetically(spec.tool_permissions)

    if dynamic_context.strip():
        lower_section = f"=== DYNAMIC TASK CONTEXT & INSTRUCTIONS ===\n{dynamic_context.strip()}"
        full_prompt = f"{stable_prefix}{LEGION_CACHE_BOUNDARY}{lower_section}"
    else:
        full_prompt = ensure_system_prompt_cache_boundary(stable_prefix)

    return full_prompt, system_prompt_digest, tool_digest


__all__ = [
    "LEGION_CACHE_BOUNDARY",
    "compute_sha256_digest",
    "sort_tools_alphabetically",
    "split_system_prompt_cache_boundary",
    "ensure_system_prompt_cache_boundary",
    "assemble_cached_system_prompt",
    "assemble_cached_system_prompt_from_spec",
]
